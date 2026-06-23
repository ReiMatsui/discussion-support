"""リアルタイム議事録ツール — エントリーポイント + STT接続 + 論点抽出.

STTバックエンド（Soniox/Speechmatics/…）経由で音声をストリーミングし、
声紋話者分離・AI ファシリテーション・議事録保存を統合する。

das連携フック:
  ON_UTTERANCE に callable(speaker:str, text:str) を設定 →
  確定発話ごとに呼ばれる（cli.py がオーケストレータへ流すのに使用）。

主要モジュール構成:
  _stt_backend.py     STTBackend Protocol
  _stt_soniox.py      Soniox 実装
  _stt_speechmatics.py Speechmatics 実装
  _recv_loop.py       WebSocket受信 + flush（声紋判定・エコー除去）
  _workers.py         音声入力・送信・トピック抽出・ターンテイキング
  _session_state.py   共有状態 + ファイル出力
  _voice_profiles.py  声紋プロファイル
  _realtime_agent.py  OpenAI Realtime API ファシリテーター
  _conversation_partner.py  Realtime API 議論パートナー
  _discussion_simulator.py  Chat+TTS シミュレーション
  _polish.py          非同期バッチ再処理（清書）
  _ui.py              HTTPサーバー + ターミナル出力
  _constants.py       定数・プロンプト・HTML テンプレート
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import threading

from das.asr._constants import (
    _AGENT_TRIGGER,
    OPENAI_API,
    SR,
    _TOPIC_PROMPT,
)
from das.asr._stt_backend import STTBackend
from das.asr._stt_soniox import SonioxBackend
from das.asr._stt_speechmatics import SpeechmaticsBackend
from das.asr._conversation_partner import ConversationPartner
from das.asr._discussion_simulator import DiscussionSimulator
from das.asr._realtime_agent import RealtimeAgent
from das.asr._voice_profiles import VoiceProfiles
from das.asr._session_state import SessionState
from das.asr._recv_loop import RecvLoop
from das.asr._ui import _UIHandler
from das.asr._workers import (
    _cleanup,
    _connect_agent,
    _on_agent_text_factory,
    _on_partner_text_factory,
    _run_from_mic,
    _run_from_wav,
    _run_sender,
    _run_stdin_commands,
    _run_topic_worker,
)

ON_UTTERANCE = None   # das連携: 確定発話ごとに (話者表示名, テキスト) で呼ばれる
_SYS_HOOK = None      # main()実行中のみ登録される(add_sys+saveへの橋)


def post_system(text: str) -> None:
    """das連携: ライブ議事録のタイムラインにシステム行(💡介入など)を外部から追加する."""
    if _SYS_HOOK is not None:
        _SYS_HOOK(text)


def _build_backend(args) -> STTBackend:
    """CLIオプションに基づいてSTTバックエンドを構築する."""
    if args.stt == "speechmatics":
        sm_key = os.environ.get("SPEECHMATICS_API_KEY")
        if not sm_key:
            raise SystemExit(
                "環境変数 SPEECHMATICS_API_KEY を設定してください"
                "（https://portal.speechmatics.com/settings/api-keys）")
        return SpeechmaticsBackend(api_key=sm_key)
    else:
        api_key = os.environ.get("SONIOX_API_KEY")
        if not api_key:
            raise SystemExit(
                "環境変数 SONIOX_API_KEY を設定してください"
                "（https://console.soniox.com）")
        return SonioxBackend(api_key=api_key)


def load_env(path: str = ".env") -> None:
    """プロジェクト直下の .env からAPIキー等を読み込む（既に設定済みの環境変数を優先）.

    形式: KEY=VALUE の行（#始まりはコメント）。依存なしの最小実装。
    """
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


# ---------- 論点抽出（非同期LLM処理） ----------


def _extract_topics(utterances: list[dict], existing: list[str],
                    api_key: str, model: str) -> list[dict]:
    """OpenAI APIで新論点を抽出する（同期呼び出し、バックグラウンドスレッド用）."""
    if not utterances or not api_key:
        return []
    utt_text = "\n".join(f"- {u['speaker']}: {u['text']}" for u in utterances)
    ex_text = "\n".join(f"- {t}" for t in existing) if existing else "（まだなし）"
    prompt = _TOPIC_PROMPT.format(existing=ex_text, utterances=utt_text)
    # GPT-5系/o系はtemperature指定不可、max_tokensはmax_completion_tokensに改名
    name = model.lower()
    is_new = name.startswith(("gpt-5", "o1", "o3", "o4"))
    params: dict = {"model": model,
                    "messages": [{"role": "user", "content": prompt}]}
    if not is_new:
        params["temperature"] = 0.3
        params["max_tokens"] = 512
    else:
        params["max_completion_tokens"] = 512
    body = json.dumps(params).encode()
    import urllib.request
    req = urllib.request.Request(OPENAI_API, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read())
        text = resp["choices"][0]["message"]["content"].strip()
        # JSON配列を抽出（前後にmarkdownコードブロックがある場合も対応）
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except Exception:
        return []



def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--model", default="stt-rt-v4")
    ap.add_argument("--wav", default=None, help="指定で実マイクの代わりにファイル擬似ライブ")
    ap.add_argument("--play", action="store_true",
                    help="--wav使用時、注入と同時にスピーカーからも再生する（観戦用）")
    ap.add_argument("--join", action="store_true",
                    help="--wav使用時、再生しつつ自分のマイクも混ぜて参加する（イヤホン推奨。"
                         "wav終了後もマイクは生き続けるのでCtrl+Cで終了）")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="保存先mdファイル（省略時 transcripts/日時.md）")
    ap.add_argument("--no-open", action="store_true", help="ブラウザを自動で開かない")
    ap.add_argument("--no-vp", action="store_true", help="声紋照合を無効化（Sonioxのラベルをそのまま使う）")
    ap.add_argument("--voices", default="voices.json", help="声紋プロファイルの保存先(既定 voices.json)")
    ap.add_argument("--vp-model", default="redimnet", choices=["redimnet", "ecapa", "resemblyzer"],
                    help="声紋モデル(既定redimnet=2024年世代、実測の分離・通し精度とも最良。"
                         "読み込み失敗時は ecapa → resemblyzer へ自動フォールバック)")
    ap.add_argument("--vp-match", type=float, default=None,
                    help="即時判定のしきい値。省略時はモデル別の既定値"
                         "(redimnet 0.42 / ecapa 0.35 / resemblyzer 0.75)")
    ap.add_argument("--vp-no-auto", action="store_true",
                    help="未知の声の自動登録（匿名「人物N」）を無効化")
    ap.add_argument("--vp-debug", action="store_true", help="発話ごとの声紋判定の内訳を表示")
    ap.add_argument("--polish", action="store_true",
                    help="終了時に清書を行う（非同期APIでの全体再処理。デフォルトオフ）")
    ap.add_argument("--no-polish", action="store_true",
                    help="(後方互換用、現在はデフォルトでオフ)")
    ap.add_argument("--stt", default="soniox", choices=["soniox", "speechmatics"],
                    help="リアルタイムSTTの供給源。speechmaticsは要 SPEECHMATICS_API_KEY"
                         "（話者分離の評判が良い代替。声紋層など他の機能は不変）")
    ap.add_argument("--port", type=int, default=8231,
                    help="UIサーバーのポート番号（ブラウザからの話者リネームに必要。0で無効）")
    ap.add_argument("--agent", action="store_true",
                    help="AIエージェント（ファシリテーター）を有効化。OPENAI_API_KEYが必要。"
                         "Realtime API v2 WebSocketで会議に参加する")
    ap.add_argument("--agent-voice", default="shimmer",
                    help="AIエージェントの声（alloy/ash/ballad/coral/echo/sage/shimmer/verse）")
    ap.add_argument("--agent-trigger", type=int, default=_AGENT_TRIGGER,
                    help=f"AIの応答を検討する発話間隔（既定{_AGENT_TRIGGER}）")
    ap.add_argument("--simulate", metavar="TOPIC",
                    help="AI議論シミュレーション。Chat API+TTSで複数話者の議論を自動生成し、"
                         "ファシリテーターが介入する。--agentと組み合わせて使用。"
                         "例: --simulate 'AIツール導入の是非'")
    ap.add_argument("--sim-scenario", default=None,
                    choices=["stalled", "biased", "derailed", "consensus_needed", "healthy"],
                    help="シミュレーションの議論パターン")
    ap.add_argument("--debate", metavar="TOPIC",
                    help="AI会話相手と議論。Realtime APIで音声対話し、"
                         "ファシリテーターが介入する。--agentと組み合わせて使用。"
                         "例: --debate 'AIツール導入の是非'")
    ap.add_argument("--debate-voice", default="echo",
                    help="会話相手の声（既定echo。ファシリテーターのalloyと被らないこと）")
    args = ap.parse_args(argv)
    _serve = args.port > 0

    load_env()   # .env からAPIキーを読み込み（export済みの値が優先）
    if args.wav and not os.path.exists(args.wav):
        raise SystemExit(f"音声ファイルがありません: {args.wav}\n"
                         "（テスト音声は scripts/make_overlap_testset.py 等で先に生成してください）")

    backend = _build_backend(args)
    api_key = os.environ.get("SONIOX_API_KEY")  # polish用（STTバックエンドとは独立）

    try:
        from websockets.sync.client import connect
    except ImportError:
        raise SystemExit("uv add websockets を実行してください")

    started = datetime.datetime.now()
    if args.out:
        out_path = args.out
    else:
        os.makedirs("transcripts", exist_ok=True)
        out_path = os.path.join("transcripts", started.strftime("%Y-%m-%d_%H%M") + ".md")
    html_path = os.path.splitext(out_path)[0] + ".html"
    diag_path = os.path.splitext(out_path)[0] + ".diag.jsonl"   # 発話ごとの判定根拠(劣化解析用)
    turns_path = os.path.splitext(out_path)[0] + ".turns.jsonl"  # das(議論支援)連携用

    # --- 声紋モデル読み込み ---
    tracker: VoiceProfiles | None = None
    if not args.no_vp:
        print("# 声紋モデルを読み込み中…", flush=True)
        for model in dict.fromkeys([args.vp_model, "ecapa", "resemblyzer"]):
            try:
                tracker = VoiceProfiles(path=args.voices, thresh=args.vp_match,
                                        auto=not args.vp_no_auto, model=model)
                if model != args.vp_model:
                    print(f"# 注意: {args.vp_model} を読み込めなかったため {model} で動作します"
                          f"（依存: uv add speechbrain torchaudio / redimnetは初回ネット接続必要）",
                          flush=True)
                print(f"# 声紋モデル: {model}", flush=True)
                break
            except Exception as e:   # 依存欠如(ImportError)もDL失敗等も次の候補へ
                print(f"#   {model}: 読み込み失敗 ({type(e).__name__})", flush=True)
                continue
        if tracker is None:
            print("# 警告: 声紋照合がOFFです！ 依存が未導入のため人物の確定・補正は行われません。", flush=True)
            print("#   有効化するには: uv add speechbrain torchaudio  →  再起動", flush=True)
        elif tracker.profiles:
            print(f"# 声紋プロファイル: {', '.join(tracker.profiles)}（{args.voices}）", flush=True)
        else:
            print(f"# 声紋プロファイル: なし。未知の声は「人物N」として自動追跡、"
                  f"「1=松井」で実名化すると次回から自動表示（{args.voices}）", flush=True)

    # --- SessionState: 共有状態の一括管理 ---
    wav_path = os.path.splitext(out_path)[0] + ".wav"
    state = SessionState(args=args, started=started, out_path=out_path,
                         html_path=html_path, diag_path=diag_path,
                         turns_path=turns_path, wav_path=wav_path,
                         tracker=tracker, serve=_serve)

    # --- AIエージェント ---
    _agent_oai_key = os.environ.get("OPENAI_API_KEY", "")
    if args.agent:
        if not _agent_oai_key:
            print("# AI Agent: OPENAI_API_KEY が未設定です。--agent は無効になります。", flush=True)
        else:
            state.agent = RealtimeAgent(api_key=_agent_oai_key, voice=args.agent_voice,
                                        mode="facilitator", trigger_n=args.agent_trigger)
            if tracker is not None:
                state.agent.set_tracker(tracker)

    # --- WAVストリーミング書き出し（クラッシュ時もファイルが残る） ---
    try:
        state.pcm_file = open(wav_path, "wb")
        import struct as _struct
        state.pcm_file.write(b"RIFF" + _struct.pack("<I", 0) + b"WAVEfmt " +
                              _struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16) +
                              b"data" + _struct.pack("<I", 0))
        state.pcm_file.flush()
    except OSError as e:
        print(f"# 警告: 録音ファイルを開けません: {e}", flush=True)
        state.pcm_file = None

    global _SYS_HOOK

    def _sys_hook(text: str) -> None:
        state.add_sys(None, text)
        state.save()
    _SYS_HOOK = _sys_hook

    # --- 論点抽出 ---
    _oai_key = os.environ.get("OPENAI_API_KEY", "")
    _oai_model = os.environ.get("OPENAI_MODEL_FAST", "gpt-5-mini")

    # --- AIエージェント: コールバック ---
    _on_agent_text = _on_agent_text_factory(state)

    # --- UIサーバー（ブラウザからの話者リネーム用）---
    _httpd = None
    if _serve:
        from http.server import HTTPServer
        try:
            _httpd = HTTPServer(("127.0.0.1", args.port), _UIHandler.create(state))
            threading.Thread(target=_httpd.serve_forever, daemon=True).start()
        except OSError as e:
            print(f"# 警告: UIサーバーをポート{args.port}で起動できません ({e})", flush=True)
            _serve = False
            state._serve = False


    if args.simulate and args.debate:
        raise SystemExit("--simulate と --debate は同時に使えません")

    # --- DiscussionSimulator ---
    if args.simulate:
        if not _oai_key:
            raise SystemExit("--simulate には OPENAI_API_KEY が必要です")
        if not args.agent:
            print("# ヒント: --agent を付けるとファシリテーターが介入します", flush=True)
        if args.agent and args.agent_voice in DiscussionSimulator.SPEAKERS.values():
            print(f"# 警告: --agent-voice={args.agent_voice} はSimulator話者と重複しています。"
                  f"声紋分離に影響する可能性があります。alloy を推奨します。", flush=True)
        state.simulator = DiscussionSimulator(
            api_key=_oai_key, topic=args.simulate,
            scenario=args.sim_scenario)
    # --- ConversationPartner（--debate モード）---
    if args.debate:
        if not _oai_key:
            raise SystemExit("--debate には OPENAI_API_KEY が必要です")
        if not args.agent:
            print("# ヒント: --agent を付けるとファシリテーターが介入します", flush=True)
        if args.agent and args.debate_voice == args.agent_voice:
            print(f"# 警告: --debate-voice と --agent-voice が同じ ({args.debate_voice})。"
                  f"声紋分離に影響します。", flush=True)
        state.partner = ConversationPartner(
            api_key=_oai_key, voice=args.debate_voice, topic=args.debate)
        if tracker is not None:
            state.partner.set_tracker(tracker)

    print(f"# {backend.name} に接続中…", flush=True)
    with connect(backend.ws_url(), additional_headers=backend.ws_headers()) as ws:
        ws.send(json.dumps(backend.start_message(args.model, args.lang)))
        # 音声ソース選択: simulate > wav > mic
        if state.simulator is not None:
            if state.agent is not None:
                state.simulator._agent_ref = state.agent
            state.simulator.start(state.audio_q, state.stop, play_audio=True)
            print(f"# Simulator: 議論を自動生成中（議題: {args.simulate}）", flush=True)
        else:
            if args.wav:
                threading.Thread(target=_run_from_wav, args=(state, args),
                                 daemon=True).start()
            else:
                threading.Thread(target=_run_from_mic, args=(state, args.device),
                                 daemon=True).start()
        threading.Thread(target=_run_stdin_commands, args=(state,),
                         daemon=True).start()
        if _oai_key:
            threading.Thread(target=_run_topic_worker,
                            args=(state, _oai_key, _oai_model), daemon=True).start()
            print("# 論点抽出: 有効（5発話ごとにLLMで分析）", flush=True)
        else:
            print("# 論点抽出: 無効（OPENAI_API_KEYが未設定）", flush=True)
        if state.agent is not None:
            _connect_agent(state, _on_agent_text)
        if state.partner is not None:
            state.partner.on_ai_utterance = _on_partner_text_factory(state)
            state.partner.connect()
            print(f"# Partner: voice={state.partner.voice} topic={state.partner.topic}",
                  flush=True)

        threading.Thread(target=_run_sender, args=(state, ws, backend),
                         daemon=True).start()

        state.save()
        print("# 開始。話してください（「1=松井」で声を登録 / Ctrl+Cで終了）", flush=True)
        print(f"# 保存先: {out_path}", flush=True)
        print(f"# ブラウザ表示: open {html_path}（ライブ中は2秒ごと自動更新）\n", flush=True)
        if not args.no_open:
            import webbrowser
            if _serve:
                webbrowser.open(f"http://127.0.0.1:{args.port}/")
            else:
                webbrowser.open("file://" + os.path.abspath(html_path))

        recv = RecvLoop(state, args, backend)
        try:
            recv.run(ws)
        finally:
            _cleanup(state, args, api_key, tracker, wav_path, out_path, html_path)


if __name__ == "__main__":
    main()
