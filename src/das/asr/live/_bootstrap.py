"""セッション初期化・起動ロジック.

live.py (エントリーポイント) から呼ばれ、STTバックエンド構築・
声紋モデル読み込み・スレッド起動・受信ループ実行までを担う。
CLI引数の定義は live.py 側 (click) に残し、ここではパース済みの
値だけを受け取る。
"""
from __future__ import annotations

import datetime
import json
import os
import threading
from dataclasses import dataclass

from das.asr.live._constants import _TOPIC_PROMPT, OPENAI_API, SR
from das.asr.live._recv_loop import RecvLoop
from das.asr.live._session_state import SessionState
from das.asr.live._ui import _UIHandler
from das.asr.live._voice_profiles import VoiceProfiles
from das.asr.live._workers import (
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
from das.asr.live.agents._partner import ConversationPartner
from das.asr.live.agents._realtime import RealtimeAgent
from das.asr.live.agents._simulator import DiscussionSimulator
from das.asr.live.stt import STTBackend
from das.asr.live.stt._soniox import SonioxBackend
from das.asr.live.stt._speechmatics import SpeechmaticsBackend

# ---------------------------------------------------------------------------
# CLI引数をまとめるデータクラス（argparse.Namespace の代替）
# ---------------------------------------------------------------------------

@dataclass
class LiveArgs:
    """live.py の click オプションを内部に渡すための構造体."""
    lang: str = "ja"
    model: str = "stt-rt-v4"
    wav: str | None = None
    play: bool = False
    join: bool = False
    device: str | None = None
    out: str | None = None
    no_open: bool = False
    no_vp: bool = False
    voices: str = "voices.json"
    vp_model: str = "redimnet"
    vp_match: float | None = None
    vp_no_auto: bool = False
    vp_debug: bool = False
    polish: bool = False
    stt: str = "soniox"
    port: int = 8231
    agent: bool = False
    agent_voice: str = "shimmer"
    agent_trigger: int = 10
    simulate: str | None = None
    sim_scenario: str | None = None
    debate: str | None = None
    debate_voice: str = "echo"


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

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


def build_backend(args: LiveArgs) -> STTBackend:
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


def extract_topics(utterances: list[dict], existing: list[str],
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
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# メインのセッション起動
# ---------------------------------------------------------------------------

def run_session(args: LiveArgs, *, on_utterance_ref: list) -> None:
    """セッションを初期化し、STT受信ループを実行する.

    on_utterance_ref は [callable | None] の1要素リスト。
    live.py の ON_UTTERANCE をセッション内から参照するために使う。
    """
    from das.asr.live import _SYS_HOOK_REF  # 遅延import（循環回避）

    load_env()
    if args.wav and not os.path.exists(args.wav):
        raise SystemExit(f"音声ファイルがありません: {args.wav}\n"
                         "（テスト音声は scripts/make_overlap_testset.py 等で先に生成してください）")

    backend = build_backend(args)
    api_key = os.environ.get("SONIOX_API_KEY")  # polish用（STTバックエンドとは独立）
    _serve = args.port > 0

    try:
        from websockets.sync.client import connect
    except ImportError as exc:
        raise SystemExit("uv add websockets を実行してください") from exc

    started = datetime.datetime.now()
    if args.out:
        out_path = args.out
    else:
        os.makedirs("transcripts", exist_ok=True)
        out_path = os.path.join("transcripts", started.strftime("%Y-%m-%d_%H%M") + ".md")
    html_path = os.path.splitext(out_path)[0] + ".html"
    diag_path = os.path.splitext(out_path)[0] + ".diag.jsonl"
    turns_path = os.path.splitext(out_path)[0] + ".turns.jsonl"

    # --- 声紋モデル読み込み ---
    tracker: VoiceProfiles | None = None
    if not args.no_vp:
        print("# 声紋モデルを読み込み中…", flush=True)
        for vp_model in dict.fromkeys([args.vp_model, "ecapa", "resemblyzer"]):
            try:
                tracker = VoiceProfiles(path=args.voices, thresh=args.vp_match,
                                        auto=not args.vp_no_auto, model=vp_model)
                if vp_model != args.vp_model:
                    print(f"# 注意: {args.vp_model} を読み込めなかったため {vp_model} で動作します"
                          f"（依存: uv add speechbrain torchaudio / redimnetは初回ネット接続必要）",
                          flush=True)
                print(f"# 声紋モデル: {vp_model}", flush=True)
                break
            except Exception as e:
                print(f"#   {vp_model}: 読み込み失敗 ({type(e).__name__})", flush=True)
                continue
        if tracker is None:
            print("# 警告: 声紋照合がOFFです！ 依存が未導入のため人物の確定・補正は行われません。", flush=True)
            print("#   有効化するには: uv add speechbrain torchaudio  →  再起動", flush=True)
        elif tracker.profiles:
            print(f"# 声紋プロファイル: {', '.join(tracker.profiles)}（{args.voices}）", flush=True)
        else:
            print(f"# 声紋プロファイル: なし。未知の声は「人物N」として自動追跡、"
                  f"「1=松井」で実名化すると次回から自動表示（{args.voices}）", flush=True)

    # --- SessionState ---
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

    # --- WAVストリーミング書き出し ---
    try:
        state.pcm_file = open(wav_path, "wb")  # noqa: SIM115
        import struct as _struct
        state.pcm_file.write(b"RIFF" + _struct.pack("<I", 0) + b"WAVEfmt " +
                              _struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16) +
                              b"data" + _struct.pack("<I", 0))
        state.pcm_file.flush()
    except OSError as e:
        print(f"# 警告: 録音ファイルを開けません: {e}", flush=True)
        state.pcm_file = None

    def _sys_hook(text: str) -> None:
        state.add_sys(None, text)
        state.save()
    _SYS_HOOK_REF[0] = _sys_hook

    # --- 論点抽出 ---
    _oai_key = os.environ.get("OPENAI_API_KEY", "")
    _oai_model = os.environ.get("OPENAI_MODEL_FAST", "gpt-5-mini")

    # --- AIエージェント: コールバック ---
    _on_agent_text = _on_agent_text_factory(state)

    # --- UIサーバー ---
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

    # --- ConversationPartner ---
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
