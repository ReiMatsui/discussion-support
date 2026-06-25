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

from das.asr.live._assemblyai_diarization import AssemblyAIStreamingDiarizationProvider
from das.asr.live._constants import (
    _AGENDA_PROMPT,
    _DRIFT_PROMPT,
    _PARTICIPATION_PROMPT,
    _TOPIC_PROMPT,
    OPENAI_API,
)
from das.asr.live._diarization import SpeakerResolver
from das.asr.live._pyannote_diarization import PyannoteStreamingDiarizationProvider
from das.asr.live._recv_loop import RecvLoop
from das.asr.live._session_state import SessionState
from das.asr.live._ui import _UIHandler
from das.asr.live._voice_profiles import VoiceProfiles
from das.asr.live._workers import (
    _cleanup,
    _connect_agent,
    _on_agent_text_factory,
    _on_partner_text_factory,
    _run_agenda_detector,
    _run_drift_checker,
    _run_from_mic,
    _run_from_wav,
    _run_participation_checker,
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
    # Sonioxのエンドポイント検出（文の切れ目で区切る＝議事録が読みやすい）。既定ON。
    soniox_endpoint: bool = True
    diarization: str = "none"  # none / pyannote / assemblyai
    diarization_max_speakers: int | None = None
    port: int = 8231
    agent: bool = False
    agent_voice: str = "shimmer"
    agent_trigger: int = 10
    simulate: str | None = None
    sim_scenario: str | None = None
    debate: str | None = None
    debate_voice: str = "echo"
    topic: str | None = None   # 人間同士モードの議題（脱線判定の基準）
    proactivity: str = "standard"  # 介入の積極性（controlled/standard/active）


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
        return SpeechmaticsBackend(
            api_key=sm_key,
            max_speakers=args.diarization_max_speakers,
        )
    else:
        api_key = os.environ.get("SONIOX_API_KEY")
        if not api_key:
            raise SystemExit(
                "環境変数 SONIOX_API_KEY を設定してください"
                "（https://console.soniox.com）")
        return SonioxBackend(
            api_key=api_key,
            enable_endpoint_detection=getattr(args, "soniox_endpoint", False))


def _build_chat_params(model: str, prompt: str, *, max_out: int,
                       temperature: float) -> dict:
    """Chat Completions のリクエストパラメータを構築する（モデル系統別）.

    gpt-5系/o系は推論モデルで、temperature指定不可・max_tokensは
    max_completion_tokensに改名。さらに推論トークンが出力枠を消費するため、
    枠が小さいと本文(JSON)が空のまま返る → 静かに失敗する（Fix 9の根因）。
    対策として reasoning_effort を抑え、出力枠を十分に確保する。
    """
    name = model.lower()
    params: dict = {"model": model,
                    "messages": [{"role": "user", "content": prompt}]}
    if name.startswith("gpt-5"):
        params["reasoning_effort"] = "minimal"  # 短いJSON抽出に推論は不要
        params["max_completion_tokens"] = max_out
    elif name.startswith(("o1", "o3", "o4")):
        params["reasoning_effort"] = "low"       # o系は minimal 非対応
        params["max_completion_tokens"] = max_out
    else:
        params["temperature"] = temperature
        params["max_tokens"] = max_out
    return params


def _post_chat_json(params: dict, api_key: str, *, timeout: int, label: str):
    """Chat Completions を叩き、本文をJSONとして返す。失敗時はNone（理由をログ）."""
    import urllib.request
    body = json.dumps(params).encode()
    req = urllib.request.Request(OPENAI_API, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read())
        text = (resp["choices"][0]["message"].get("content") or "").strip()
        if not text:
            # 空応答（推論で出力枠を使い切った等）— 静かに失敗させず可視化する
            finish = resp["choices"][0].get("finish_reason")
            print(f"# [{label}] 空応答（finish_reason={finish}）。"
                  f"max_completion_tokens不足の可能性", flush=True)
            return None
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except Exception as e:
        print(f"# [{label}] API/解析エラー: {type(e).__name__}: {e}", flush=True)
        return None


def extract_topics(utterances: list[dict], existing: list[str],
                   api_key: str, model: str) -> list[dict]:
    """OpenAI APIで新論点を抽出する（同期呼び出し、バックグラウンドスレッド用）."""
    if not utterances or not api_key:
        return []
    utt_text = "\n".join(f"- {u['speaker']}: {u['text']}" for u in utterances)
    ex_text = "\n".join(f"- {t}" for t in existing) if existing else "（まだなし）"
    prompt = _TOPIC_PROMPT.format(existing=ex_text, utterances=utt_text)
    params = _build_chat_params(model, prompt, max_out=2000, temperature=0.3)
    result = _post_chat_json(params, api_key, timeout=30, label="topic")
    return result if isinstance(result, list) else []


def check_drift(utterances: list[dict], topics: list[dict],
                api_key: str, model: str) -> dict:
    """論点からの脱線を判定する（同期呼び出し、バックグラウンドスレッド用）.

    Returns:
        {"drift": bool, "reason": str} or {"drift": False} on error.
    """
    if not utterances or not topics or not api_key:
        return {"drift": False}
    utt_text = "\n".join(f"- {u['speaker']}: {u['text']}" for u in utterances)
    topic_text = "\n".join(f"- {t['topic']}" for t in topics)
    prompt = _DRIFT_PROMPT.format(topics=topic_text, utterances=utt_text)
    params = _build_chat_params(model, prompt, max_out=800, temperature=0.0)
    result = _post_chat_json(params, api_key, timeout=15, label="drift")
    if not isinstance(result, dict):
        return {"drift": False}
    print(f"# [drift] 判定結果: {result}", flush=True)
    return result


def detect_agenda(utterances: list[dict], api_key: str, model: str) -> str | None:
    """会議冒頭の発話から議題を1回推定する（S3）.

    Returns: 議題の文字列 / 判断できなければ None。
    """
    if not utterances or not api_key:
        return None
    utt_text = "\n".join(f"- {u['speaker']}: {u['text']}" for u in utterances)
    prompt = _AGENDA_PROMPT.format(utterances=utt_text)
    params = _build_chat_params(model, prompt, max_out=400, temperature=0.0)
    result = _post_chat_json(params, api_key, timeout=20, label="agenda")
    if not isinstance(result, dict):
        return None
    agenda = result.get("agenda")
    if isinstance(agenda, str) and agenda.strip():
        return agenda.strip()
    return None


def check_participation(participation: list[dict], utterances: list[dict],
                        api_key: str, model: str) -> dict:
    """発話量の偏りから、誰かに声かけすべきか判定する（S4）.

    participation: [{"speaker": 名前, "time_share": 0-1, "turns": int,
                     "silent_sec": float}, ...]
    Returns: {"invite": bool, "speaker": 名前|None, "reason": str}
    """
    if not participation or not api_key:
        return {"invite": False}
    part_text = "\n".join(
        f"- {p['speaker']}: 発話時間{p['time_share'] * 100:.0f}% / "
        f"{p['turns']}回 / 最終発言{p['silent_sec']:.0f}秒前"
        for p in participation)
    utt_text = "\n".join(f"- {u['speaker']}: {u['text']}" for u in utterances)
    prompt = _PARTICIPATION_PROMPT.format(participation=part_text,
                                          utterances=utt_text)
    params = _build_chat_params(model, prompt, max_out=400, temperature=0.0)
    result = _post_chat_json(params, api_key, timeout=15, label="invite")
    if not isinstance(result, dict):
        return {"invite": False}
    return result


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
    diarizer = None
    if args.diarization == "pyannote":
        pyannote_key = os.environ.get("PYANNOTEAI_API_KEY")
        if not pyannote_key:
            raise SystemExit("環境変数 PYANNOTEAI_API_KEY を設定してください")
        diarizer = PyannoteStreamingDiarizationProvider(pyannote_key)
        print("# 話者分離: pyannoteAI streaming を使用", flush=True)
    elif args.diarization == "assemblyai":
        assemblyai_key = os.environ.get("ASSEMBLYAI_API_KEY")
        if not assemblyai_key:
            raise SystemExit("環境変数 ASSEMBLYAI_API_KEY を設定してください")
        diarizer = AssemblyAIStreamingDiarizationProvider(
            assemblyai_key,
            max_speakers=args.diarization_max_speakers,
        )
        hint = (
            f" max_speakers={args.diarization_max_speakers}"
            if args.diarization_max_speakers else ""
        )
        print(f"# 話者分離: AssemblyAI streaming を使用{hint}", flush=True)

    state = SessionState(args=args, started=started, out_path=out_path,
                         html_path=html_path, diag_path=diag_path,
                         turns_path=turns_path, wav_path=wav_path,
                         tracker=tracker, serve=_serve,
                         diarization_provider=diarizer,
                         speaker_resolver=SpeakerResolver())

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
    state.open_wav()

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
        # SSE(長時間接続)が他リクエストを塞がないよう、スレッド対応サーバーを使う
        from http.server import ThreadingHTTPServer
        try:
            _httpd = ThreadingHTTPServer(("127.0.0.1", args.port),
                                         _UIHandler.create(state))
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

    # --- 議題を脱線検出の基準論点としてシード（Fix 8 / 人間モードはS1で--topic対応） ---
    # 明示的な議題があれば、論点抽出LLMの成否に依存せず最初から脱線検出を効かせる。
    # 人間同士モード(--topic)・debate・simulate のいずれの議題でもシードできる。
    # --- 積極性プロファイルを適用（S5） ---
    from das.asr.live._constants import _PROACTIVITY_PROFILES
    if args.proactivity in _PROACTIVITY_PROFILES:
        state.proactivity = dict(_PROACTIVITY_PROFILES[args.proactivity])
        print(f"# 介入の積極性: {args.proactivity}", flush=True)

    # 会話モード(converse)で動的にパートナーを生成するための設定を保持（F3）
    if state.agent is not None:
        state._partner_cfg = {"api_key": _oai_key,
                              "voice": args.debate_voice,
                              "topic": args.topic or args.debate}

    _explicit_agenda = False
    if state.agent is not None:
        _agenda = args.topic or args.debate or args.simulate
        if _agenda:
            state.seed_topic(_agenda)
            _explicit_agenda = True
            print(f"# 脱線検出: 議題を基準論点としてシード → {_agenda}", flush=True)

    print(f"# {backend.name} に接続中…", flush=True)
    import contextlib as _contextlib

    def _connect_stt():
        _ws = connect(backend.ws_url(), additional_headers=backend.ws_headers())
        _ws.send(json.dumps(backend.start_message(args.model, args.lang)))
        return _ws

    state.stt_ws = _connect_stt()
    if state.diarization_provider is not None:
        state.diarization_provider.start()
    try:
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
            if state.agent is not None:
                threading.Thread(target=_run_drift_checker,
                                args=(state, _oai_key, _oai_model), daemon=True).start()
                print("# 脱線検出: 有効（3発話ごとに並列チェック）", flush=True)
                # --- 参加度の声かけ（発言の少ない人を誘う, S4） ---
                threading.Thread(target=_run_participation_checker,
                                args=(state, _oai_key, _oai_model), daemon=True).start()
                print("# 参加度の声かけ: 有効（発話量の偏りを監視）", flush=True)
                # --- 議題未指定なら冒頭アジェンダ自動検出（S3） ---
                if not _explicit_agenda:
                    threading.Thread(target=_run_agenda_detector,
                                    args=(state, _oai_key, _oai_model),
                                    daemon=True).start()
                    print("# 議題自動検出: 有効（冒頭の発話から推定）", flush=True)
        else:
            print("# 論点抽出: 無効（OPENAI_API_KEYが未設定）", flush=True)
        if state.agent is not None:
            _connect_agent(state, _on_agent_text)
        if state.partner is not None:
            state.partner.on_ai_utterance = _on_partner_text_factory(state)
            state.partner.connect()
            print(f"# Partner: voice={state.partner.voice} topic={state.partner.topic}",
                  flush=True)

        threading.Thread(target=_run_sender, args=(state, backend),
                         daemon=True).start()

        state.save()
        print("# 開始。話してください（「1=松井」で声を登録 / UIの停止ボタン or Ctrl+Cで終了）",
              flush=True)
        print(f"# 保存先: {out_path}", flush=True)
        if _serve:
            print(f"# ブラウザUI: http://127.0.0.1:{args.port}/ "
                  f"（モード切替・ライブ更新・新しい会議・停止）\n", flush=True)
        else:
            print(f"# ブラウザ表示: open {html_path}\n", flush=True)
        if not args.no_open:
            import webbrowser
            if _serve:
                webbrowser.open(f"http://127.0.0.1:{args.port}/")
            else:
                webbrowser.open("file://" + os.path.abspath(html_path))

        # UIからの停止フック: stopを立て、STTのWebSocketを閉じて受信ループを抜ける（F1）
        def _request_stop():
            state.stop.set()
            with _contextlib.suppress(Exception):
                if state.stt_ws is not None:
                    state.stt_ws.close()
        state.request_stop = _request_stop

        # UIからのフルリセット要求: STT接続を作り直す。実処理はメインスレッドが行う。
        def _request_reset():
            state.resetting = True
            state.rev += 1  # UIに「リセット中」を即通知
            state.reset_requested.set()
            with _contextlib.suppress(Exception):
                if state.stt_ws is not None:
                    state.stt_ws.close()
        state.request_reset = _request_reset

        # 受信ループ。reset要求が来たらSTTを張り直して次の会議へ。
        recv = RecvLoop(state, args, backend)
        while not state.stop.is_set():
            recv.run(state.stt_ws)
            if state.stop.is_set():
                break
            if state.reset_requested.is_set():
                print("# STTセッションを作り直しています…", flush=True)
                with _contextlib.suppress(Exception):
                    if state.stt_ws is not None:
                        state.stt_ws.close()
                state.reset_for_new_meeting()
                state.stt_ws = _connect_stt()
                recv = RecvLoop(state, args, backend)
                state.reset_requested.clear()
                state.resetting = False
                state.rev += 1
                print("# 新しい会議を開始しました", flush=True)
    finally:
        with _contextlib.suppress(Exception):
            if state.stt_ws is not None:
                state.stt_ws.close()
        if state.diarization_provider is not None:
            state.diarization_provider.close()
        _cleanup(state, args, api_key, tracker, wav_path, out_path, html_path)
