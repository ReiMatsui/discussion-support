"""セッション初期化・起動ロジック.

live.py (エントリーポイント) から呼ばれ、STTバックエンド構築・
声紋モデル読み込み・スレッド起動・受信ループ実行までを担う。
CLI引数の定義は live.py 側 (click) に残し、ここではパース済みの
値だけを受け取る。
"""
from __future__ import annotations

import contextlib
import datetime
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from das.asr.live._af_runtime import run_af_runtime
from das.asr.live._assemblyai_diarization import AssemblyAIStreamingDiarizationProvider
from das.asr.live._audio_io import (
    _run_from_mic,
    _run_from_wav,
    _run_sender,
)
from das.asr.live._cluster_naming import ClusterVoiceNamer
from das.asr.live._constants import (
    _AGENDA_PROMPT,
    _DRIFT_PROMPT,
    _FACTCHECK_PROMPT,
    _PARTICIPATION_PROMPT,
    _SUMMARY_VALUE_PROMPT,
    _TOPIC_PROMPT,
    _TRIAGE_PROMPT,
    OPENAI_API,
)
from das.asr.live._diarization import SpeakerResolver
from das.asr.live._pyannote_diarization import PyannoteStreamingDiarizationProvider
from das.asr.live._recv_loop import RecvLoop
from das.asr.live._seat_audio import SeatAudio, seat_embedder
from das.asr.live._session_state import SessionState
from das.asr.live._sortformer_diarization import SortformerLocalDiarizationProvider
from das.asr.live._ui import _UIHandler
from das.asr.live._voice_profiles import VoiceProfiles
from das.asr.live._workers import (
    _cleanup,
    _connect_agent,
    _on_agent_text_factory,
    _on_partner_text_factory,
    _run_af_checker,
    _run_agenda_detector,
    _run_drift_checker,
    _run_fact_checker,
    _run_participation_checker,
    _run_stdin_commands,
    _run_structuring_checker,
    _run_topic_worker,
    _run_triage_worker,
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
    model: str = "stt-rt-v5"
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
    stt: str = "soniox"
    # Sonioxのエンドポイント検出（文の切れ目で区切る＝議事録が読みやすい）。既定ON。
    soniox_endpoint: bool = True
    diarization: str = "none"  # none / pyannote / assemblyai / sortformer
    diarization_max_speakers: int | None = None
    # --diarization sortformer 用: NeMo 専用 venv の python（未指定なら
    # 環境変数 SORTFORMER_PYTHON → 既定パスの順で解決）と、レイテンシ設定。
    sortformer_python: str | None = None
    sortformer_latency: str = "low"
    sortformer_device: str = "cpu"
    # ハイブリッド構成（docs/design/pyannote_live1_trial_2026-07-09.md §9）:
    # --diarization pyannote と併用時のみ有効。pyannoteの生クラスタ単位で
    # 声紋照合し、名前を確定する（3役分業: Soniox=文字起こし/pyannote=クラスタ
    # リング/声紋照合=クラスタ単位の名前付け）。tracker(声紋)が無効なら無視される。
    vp_cluster_naming: bool = False
    # 二重帳簿の根治（handoff_2026-07-25_dual_ledger_rootcure.md 案B, opt-in）。
    # 声紋側が新しい人物Nを鋳造する瞬間だけ、席を持つクラスタの蓄積声紋と
    # 対称比較して同一人物なら統合する。既定 False ＝従来挙動。
    vp_mint_cluster_link: bool = False
    setup: bool = True
    port: int = 8231
    agent: bool = True
    agent_voice: str = "shimmer"
    agent_trigger: int = 10
    simulate: str | None = None
    sim_scenario: str | None = None
    debate: str | None = None
    debate_voice: str = "echo"
    topic: str | None = None   # 人間同士モードの議題（脱線判定の基準）
    proactivity: str = "standard"  # 介入の積極性（controlled/standard/active）
    af: bool = False  # AF ベース介入を有効化 (H1 フェーズ4)。既定 OFF (モード方針)。
    # AF ランタイムが事前に取り込む文書ディレクトリ（--af 有効時のみ使う）。
    # 未指定なら取り込まない。従来 run_session は getattr(args, "docs", None) を
    # 読んでいたが LiveArgs にこのフィールドが無く **常に None** だったため、
    # AFRuntime.ingest_documents が一度も走っていなかった（2026-07-25 監査）。
    docs: str | None = None


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


def start_ui_server(state: SessionState, port: int):
    """UIサーバーを起動する。指定ポートが使用中なら空きポートへ自動フォールバック.

    従来はポート使用中だと警告1行でUI無効のまま続行し、ブラウザの既存タブが
    「別プロセス（前セッションの生き残り）のUI」に繋がったままになる罠があった
    （2026-07-25 監査 B: 設定・開始操作が全部別プロセスへ飛ぶ）。ポートが
    塞がっていても新しいUIを必ず立て、実際のURLを目立つ形で表示する。

    戻り値: (httpd, 実際のポート)。起動不能なら (None, port)。
    """
    from http.server import ThreadingHTTPServer
    handler = _UIHandler.create(state)
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
        port = httpd.server_address[1]   # port=0 指定でも実ポートを返す
    except OSError as first_err:
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        except OSError as second_err:
            print(f"# 警告: UIサーバーを起動できません ({second_err})", flush=True)
            return None, port
        actual = httpd.server_address[1]
        print(f"# 注意: ポート{port}は使用中のため（{first_err}）、"
              f"空きポート{actual}でUIを起動します。", flush=True)
        print(f"#   既に開いているブラウザタブは前のセッションに繋がっている"
              f"可能性があります。このセッションのUIは "
              f"http://127.0.0.1:{actual}/ です", flush=True)
        port = actual
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port


_TOPICS_SCHEMA = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "speaker": {"type": "string"},
                },
                "required": ["topic", "speaker"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["topics"],
    "additionalProperties": False,
}

_DRIFT_SCHEMA = {
    "type": "object",
    "properties": {
        "drift": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["drift", "reason"],
    "additionalProperties": False,
}

_AGENDA_SCHEMA = {
    "type": "object",
    "properties": {
        "agenda": {"type": "string"},
    },
    "required": ["agenda"],
    "additionalProperties": False,
}

_PARTICIPATION_SCHEMA = {
    "type": "object",
    "properties": {
        "invite": {"type": "boolean"},
        "speaker": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["invite", "speaker", "reason"],
    "additionalProperties": False,
}

_FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "should_correct": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "claim": {"type": "string"},
        "correction": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["should_correct", "confidence", "claim", "correction", "reason"],
    "additionalProperties": False,
}

_TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "factual_claim": {"type": "boolean"},
        "facilitator_request": {"type": "string"},
    },
    "required": ["factual_claim", "facilitator_request"],
    "additionalProperties": False,
}

_SUMMARY_VALUE_SCHEMA = {
    "type": "object",
    "properties": {
        "intervene": {"type": "boolean"},
        "focus": {"type": "string"},
    },
    "required": ["intervene", "focus"],
    "additionalProperties": False,
}

_FACT_STYLE_ADVICE_RE = re.compile(
    r"(表現|言い方|言い換え|言い換える|語彙|文体|演出|比喩|"
    r"と表現|といった表現|と言うほう|と言った方|と言ったほう|"
    r"した方が|する方が|しておくと|誤解しにくい|正確です)"
)


# 推論の強さは弱いほうから試す。短いJSON抽出に推論は要らないので最小を望むが、
# **どの値が使えるかはモデルの世代で変わる**。実際 gpt-5.4-mini は 'minimal' を
# 拒否し（使えるのは none/low/medium/high/xhigh）、既定モデルが上がった瞬間に
# LLM機能が全滅した（2026-07-29）。世代ごとの対応表を持つと、次にモデルが
# 上がったとき同じことが起きる。だから**弾かれたら次を試して覚える**。
_EFFORT_ORDER = ("minimal", "none", "low")
# モデル名（小文字）-> そのモデルで通った値。セッション内だけ持つ。
_EFFORT: dict[str, str] = {}


def _build_chat_params(model: str, prompt: str, *, max_out: int,
                       temperature: float, schema_name: str | None = None,
                       schema: dict | None = None) -> dict:
    """Chat Completions のリクエストパラメータを構築する（モデル系統別）.

    gpt-5系/o系は推論モデルで、temperature指定不可・max_tokensは
    max_completion_tokensに改名。さらに推論トークンが出力枠を消費するため、
    枠が小さいと本文(JSON)が空のまま返る → 静かに失敗する（Fix 9の根因）。
    対策として reasoning_effort を抑え、出力枠を十分に確保する。
    """
    name = model.lower()
    params: dict = {"model": model,
                    "messages": [{"role": "user", "content": prompt}]}
    if schema_name and schema:
        params["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        }
    if name.startswith("gpt-5"):
        params["reasoning_effort"] = _EFFORT.get(name, _EFFORT_ORDER[0])
        params["max_completion_tokens"] = max_out
    elif name.startswith(("o1", "o3", "o4")):
        params["reasoning_effort"] = "low"       # o系は minimal 非対応
        params["max_completion_tokens"] = max_out
    else:
        params["temperature"] = temperature
        params["max_tokens"] = max_out
    return params


def _next_effort(params: dict, detail: str) -> str | None:
    """`reasoning_effort` が拒否されたら、次に試す値を返す（無ければ None）.

    弾かれた値より弱い順の次を選び、モデル名に対して覚える。以後の呼び出しは
    `_build_chat_params` がその値で組む（1セッション1回だけ余分に叩く）。
    """
    if "reasoning_effort" not in detail:
        return None      # 別の理由の 400（モデル名違い等）はここで扱わない
    cur = params.get("reasoning_effort")
    if cur not in _EFFORT_ORDER:
        return None
    nxt = _EFFORT_ORDER[_EFFORT_ORDER.index(cur) + 1:]
    if not nxt:
        return None
    _EFFORT[str(params.get("model", "")).lower()] = nxt[0]
    return nxt[0]


def _post_chat_json(params: dict, api_key: str, *, timeout: int, label: str,
                    _retried: bool = False):
    """Chat Completions を叩き、本文をJSONとして返す。失敗時はNone（理由をログ）."""
    import urllib.error
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
    except urllib.error.HTTPError as e:
        # **本文を必ず出す**。OpenAI は 400 の理由（未知のモデル名、その
        # モデルでは使えないパラメータ等）を本文に書いてくるので、ここを
        # 捨てると「400 Bad Request」だけが延々と流れて原因が分からない
        # （2026-07-29 に実会話で発生し、LLM機能が全滅しているのに何が
        # 悪いのか特定できなかった）。
        detail = ""
        with contextlib.suppress(Exception):
            detail = e.read().decode("utf-8", "replace")[:600]
        nxt = None if _retried else _next_effort(params, detail)
        if nxt is not None:
            print(f"# [{label}] reasoning_effort='{params['reasoning_effort']}' は"
                  f"{params.get('model')} で使えないため '{nxt}' で再試行します",
                  flush=True)
            return _post_chat_json({**params, "reasoning_effort": nxt}, api_key,
                                   timeout=timeout, label=label, _retried=True)
        print(f"# [{label}] APIエラー {e.code}: {detail or e.reason}"
              f"（model={params.get('model')}）", flush=True)
        return None
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
    params = _build_chat_params(
        model, prompt, max_out=2000, temperature=0.3,
        schema_name="topics_result", schema=_TOPICS_SCHEMA)
    result = _post_chat_json(params, api_key, timeout=30, label="topic")
    if not isinstance(result, dict):
        return []
    topics = result.get("topics")
    return topics if isinstance(topics, list) else []


def check_drift(utterances: list[dict], topics: list[dict],
                api_key: str, model: str) -> dict:
    """論点からの脱線を判定する（同期呼び出し、バックグラウンドスレッド用）.

    Returns:
        {"drift": bool, "reason": str} or {"drift": False} on error.
    """
    if not utterances or not topics or not api_key:
        return {"drift": False}
    utt_text = "\n".join(f"- {u['speaker']}: {u['text']}" for u in utterances)
    agenda_topics = [
        t for t in topics
        if t.get("speaker") in ("議題", "議題(自動)")
    ]
    explicit_agenda = [
        t for t in agenda_topics
        if t.get("speaker") == "議題"
    ]
    agenda_text = (
        "\n".join(f"- {t['topic']}" for t in explicit_agenda)
        if explicit_agenda else "（明示議題なし）"
    )
    flow_topics = [t for t in topics if t not in explicit_agenda]
    topic_text = "\n".join(f"- {t['topic']}" for t in flow_topics) or "（まだなし）"
    prompt = _DRIFT_PROMPT.format(
        agenda=agenda_text,
        topics=topic_text,
        utterances=utt_text,
    )
    params = _build_chat_params(
        model, prompt, max_out=800, temperature=0.0,
        schema_name="drift_result", schema=_DRIFT_SCHEMA)
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
    params = _build_chat_params(
        model, prompt, max_out=400, temperature=0.0,
        schema_name="agenda_result", schema=_AGENDA_SCHEMA)
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

    participation: [{"speaker": 名前, "time_share": 0-1,
                     "participation_share": 0-1 | optional,
                     "participation_share_label": str | optional,
                     "turns": int, "silent_sec": float}, ...]
    Returns: {"invite": bool, "speaker": 名前|None, "reason": str}
    """
    if not participation or not api_key:
        return {"invite": False}
    part_text = "\n".join(
        f"- {p['speaker']}: "
        f"{p.get('participation_share_label', '発話時間')}"
        f"{p.get('participation_share', p['time_share']) * 100:.0f}% / "
        f"{p['turns']}回 / 最終発言{p['silent_sec']:.0f}秒前"
        for p in participation)
    utt_text = "\n".join(f"- {u['speaker']}: {u['text']}" for u in utterances)
    prompt = _PARTICIPATION_PROMPT.format(participation=part_text,
                                          utterances=utt_text)
    params = _build_chat_params(
        model, prompt, max_out=400, temperature=0.0,
        schema_name="participation_result", schema=_PARTICIPATION_SCHEMA)
    result = _post_chat_json(params, api_key, timeout=15, label="invite")
    if not isinstance(result, dict):
        return {"invite": False}
    return result


def classify_utterance(utterances: list[dict[str, str]], api_key: str,
                       model: str) -> dict[str, object]:
    """確定発話1件を表層分類する（fact候補か / ファシリテーターへの依頼か）.

    判定対象は ``utterances`` の最後の1件で、前の要素は参照文脈（指示語・
    省略の補完用）。fact prefilter の正規表現群と音声呼びかけの regex 検出を
    置き換える、発話ごと1回だけの軽量 LLM 分類（H6/M2）。

    Returns:
        {"factual_claim": bool, "facilitator_request": str}
        API/解析の一時失敗時は {"retryable_error": True} を含む。
    """
    if not utterances or not api_key:
        return {"factual_claim": False, "facilitator_request": ""}
    lines = []
    for i, u in enumerate(utterances):
        label = "判定対象" if i == len(utterances) - 1 else "参照"
        lines.append(f"- [{label}] {u['speaker']}: {u['text']}")
    prompt = _TRIAGE_PROMPT.format(utterances="\n".join(lines))
    params = _build_chat_params(
        model, prompt, max_out=300, temperature=0.0,
        schema_name="utterance_triage", schema=_TRIAGE_SCHEMA)
    result = _post_chat_json(params, api_key, timeout=10, label="triage")
    if not isinstance(result, dict):
        return {"factual_claim": False, "facilitator_request": "",
                "retryable_error": True}
    return {
        "factual_claim": bool(result.get("factual_claim")),
        "facilitator_request": str(result.get("facilitator_request") or "").strip(),
    }


def check_summary_value(utterances: list[dict[str, Any]],
                        topics: list[dict[str, Any]],
                        api_key: str, model: str) -> dict[str, Any]:
    """今、短い整理・要約の介入が議論に価値を足すかを判定する（C3）.

    「10発話たまったら無条件に介入」の代わりに、系のどこかに「今は黙るべき」の
    判断を置くための価値判定。直近発話と論点一覧を見て intervene を返す。
    迷ったら false（過剰介入の回避）。

    Returns:
        {"intervene": bool, "focus": str}。API/解析失敗時は intervene=False。
    """
    if not utterances or not api_key:
        return {"intervene": False, "focus": ""}
    utt_text = "\n".join(f"- {u['speaker']}: {u['text']}" for u in utterances)
    topic_text = "\n".join(f"- {t['topic']}" for t in topics) or "（まだなし）"
    prompt = _SUMMARY_VALUE_PROMPT.format(topics=topic_text, utterances=utt_text)
    params = _build_chat_params(
        model, prompt, max_out=400, temperature=0.0,
        schema_name="summary_value", schema=_SUMMARY_VALUE_SCHEMA)
    result = _post_chat_json(params, api_key, timeout=15, label="structuring")
    if not isinstance(result, dict):
        return {"intervene": False, "focus": ""}
    return {
        "intervene": bool(result.get("intervene")),
        "focus": str(result.get("focus") or "").strip(),
    }


def check_fact_correction(utterances: list[dict], api_key: str, model: str) -> dict:
    """直近会話の明確な事実誤りを判定する.

    Returns:
        {"should_correct": bool, "confidence": str, "correction": str, ...}
        or {"should_correct": False} on error/low confidence.
    """
    if not utterances or not api_key:
        return {"should_correct": False}
    lines = []
    for i, u in enumerate(utterances):
        label = "判定対象" if i == len(utterances) - 1 else "参照"
        lines.append(f"- [{label}] {u['speaker']}: {u['text']}")
    utt_text = "\n".join(lines)
    prompt = _FACTCHECK_PROMPT.format(utterances=utt_text)
    params = _build_chat_params(
        model, prompt, max_out=700, temperature=0.0,
        schema_name="fact_correction_result", schema=_FACT_SCHEMA)
    result = _post_chat_json(params, api_key, timeout=15, label="fact")
    if not isinstance(result, dict):
        return {"should_correct": False, "retryable_error": True}
    if result.get("should_correct") is not True:
        return {"should_correct": False}
    if result.get("confidence") != "high":
        return {"should_correct": False}
    correction = str(result.get("correction") or "").strip()
    if not correction:
        return {"should_correct": False}
    if _FACT_STYLE_ADVICE_RE.search(correction):
        return {"should_correct": False}
    result["correction"] = correction
    return result


def vp_cluster_naming_disabled_warning(
        diarization: str, vp_cluster_naming: bool) -> str | None:
    """--vp-cluster-naming が効かない構成なら警告文を返す（有効構成なら None）.

    クラスタ単位の声紋名前付けは匿名クラスタ型の diarization（pyannote /
    sortformer）が前提のため、それ以外では機能しない。従来は assemblyai
    併用時のみ警告し、--diarization none（既定）では黙って無効化されていた
    （2026-07-15 レビュー F6。例: --hybrid --soniox-args "--diarization none"
    では、後勝ちで diarization だけが none に上書きされ、ユーザーはハイブリッド
    構成のつもりのまま気づけない）。diarization の値に依らず警告する。
    """
    if vp_cluster_naming and diarization not in ("pyannote", "sortformer"):
        return ("# 注意: --vp-cluster-naming は --diarization pyannote/sortformer "
                f"専用のため無効です（--diarization {diarization} では無視されます）")
    return None


def write_session_config(state, args: LiveArgs, tracker) -> None:
    """このランの構成を diag の先頭に1行残す（オフライン再生の前提を復元するため）.

    帰属の結果は構成に強く依存する（想定話者数・声紋モデル・自動登録の有無・
    ハイブリッド構成か・鋳造リンクが有効か）。従来これらはどこにも記録されず、
    記録を後から採点するときに「どの設定で録れたランか」を人手で思い出すしか
    なかった（2026-07-25 の実会話3本が上限1のまま録れていた事故は、まさにこれが
    記録に残っていなかったために事後まで気づけなかった）。
    """
    cfg = {
        "type": "session_config",
        "diarization": args.diarization,
        "diarization_max_speakers": args.diarization_max_speakers,
        "vp_cluster_naming": bool(args.vp_cluster_naming),
        "vp_mint_cluster_link": bool(args.vp_mint_cluster_link),
        "vp_model": None if tracker is None else tracker.model,
        "vp_auto": None if tracker is None else bool(tracker.auto),
        "vp_hybrid": None if tracker is None else bool(tracker.hybrid),
        "stt": args.stt,
    }
    try:
        with open(state.diag_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(cfg, ensure_ascii=False) + "\n")
    except OSError:
        pass


def vp_mint_cluster_link_disabled_warning(
        vp_cluster_naming: bool, vp_mint_cluster_link: bool) -> str | None:
    """--vp-mint-cluster-link が効かない構成なら警告文を返す（有効構成なら None）.

    鋳造リンクはクラスタの蓄積声紋と比較する機構なので、クラスタ単位の
    名前付け（--vp-cluster-naming）が無い構成では比較相手が存在しない。
    黙って無効化すると「入れたつもり」で検証してしまうため明示する
    （--vp-cluster-naming 側の F6 と同じ方針）。
    """
    if vp_mint_cluster_link and not vp_cluster_naming:
        return ("# 注意: --vp-mint-cluster-link は --vp-cluster-naming と"
                "併用したときだけ機能します（単独指定では無視されます）")
    return None


# ---------------------------------------------------------------------------
# メインのセッション起動
# ---------------------------------------------------------------------------

def _build_tracker(args) -> VoiceProfiles | None:
    """声紋モデルを読み込む（読めない場合は順に代替モデルへ落とす）.

    どのモデルも読めなければ None を返し、声紋照合なしで進む——起動そのものは
    止めない（文字起こしは動くため）。ただし黙って落ちると「人物が確定しない
    のはなぜか」が分からないので、警告と復旧手順を必ず出す。
    """
    if args.no_vp:
        return None
    tracker: VoiceProfiles | None = None
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
        print("# 声紋プロファイル: なし。未知の声は名前未登録の参加者として自動追跡、"
              f"ブラウザUIで名前を登録すると次回から自動表示（{args.voices}）", flush=True)
    if tracker is not None:
        tracker.set_max_human_speakers(args.diarization_max_speakers)
    return tracker


def _speaker_cap_hint(args) -> str:
    """起動ログに出す「 max_speakers=N」（未指定なら空）."""
    return (f" max_speakers={args.diarization_max_speakers}"
            if args.diarization_max_speakers else "")


def _build_diarizer(args):
    """--diarization の指定に応じて話者分離の供給元を作る（未指定なら None）."""
    diarizer = None
    warning = vp_cluster_naming_disabled_warning(args.diarization, args.vp_cluster_naming)
    if warning:
        print(warning, flush=True)
    warning = vp_mint_cluster_link_disabled_warning(
        args.vp_cluster_naming, args.vp_mint_cluster_link)
    if warning:
        print(warning, flush=True)
    if args.diarization == "pyannote":
        pyannote_key = os.environ.get("PYANNOTEAI_API_KEY")
        if not pyannote_key:
            raise SystemExit("環境変数 PYANNOTEAI_API_KEY を設定してください")
        diarizer = PyannoteStreamingDiarizationProvider(
            pyannote_key,
            max_speakers=args.diarization_max_speakers,
        )
        print(f"# 話者分離: pyannoteAI streaming を使用"
              f"{_speaker_cap_hint(args)}", flush=True)
    elif args.diarization == "assemblyai":
        assemblyai_key = os.environ.get("ASSEMBLYAI_API_KEY")
        if not assemblyai_key:
            raise SystemExit("環境変数 ASSEMBLYAI_API_KEY を設定してください")
        diarizer = AssemblyAIStreamingDiarizationProvider(
            assemblyai_key,
            max_speakers=args.diarization_max_speakers,
        )
        print(f"# 話者分離: AssemblyAI streaming を使用"
              f"{_speaker_cap_hint(args)}", flush=True)
    elif args.diarization == "sortformer":
        # ローカル Streaming Sortformer（opt-in 検証用, 2026-07-22）。
        # NeMo 専用 venv のサブプロセスで動くため API キーは不要。
        diarizer = SortformerLocalDiarizationProvider(
            python_path=args.sortformer_python,
            latency=args.sortformer_latency,
            device=args.sortformer_device,
            max_speakers=args.diarization_max_speakers,
        )
        print(f"# 話者分離: ローカル Streaming Sortformer を使用"
              f"（latency={args.sortformer_latency} device={args.sortformer_device}。"
              f"注: 話者は最大4人・マイク残響環境では現行pyannote構成より弱い実測。"
              f"docs/design/sortformer_feasibility_2026-07-22.md）", flush=True)
    return diarizer


def _build_cluster_layer(args, tracker):
    """ハイブリッド構成（クラスタ単位の声紋名前付け＋席の音声）を組む.

    戻り値: (cluster_namer, seat_audio)。条件を満たさなければ (None, None) で、
    Soniox単独・pyannote単独の挙動は変わらない。
    """
    cluster_namer = None
    seat_audio = None
    # --- ハイブリッド構成: 匿名クラスタ単位の声紋名前付け ---
    # (docs/design/pyannote_live1_trial_2026-07-09.md §9)。匿名クラスタ型の
    # diarization（pyannote / sortformer）かつ --vp-cluster-naming 指定時、
    # かつ声紋照合(tracker)が有効な時だけ生成する。
    # tracker が無い（--no-vp や依存未導入）場合は照合しようがないため無視する。
    cluster_namer = None
    seat_audio = None
    if args.diarization in ("pyannote", "sortformer") and args.vp_cluster_naming:
        if tracker is not None:
            cluster_namer = ClusterVoiceNamer(tracker)
            # 席落ち発話の割当て（handoff §27）。クラスタ分裂で席を得られず
            # 未確定に落ちる発話を、席を持つ人の実音声と比べて寄せ直す。
            # ハイブリッド構成に閉じるので pyannote単独・Soniox単独は不変。
            seat_audio = SeatAudio(tracker, embedder=seat_embedder(tracker))
            # ハイブリッド時のみ、短発話(short_floor〜min_sec)の声紋照合を既知1人
            # でも試みる（VoiceProfiles.hybrid のコメント参照。実測: 声紋一致92%
            # vs 前話者追従28%, transcripts/2026-07-14_1729 GT評価）。
            tracker.set_hybrid(True)
            print("# 話者名前付け: pyannoteクラスタ単位の声紋照合ハイブリッド構成を使用"
                  "（docs/design/pyannote_live1_trial_2026-07-09.md §9）", flush=True)
            if args.vp_mint_cluster_link:
                print("# 鋳造リンク(opt-in): 新しい人物の鋳造時に、席を持つクラスタと"
                      "対称比較して同一人物なら統合します"
                      "（docs/design/handoff_2026-07-25_dual_ledger_rootcure.md 案B）",
                      flush=True)
        else:
            print("# 注意: --vp-cluster-naming は声紋照合(tracker)が無効なため無視されます"
                  "（--no-vp解除 or 依存導入が必要）", flush=True)
    return cluster_namer, seat_audio


def _start_llm_workers(state, args, *, oai_key: str, oai_model: str,
                       out_path: str, explicit_agenda: bool) -> None:
    """LLM を使う常駐ワーカーを起動する（APIキーが無ければ何も起こさない）.

    どれも会議の進行を助けるための背景処理で、話者の帰属には関わらない。
    エージェントが居ないときは論点抽出だけを動かす——他は介入するための
    判断材料であり、介入先が無ければ API を叩くだけ無駄になる。
    """
    if oai_key:
        threading.Thread(target=_run_topic_worker,
                        args=(state, oai_key, oai_model), daemon=True).start()
        print("# 論点抽出: 有効（5発話ごとにLLMで分析）", flush=True)
        if state.agent is not None:
            threading.Thread(target=_run_drift_checker,
                            args=(state, oai_key, oai_model), daemon=True).start()
            print("# 脱線検出: 有効（3発話ごとに並列チェック）", flush=True)
            threading.Thread(target=_run_triage_worker,
                            args=(state, oai_key, oai_model), daemon=True).start()
            print("# 発話分類: 有効（fact候補・ファシリテーター呼びかけをLLMで判定）",
                  flush=True)
            threading.Thread(target=_run_fact_checker,
                            args=(state, oai_key, oai_model), daemon=True).start()
            print("# 事実誤り補正: 有効（高確信の定義・式だけ短く補足）", flush=True)
            # --- 参加度の声かけ（発言の少ない人を誘う, S4） ---
            threading.Thread(target=_run_participation_checker,
                            args=(state, oai_key, oai_model), daemon=True).start()
            print("# 参加度の声かけ: 有効（発話量の偏りを監視）", flush=True)
            # --- 整理介入の価値判定（count の無条件介入を置換, C3） ---
            threading.Thread(target=_run_structuring_checker,
                            args=(state, oai_key, oai_model), daemon=True).start()
            print("# 整理介入: 有効（N発話到達時にLLMで価値判定）", flush=True)
            # --- AF ランタイム + AF 介入 (H1 フェーズ3/4) ---
            # extraction/linking を毎発話回すため API コストが増える。**既定では
            # 無効**で、--af または DAS_AF_RUNTIME=1 のときだけ常駐＋介入する
            # (モード方針 2026-07-03: 既定 OFF・ルールベースモード恒久維持)。
            _af_enabled = (
                os.environ.get("DAS_AF_RUNTIME") == "1"
                or bool(getattr(args, "af", False))
            )
            if _af_enabled:
                _af_snapshot = os.path.splitext(out_path)[0] + ".af.json"
                threading.Thread(
                    target=run_af_runtime,
                    args=(state, oai_key, oai_model),
                    kwargs={
                        "docs_dir": args.docs,
                        "snapshot_path": _af_snapshot,
                    },
                    daemon=True,
                ).start()
                # AF checker: AF から介入候補を作り Controller 採否へ流す (フェーズ4)
                threading.Thread(
                    target=_run_af_checker, args=(state,), daemon=True,
                ).start()
                print("# AF 介入: 有効（--af / DAS_AF_RUNTIME=1）", flush=True)
            # --- 議題未指定なら冒頭アジェンダ自動検出（S3） ---
            if not explicit_agenda:
                threading.Thread(target=_run_agenda_detector,
                                args=(state, oai_key, oai_model),
                                daemon=True).start()
                print("# 議題自動検出: 有効（冒頭の発話から推定）", flush=True)
    else:
        print("# 論点抽出: 無効（OPENAI_API_KEYが未設定）", flush=True)


def _receive_until_stopped(state, args, backend, connect_stt) -> None:
    """STT受信ループを回し、切断と会議のリセットに耐えて回り続ける.

    抜ける条件は2つだけ——停止要求か、STT側の終了(finished)。切断は再接続
    （回数に応じて待ちを伸ばす）、リセット要求は接続を張り直して次の会議へ。
    どちらの場合も RecvLoop を作り直す——組み立て途中の発話や直近の区間を
    持ち越すと、前の会議の断片が新しい会議に混ざる。
    """
    import contextlib as _contextlib

    # 受信ループ。reset要求が来たらSTTを張り直して次の会議へ。
    recv = RecvLoop(state, args, backend)
    reconnect_attempts = 0
    while not state.stop.is_set():
        status = recv.run(state.stt_ws)
        if state.stop.is_set():
            break
        if state.reset_requested.is_set():
            reconnect_attempts = 0
            print("# STTセッションを作り直しています…", flush=True)
            with _contextlib.suppress(Exception):
                if state.stt_ws is not None:
                    state.stt_ws.close()
            state.stt_ws = None
            state.reset_for_new_meeting()
            if state.diarization_provider is not None:
                with _contextlib.suppress(Exception):
                    state.diarization_provider.close()
                state.diarization_provider.start()
            if state.waiting_to_start:
                state.resetting = False
                state.rev += 1
                print("# 開始前設定: 確認後に「会議を開始」を押してください", flush=True)
                while not state.stop.is_set() and not state.start_requested.wait(timeout=0.2):
                    pass
                if state.stop.is_set():
                    break
            try:
                state.stt_ws = connect_stt()
            except Exception as e:
                # 瞬断とリセットが重なっただけでセッション全体を落とさない。
                # 通常の切断（disconnected 分岐）は再試行するのに、リセット
                # 経路だけ素通しで例外が run_session まで抜けていた（レビュー
                # 2026-07-30）。reset_requested を立てたまま continue すれば
                # この分岐に戻ってきて再試行になる。
                print(f"# 新しい会議のSTT接続に失敗（再試行します）: {e}",
                      flush=True)
                time.sleep(1.0)
                continue
            recv = RecvLoop(state, args, backend)
            state.reset_requested.clear()
            state.resetting = False
            state.rev += 1
            print("# 新しい会議を開始しました", flush=True)
        elif status == "disconnected":
            reconnect_attempts += 1
            delay = min(5.0, 0.5 * reconnect_attempts)
            print(f"# STTに再接続中… ({reconnect_attempts}回目)", flush=True)
            with _contextlib.suppress(Exception):
                if state.stt_ws is not None:
                    state.stt_ws.close()
            if state.diarization_provider is not None:
                with _contextlib.suppress(Exception):
                    state.diarization_provider.close()
                state.diarization_provider.start()
            time.sleep(delay)
            try:
                state.stt_ws = connect_stt()
            except Exception as e:
                print(f"# STT再接続に失敗: {e}", flush=True)
                continue
            recv = RecvLoop(state, args, backend)
            print("# STTに再接続しました", flush=True)
        else:
            reconnect_attempts = 0
            if status == "finished":
                break


def run_session(args: LiveArgs) -> None:
    """セッションを初期化し、STT受信ループを実行する.

    発話ごとのフック（ON_UTTERANCE）は `_recv_loop` が live モジュールから
    直接読む。以前はここで参照を渡していたが、本文では一度も使って
    いなかった（2026-07-28 の棚卸しで発見）。
    """
    from das.asr.live import _SYS_HOOK_REF  # 遅延import（循環回避）

    load_env()
    if args.wav and not os.path.exists(args.wav):
        raise SystemExit(f"音声ファイルがありません: {args.wav}\n"
                         "（テスト音声は scripts/make_overlap_testset.py 等で先に生成してください）")

    backend = build_backend(args)
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
    tracker = _build_tracker(args)

    # --- SessionState ---
    wav_path = os.path.splitext(out_path)[0] + ".wav"
    diarizer = _build_diarizer(args)
    cluster_namer, seat_audio = _build_cluster_layer(args, tracker)

    state = SessionState(args=args, started=started, out_path=out_path,
                         html_path=html_path, diag_path=diag_path,
                         turns_path=turns_path, wav_path=wav_path,
                         tracker=tracker, serve=_serve,
                         diarization_provider=diarizer,
                         speaker_resolver=SpeakerResolver(),
                         cluster_namer=cluster_namer,
                         seat_audio=seat_audio)
    state.stt_backend = backend
    state.waiting_to_start = bool(args.setup and _serve and not args.wav and not args.simulate)
    write_session_config(state, args, tracker)

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
        # 経過時刻付きで記録する（監査D: [--:--] はいつ起きたか追えない）。
        state.add_sys(state.elapsed_ms(), text)
        state.save()
    _SYS_HOOK_REF[0] = _sys_hook

    # --- 論点抽出 ---
    _oai_key = os.environ.get("OPENAI_API_KEY", "")
    _oai_model = os.environ.get("OPENAI_MODEL_FAST", "gpt-5.4-mini")

    # --- AIエージェント: コールバック ---
    _on_agent_text = _on_agent_text_factory(state)

    # --- UIサーバー ---
    _httpd = None
    _ui_port = args.port
    if _serve:
        _httpd, _ui_port = start_ui_server(state, args.port)
        if _httpd is None:
            _serve = False
            state._serve = False
            state.waiting_to_start = False

    if args.simulate and args.debate:
        raise SystemExit("--simulate と --debate は同時に使えません")

    # --- DiscussionSimulator ---
    if args.simulate:
        if not _oai_key:
            raise SystemExit("--simulate には OPENAI_API_KEY が必要です")
        if not args.agent:
            print("# ヒント: --no-agent 指定中のためファシリテーターは介入しません", flush=True)
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
            print("# ヒント: --no-agent 指定中のためファシリテーターは介入しません", flush=True)
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
    if state.set_proactivity(args.proactivity).get("ok"):
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

    import contextlib as _contextlib

    def _connect_stt():
        _ws = connect(backend.ws_url(), additional_headers=backend.ws_headers())
        _ws.send(json.dumps(backend.start_message(args.model, args.lang)))
        state.mark_stt_connection_started()
        return _ws

    state.save()
    if _serve:
        print(f"# ブラウザUI: http://127.0.0.1:{_ui_port}/ "
              f"（開始前設定・モード切替・ライブ更新・新しい会議・停止）", flush=True)
    else:
        print(f"# ブラウザ表示: open {html_path}", flush=True)
    if not args.no_open:
        import webbrowser
        if _serve:
            webbrowser.open(f"http://127.0.0.1:{_ui_port}/")
        else:
            webbrowser.open("file://" + os.path.abspath(html_path))
    audio_started = False
    if state.waiting_to_start and not args.wav and not args.simulate:
        threading.Thread(target=_run_from_mic, args=(state, args.device),
                         daemon=True).start()
        threading.Thread(target=_run_sender, args=(state, backend),
                         daemon=True).start()
        audio_started = True
    if state.waiting_to_start:
        print("# 開始前設定: ブラウザで参加人数などを確認し、「会議を開始」を押してください", flush=True)
        while not state.stop.is_set() and not state.start_requested.wait(timeout=0.2):
            pass
    if state.stop.is_set():
        _cleanup(state, tracker, wav_path, out_path, html_path)
        return

    print(f"# {backend.name} に接続中…", flush=True)
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
            elif not audio_started:
                threading.Thread(target=_run_from_mic, args=(state, args.device),
                                 daemon=True).start()
        threading.Thread(target=_run_stdin_commands, args=(state,),
                         daemon=True).start()
        _start_llm_workers(state, args, oai_key=_oai_key,
                           oai_model=_oai_model, out_path=out_path,
                           explicit_agenda=_explicit_agenda)
        if state.agent is not None:
            _connect_agent(state, _on_agent_text)
        if state.partner is not None:
            state.partner.on_ai_utterance = _on_partner_text_factory(state)
            state.partner.connect()
            print(f"# Partner: voice={state.partner.voice} topic={state.partner.topic}",
                  flush=True)

        if not audio_started:
            threading.Thread(target=_run_sender, args=(state, backend),
                             daemon=True).start()

        state.save()
        print("# 開始。話してください（名前登録はブラウザUIから / UIの停止ボタン or Ctrl+Cで終了）",
              flush=True)
        print(f"# 保存先: {out_path}", flush=True)

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

        _receive_until_stopped(state, args, backend, _connect_stt)
    except KeyboardInterrupt:
        # Ctrl+C はトレースバックを出さず、UIの停止ボタンと同じ扱いで安全に終了する
        # （ブラウザタブを閉じてしまった場合も、タブを開き直すか Ctrl+C で停止できる）。
        print("\n# Ctrl+C を受信。議事録を保存して安全に終了します…", flush=True)
        state.stop.set()
    finally:
        with _contextlib.suppress(Exception):
            if state.stt_ws is not None:
                state.stt_ws.close()
        if state.diarization_provider is not None:
            state.diarization_provider.close()
        _cleanup(state, tracker, wav_path, out_path, html_path)
