#!/usr/bin/env python3
"""pyannoteAI バッチ diarization vs 現行話者タイムラインの一致度ベンチマーク.

目的:
  現行システム（Soniox STT + 声紋照合）が transcripts/*.turns.jsonl に書き出した
  話者タイムラインと、pyannoteAI のバッチ diarization API (``POST /v1/diarize``)
  が同じ wav から出す話者区間を突き合わせ、「乗り換える価値があるか」の一次判断
  材料を作る。詳しい使い方・判断基準は docs/design/pyannote_live1_trial_2026-07-09.md
  を参照。

pyannoteAI API 要点（2026-07 時点 docs.pyannote.ai 調べ）:
  - 認証: ``Authorization: Bearer <API_KEY>`` (JWT bearer)。
  - ジョブ投入: ``POST https://api.pyannote.ai/v1/diarize``
      body: {"url": <公開URL or media://object-key>, "numSpeakers"?, "minSpeakers"?,
             "maxSpeakers"?, "model"? ("precision-2" 既定 / "community-1")}
      -> {"jobId": "...", "status": "created"}
  - ローカルファイルは直接送れないため、一時アップロードAPIを使う:
      1. ``POST /v1/media/input`` body {"url": "media://<key>"}
         -> {"url": "<presigned PUT URL>"}
      2. そのURLへ ``PUT`` で生バイトをアップロード（48時間で自動削除）
      3. diarize リクエストの url に ``media://<key>`` を指定
  - 結果取得（ポーリング）: ``GET /v1/jobs/{jobId}``
      -> {"jobId", "status": pending|created|running|succeeded|failed|canceled,
          "output": {"diarization": [{"speaker": "SPEAKER_00", "start": 15.0, "end": 30.5}, ...]}}
      status が succeeded になって初めて output が入る。結果は24時間で自動削除される
      ため、必要なら自前で保存する（本スクリプトは *.pyannote_bench.json に保存する）。
  - 話者数のヒント: numSpeakers（確定数） or minSpeakers/maxSpeakers（範囲）。
    指定すると精度が上がるとされる。未指定なら自動推定。
  - 課金: 成功したジョブのみ課金、最低20秒課金。/diarize はバッチ、
    Live-1 (``POST /v1/live``) はストリーミングで同様に秒単位課金（今回のスクリプトは
    バッチ API のみを叩く。Live-1 との精度同等性の確認が目的なので、まずバッチで
    現行比較を行う）。

モデルについて（2026-07-09 Live-1調査で再確認）:
  docs.pyannote.ai/models を確認した結果、バッチ diarization の最新・最高精度モデルは
  引き続き ``precision-2``（Community-1 比 +28% 精度、既定モデル）であり、本スクリプトが
  これまで使っていたモデル指定（既定 ``precision-2`` 相当）は最新のままだった。
  = 前回実行分もモデル自体は最新だったので、モデル変更に起因する再実行は不要。
  ただし従来は ``--model`` 未指定時にAPI側の既定へ委ねていた（サーバ内部で変わる可能性が
  ある）ため、本改修で ``--model`` の既定値を ``precision-2`` に明示指定するよう変更した。

環境変数:
  PYANNOTEAI_API_KEY  既存コード（src/das/asr/live/_bootstrap.py）が使っている名前。
                       本スクリプトはこちらを優先して読む。
  PYANNOTE_API_KEY     依頼書に指定された名前。上が未設定なら fallback として使う。

使い方:
  # 実APIを叩く（要 PYANNOTEAI_API_KEY）
  uv run python scripts/benchmark_pyannote.py --session 2026-06-12_1346
  uv run python scripts/benchmark_pyannote.py --session 2026-06-12_1346 --session 2026-06-12_1351 \
      --num-speakers 2

  # wav を直接指定（同名 .turns.jsonl を自動解決）
  uv run python scripts/benchmark_pyannote.py --wav transcripts/2026-06-12_1346.wav

  # APIキーなしでロジックだけ検証（モックJSONを読む）
  uv run python scripts/benchmark_pyannote.py --session 2026-06-12_1346 --dry-run \
      --from-json mock_pyannote_output.json

  # APIキーもモックJSONもなくロジックだけ通す（合成モックを内部生成）
  uv run python scripts/benchmark_pyannote.py --session 2026-06-12_1346 --dry-run

出力:
  - 標準出力にセッションごと・全体のサマリ
  - transcripts/<session>.pyannote_bench.json に詳細（比較パラメータ・
    話者マッピング・不一致ターン一覧・pyannote生セグメント）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

API_BASE = "https://api.pyannote.ai"
TRANSCRIPTS_DIR = Path("transcripts")
MIN_CHARGE_SECONDS = 20
DEFAULT_POLL_INTERVAL_S = 5.0
DEFAULT_POLL_TIMEOUT_S = 900.0
DEFAULT_UNKNOWN_LABEL = "未確定"
# ms タイミングを持たない合成話者（AI応答・ファシリテーター発話など、音声を伴わない
# ターン）。turns.jsonl では ms/end_ms が null になるので has_timing で自然に除外
# されるが、名寄せ用に明示しておく。
_SYNTHETIC_SPEAKERS = {"AI", "パートナー", "ファシリテーター", "[Partner]"}


# ---------------------------------------------------------------------------
# データ構造
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Turn:
    """turns.jsonl の1行."""

    turn_id: int
    speaker: str
    text: str
    ms: int | None
    end_ms: int | None

    @property
    def has_timing(self) -> bool:
        """発話区間が確定していて pyannote 結果と時間比較できるか."""
        return (
            self.ms is not None
            and self.end_ms is not None
            and self.end_ms > self.ms
        )


@dataclass(frozen=True)
class PyannoteSegment:
    """pyannoteAI diarization API が返す1話者区間（ms換算済み）."""

    speaker: str
    start_ms: int
    end_ms: int

    def overlap_ms(self, ms: int, end_ms: int) -> int:
        return max(0, min(self.end_ms, end_ms) - max(self.start_ms, ms))


@dataclass
class TurnMismatch:
    """現行とpyannote(マッピング後)の話者が食い違ったターン."""

    turn_id: int
    ms: int
    end_ms: int
    current_speaker: str
    pyannote_speaker_raw: str | None
    pyannote_speaker_mapped: str | None
    overlap_ratio: float
    text_head: str


@dataclass
class UnresolvedFinding:
    """現行が未確定/不明扱いした区間について、pyannoteがどう割ったか."""

    turn_id: int
    ms: int
    end_ms: int
    text_head: str
    pyannote_speakers_mapped: list[str]
    pyannote_speakers_raw: list[str]
    covered_ratio: float
    split_into_multiple: bool


@dataclass
class SessionBenchResult:
    session: str
    wav_path: str
    turns_path: str
    params: dict[str, Any]
    n_turns_total: int
    n_turns_timed: int
    n_turns_compared: int
    n_matched: int
    match_rate: float
    current_speaker_count: int
    pyannote_speaker_count_raw: int
    pyannote_speaker_count_mapped: int
    speaker_count_matches: bool
    speaker_mapping: dict[str, str]
    mismatches: list[TurnMismatch] = field(default_factory=list)
    unresolved_findings: list[UnresolvedFinding] = field(default_factory=list)
    pyannote_segments: list[PyannoteSegment] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# 入力解決 (--session / --wav)
# ---------------------------------------------------------------------------


def resolve_turns_path(wav_path: Path) -> Path:
    """wavパスから同名の *.turns.jsonl を推定する."""
    return wav_path.with_name(f"{wav_path.stem}.turns.jsonl")


def resolve_sessions(
    sessions: list[str], wavs: list[str]
) -> list[tuple[str, Path, Path]]:
    """--session / --wav 指定から (session_id, wav_path, turns_path) の列を作る."""
    resolved: list[tuple[str, Path, Path]] = []
    for session_id in sessions:
        wav_path = TRANSCRIPTS_DIR / f"{session_id}.wav"
        turns_path = resolve_turns_path(wav_path)
        resolved.append((session_id, wav_path, turns_path))
    for wav_str in wavs:
        wav_path = Path(wav_str)
        session_id = wav_path.stem
        turns_path = resolve_turns_path(wav_path)
        resolved.append((session_id, wav_path, turns_path))
    return resolved


def load_turns(turns_path: Path) -> list[Turn]:
    turns: list[Turn] = []
    with turns_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            turns.append(
                Turn(
                    turn_id=d["turn_id"],
                    speaker=d["speaker"],
                    text=d.get("text", ""),
                    ms=d.get("ms"),
                    end_ms=d.get("end_ms"),
                )
            )
    return turns


def wav_duration_seconds(wav_path: Path) -> float | None:
    """診断用: wavの長さ・サンプルレートを確認する（16kHz前提の裏取り）."""
    try:
        with wave.open(str(wav_path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except (OSError, wave.Error):
        return None


def wav_sample_rate(wav_path: Path) -> int | None:
    try:
        with wave.open(str(wav_path), "rb") as w:
            return w.getframerate()
    except (OSError, wave.Error):
        return None


# ---------------------------------------------------------------------------
# pyannoteAI API 呼び出し
# ---------------------------------------------------------------------------


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def upload_media(wav_path: Path, object_key: str, api_key: str) -> str:
    """一時ストレージにアップロードし、diarize用の media:// URL を返す.

    POST /v1/media/input {"url": "media://<key>"} -> {"url": "<presigned PUT>"}
    その後、その presigned URL へ PUT で生バイトを送る。
    """
    import requests

    create_resp = requests.post(
        f"{API_BASE}/v1/media/input",
        json={"url": f"media://{object_key}"},
        headers=_auth_headers(api_key),
        timeout=30,
    )
    create_resp.raise_for_status()
    presigned_url = create_resp.json()["url"]

    with wav_path.open("rb") as fh:
        put_resp = requests.put(
            presigned_url,
            data=fh,
            headers={"Content-Type": "application/octet-stream"},
            timeout=600,
        )
    put_resp.raise_for_status()
    return f"media://{object_key}"


def create_diarize_job(
    media_url: str,
    api_key: str,
    *,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    model: str | None = None,
) -> str:
    """POST /v1/diarize でジョブを作り jobId を返す."""
    import requests

    body: dict[str, Any] = {"url": media_url}
    if num_speakers is not None:
        body["numSpeakers"] = num_speakers
    if min_speakers is not None:
        body["minSpeakers"] = min_speakers
    if max_speakers is not None:
        body["maxSpeakers"] = max_speakers
    if model:
        body["model"] = model

    resp = requests.post(
        f"{API_BASE}/v1/diarize", json=body, headers=_auth_headers(api_key), timeout=30
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") == "failed":
        raise RuntimeError(f"pyannote diarize job creation failed: {data}")
    return data["jobId"]


def poll_job(
    job_id: str,
    api_key: str,
    *,
    interval_s: float = DEFAULT_POLL_INTERVAL_S,
    timeout_s: float = DEFAULT_POLL_TIMEOUT_S,
) -> dict[str, Any]:
    """GET /v1/jobs/{jobId} を succeeded/failed/canceled になるまでポーリングする."""
    import requests

    deadline = time.monotonic() + timeout_s
    while True:
        resp = requests.get(
            f"{API_BASE}/v1/jobs/{job_id}", headers=_auth_headers(api_key), timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        if status == "succeeded":
            output = data.get("output")
            if not output:
                raise RuntimeError(f"pyannote job {job_id} succeeded but has no output")
            return output
        if status in {"failed", "canceled"}:
            raise RuntimeError(f"pyannote job {job_id} ended with status={status}: {data}")
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"pyannote job {job_id} polling timed out after {timeout_s}s "
                f"(last status={status})"
            )
        time.sleep(interval_s)


def run_pyannote_diarization(
    wav_path: Path,
    api_key: str,
    *,
    session_id: str,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    model: str | None,
    poll_interval_s: float,
    poll_timeout_s: float,
) -> dict[str, Any]:
    """アップロード→ジョブ投入→ポーリングの一連の流れを実行し output を返す."""
    object_key = f"das-bench/{session_id}"
    media_url = upload_media(wav_path, object_key, api_key)
    job_id = create_diarize_job(
        media_url,
        api_key,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        model=model,
    )
    return poll_job(job_id, api_key, interval_s=poll_interval_s, timeout_s=poll_timeout_s)


# ---------------------------------------------------------------------------
# レスポンス正規化 / モック生成
# ---------------------------------------------------------------------------


def parse_pyannote_segments(payload: Any) -> list[PyannoteSegment]:
    """pyannote APIレスポンス(またはそのoutput/diarizationの一部)からセグメント列を作る.

    受け付ける形:
      - {"output": {"diarization": [...]}}  (GET /v1/jobs/{id} 生レスポンス)
      - {"diarization": [...]}              (output だけ渡された場合)
      - [{"speaker","start","end"}, ...]    (diarization配列そのもの)
    """
    if isinstance(payload, dict):
        if "output" in payload and isinstance(payload["output"], dict):
            payload = payload["output"]
        if "diarization" in payload:
            payload = payload["diarization"]
    if not isinstance(payload, list):
        raise ValueError(
            "pyannote payload から diarization セグメント配列を取り出せませんでした: "
            f"{type(payload)!r}"
        )
    segments: list[PyannoteSegment] = []
    for seg in payload:
        start_ms = int(round(float(seg["start"]) * 1000))
        end_ms = int(round(float(seg["end"]) * 1000))
        if end_ms <= start_ms:
            continue
        segments.append(PyannoteSegment(speaker=seg["speaker"], start_ms=start_ms, end_ms=end_ms))
    segments.sort(key=lambda s: s.start_ms)
    return segments


def build_dry_run_mock(turns: list[Turn]) -> list[PyannoteSegment]:
    """--dry-run かつ --from-json 未指定のときの合成モック.

    実APIキーなしでもパース・比較ロジックを一通り動かせるように、現行turnsの
    タイミングを流用してpyannote風のセグメント列をでっちあげる。人物ラベルを
    SPEAKER_NN に付け替え、最後のタイミング付きターンをわざと隣の話者に
    誤帰属させて不一致検出ロジックが動くことも確認する。
    """
    timed = [t for t in turns if t.has_timing]
    label_to_id: dict[str, str] = {}
    segments: list[PyannoteSegment] = []
    for i, t in enumerate(timed):
        speaker_id = label_to_id.setdefault(t.speaker, f"SPEAKER_{len(label_to_id):02d}")
        is_last = i == len(timed) - 1
        if is_last and len(label_to_id) > 1:
            # わざと別話者に誤帰属させ、不一致検出のスモークテストにする
            other = next(v for k, v in label_to_id.items() if v != speaker_id)
            speaker_id = other
        segments.append(PyannoteSegment(speaker=speaker_id, start_ms=t.ms, end_ms=t.end_ms))  # type: ignore[arg-type]
    return segments


# ---------------------------------------------------------------------------
# 比較ロジック
# ---------------------------------------------------------------------------


def greedy_speaker_mapping(
    turns: list[Turn], segments: list[PyannoteSegment]
) -> dict[str, str]:
    """pyannote話者ラベル -> 現行話者ラベル の貪欲1対1対応表を作る.

    重なり時間(ms)が最大のペアから確定させていく（scipy不使用の貪欲法）。
    src/das/asr/live/_diarization.py の _best_speaker_mapping と同じ考え方。
    """
    overlap: Counter[tuple[str, str]] = Counter()
    for turn in turns:
        if not turn.has_timing:
            continue
        for seg in segments:
            ov = seg.overlap_ms(turn.ms, turn.end_ms)  # type: ignore[arg-type]
            if ov > 0:
                overlap[(seg.speaker, turn.speaker)] += ov

    mapping: dict[str, str] = {}
    used_current: set[str] = set()
    for (pyannote_sp, current_sp), _ov in overlap.most_common():
        if pyannote_sp in mapping or current_sp in used_current:
            continue
        mapping[pyannote_sp] = current_sp
        used_current.add(current_sp)
    return mapping


def dominant_pyannote_speaker(
    turn: Turn, segments: list[PyannoteSegment]
) -> tuple[str | None, float]:
    """1ターン区間に最も重なるpyannote話者と、そのターン長に対する重なり比率."""
    duration = max(1, turn.end_ms - turn.ms)  # type: ignore[operator]
    overlaps: Counter[str] = Counter()
    for seg in segments:
        ov = seg.overlap_ms(turn.ms, turn.end_ms)  # type: ignore[arg-type]
        if ov > 0:
            overlaps[seg.speaker] += ov
    if not overlaps:
        return None, 0.0
    speaker, ov_ms = overlaps.most_common(1)[0]
    return speaker, min(1.0, ov_ms / duration)


def compare_session(
    session_id: str,
    wav_path: Path,
    turns_path: Path,
    turns: list[Turn],
    segments: list[PyannoteSegment],
    params: dict[str, Any],
    unknown_label: str,
) -> SessionBenchResult:
    timed_turns = [t for t in turns if t.has_timing]
    mapping = greedy_speaker_mapping(timed_turns, segments)

    mismatches: list[TurnMismatch] = []
    unresolved: list[UnresolvedFinding] = []
    matched = 0
    compared = 0

    for turn in timed_turns:
        raw_speaker, ratio = dominant_pyannote_speaker(turn, segments)
        mapped_speaker = mapping.get(raw_speaker) if raw_speaker else None

        if turn.speaker == unknown_label:
            # (d) 現行が未確定扱いした区間を pyannote が割れているか
            covering: Counter[str] = Counter()
            for seg in segments:
                ov = seg.overlap_ms(turn.ms, turn.end_ms)  # type: ignore[arg-type]
                if ov > 0:
                    covering[seg.speaker] += ov
            raw_speakers = [sp for sp, _ in covering.most_common()]
            mapped_speakers = sorted({mapping.get(sp, sp) for sp in raw_speakers})
            duration = max(1, turn.end_ms - turn.ms)  # type: ignore[operator]
            covered_ratio = min(1.0, sum(covering.values()) / duration)
            unresolved.append(
                UnresolvedFinding(
                    turn_id=turn.turn_id,
                    ms=turn.ms,  # type: ignore[arg-type]
                    end_ms=turn.end_ms,  # type: ignore[arg-type]
                    text_head=turn.text[:20],
                    pyannote_speakers_mapped=mapped_speakers,
                    pyannote_speakers_raw=raw_speakers,
                    covered_ratio=covered_ratio,
                    split_into_multiple=len(raw_speakers) > 1,
                )
            )
            continue

        compared += 1
        is_match = mapped_speaker == turn.speaker
        if is_match:
            matched += 1
        else:
            mismatches.append(
                TurnMismatch(
                    turn_id=turn.turn_id,
                    ms=turn.ms,  # type: ignore[arg-type]
                    end_ms=turn.end_ms,  # type: ignore[arg-type]
                    current_speaker=turn.speaker,
                    pyannote_speaker_raw=raw_speaker,
                    pyannote_speaker_mapped=mapped_speaker,
                    overlap_ratio=ratio,
                    text_head=turn.text[:20],
                )
            )

    current_speakers = {
        t.speaker
        for t in timed_turns
        if t.speaker != unknown_label and t.speaker not in _SYNTHETIC_SPEAKERS
    }
    pyannote_speakers_raw = {seg.speaker for seg in segments}
    pyannote_speakers_mapped = {mapping.get(sp, sp) for sp in pyannote_speakers_raw}

    match_rate = matched / compared if compared else 0.0

    return SessionBenchResult(
        session=session_id,
        wav_path=str(wav_path),
        turns_path=str(turns_path),
        params=params,
        n_turns_total=len(turns),
        n_turns_timed=len(timed_turns),
        n_turns_compared=compared,
        n_matched=matched,
        match_rate=match_rate,
        current_speaker_count=len(current_speakers),
        pyannote_speaker_count_raw=len(pyannote_speakers_raw),
        pyannote_speaker_count_mapped=len(pyannote_speakers_mapped),
        speaker_count_matches=len(current_speakers) == len(pyannote_speakers_raw),
        speaker_mapping=mapping,
        mismatches=mismatches,
        unresolved_findings=unresolved,
        pyannote_segments=segments,
    )


# ---------------------------------------------------------------------------
# 出力
# ---------------------------------------------------------------------------


def print_summary(result: SessionBenchResult) -> None:
    print(f"\n=== {result.session} ===")
    print(f"  wav: {result.wav_path}")
    print(
        f"  ターン数: 全{result.n_turns_total} / タイミングあり{result.n_turns_timed} "
        f"/ 比較対象{result.n_turns_compared}"
    )
    print(
        f"  話者数: 現行 {result.current_speaker_count} / pyannote(生) "
        f"{result.pyannote_speaker_count_raw} / pyannote(マッピング後) "
        f"{result.pyannote_speaker_count_mapped}"
        f"  {'一致' if result.speaker_count_matches else '不一致'}"
    )
    print(f"  話者マッピング(pyannote→現行): {result.speaker_mapping}")
    print(
        f"  ターン単位一致率: {result.n_matched}/{result.n_turns_compared} "
        f"= {result.match_rate * 100:.1f}%"
    )
    if result.mismatches:
        print(f"  不一致ターン: {len(result.mismatches)}件（先頭5件）")
        for m in result.mismatches[:5]:
            print(
                f"    turn={m.turn_id} t={m.ms}-{m.end_ms}ms "
                f"現行={m.current_speaker} pyannote={m.pyannote_speaker_mapped}"
                f"(生:{m.pyannote_speaker_raw}) overlap={m.overlap_ratio:.2f} "
                f"text='{m.text_head}'"
            )
    if result.unresolved_findings:
        split = sum(1 for u in result.unresolved_findings if u.split_into_multiple)
        print(
            f"  現行『{DEFAULT_UNKNOWN_LABEL}』扱い区間: {len(result.unresolved_findings)}件 "
            f"（うちpyannoteが複数話者に分離: {split}件）"
        )
        for u in result.unresolved_findings[:5]:
            print(
                f"    turn={u.turn_id} t={u.ms}-{u.end_ms}ms "
                f"pyannote={u.pyannote_speakers_mapped} 被覆率={u.covered_ratio:.2f} "
                f"text='{u.text_head}'"
            )


def print_overall_summary(results: list[SessionBenchResult]) -> None:
    if not results:
        return
    print("\n=== 全体サマリ ===")
    total_compared = sum(r.n_turns_compared for r in results)
    total_matched = sum(r.n_matched for r in results)
    overall_rate = total_matched / total_compared if total_compared else 0.0
    print(f"  セッション数: {len(results)}")
    print(f"  ターン単位一致率(全体, マイクロ平均): {overall_rate * 100:.1f}%")
    per_session_rates = [r.match_rate for r in results]
    print(
        f"  ターン単位一致率(セッション平均): "
        f"{sum(per_session_rates) / len(per_session_rates) * 100:.1f}%"
    )
    n_speaker_count_ok = sum(1 for r in results if r.speaker_count_matches)
    print(f"  話者数一致セッション: {n_speaker_count_ok}/{len(results)}")
    total_unresolved = sum(len(r.unresolved_findings) for r in results)
    total_unresolved_split = sum(
        sum(1 for u in r.unresolved_findings if u.split_into_multiple) for r in results
    )
    if total_unresolved:
        print(
            f"  現行『未確定』区間: 全{total_unresolved}件 "
            f"（pyannoteが分離: {total_unresolved_split}件）"
        )


def write_detail_json(result: SessionBenchResult, out_path: Path) -> None:
    out_path.write_text(
        json.dumps(result.to_json_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--session",
        action="append",
        default=[],
        metavar="2026-06-12_1346",
        help="transcripts/<session>.wav と .turns.jsonl のペアを解決（複数指定可）",
    )
    ap.add_argument(
        "--wav",
        action="append",
        default=[],
        metavar="PATH",
        help="wavパスを直接指定（同名の .turns.jsonl を自動解決、複数指定可）",
    )
    ap.add_argument(
        "--from-json",
        action="append",
        default=[],
        metavar="PATH",
        help="pyannote応答(またはoutput/diarization配列)のモックJSON。"
        "--session/--wav の指定順に対応させる。件数が足りない分は実APIを叩く"
        "（--dry-run時は代わりに合成モックを使う）",
    )
    ap.add_argument("--dry-run", action="store_true", help="pyannote APIを実際には呼ばない")
    ap.add_argument("--api-key", default=None, help="pyannoteAI APIキー（省略時は環境変数）")
    ap.add_argument("--num-speakers", type=int, default=None, help="話者数が既知の場合に指定")
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    ap.add_argument(
        "--model",
        default="precision-2",
        choices=["precision-2", "community-1"],
        help="pyannoteAI diarizationモデル（既定: precision-2。2026-07-09時点で"
        "docs.pyannote.ai/models 記載の最新・最高精度モデル。APIのデフォルトも"
        "precision-2だが、将来のサーバ既定変更に左右されないよう明示指定する）",
    )
    ap.add_argument(
        "--unknown-label",
        default=DEFAULT_UNKNOWN_LABEL,
        help=f"現行システムの『不明/未確定』話者ラベル（既定: {DEFAULT_UNKNOWN_LABEL}）",
    )
    ap.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_S)
    ap.add_argument("--poll-timeout", type=float, default=DEFAULT_POLL_TIMEOUT_S)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=TRANSCRIPTS_DIR,
        help="詳細JSONの出力先ディレクトリ（既定: transcripts/）",
    )
    return ap


def _load_dotenv_fallback() -> None:
    """プロジェクトルートの .env から未設定の環境変数のみ読み込む（export不要にする）。"""
    import os

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def resolve_api_key(cli_value: str | None) -> str | None:
    import os

    if cli_value:
        return cli_value
    _load_dotenv_fallback()
    return os.environ.get("PYANNOTEAI_API_KEY") or os.environ.get("PYANNOTE_API_KEY")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not args.session and not args.wav:
        print("エラー: --session か --wav を最低1つ指定してください", file=sys.stderr)
        return 2

    targets = resolve_sessions(args.session, args.wav)
    from_json_paths = [Path(p) for p in args.from_json]

    api_key = resolve_api_key(args.api_key)
    if not args.dry_run and not from_json_paths and not api_key:
        print(
            "エラー: 実APIを叩くには PYANNOTEAI_API_KEY (または PYANNOTE_API_KEY) "
            "を設定するか、--dry-run / --from-json を使ってください",
            file=sys.stderr,
        )
        return 2

    params = {
        "model": args.model,
        "num_speakers": args.num_speakers,
        "min_speakers": args.min_speakers,
        "max_speakers": args.max_speakers,
        "dry_run": args.dry_run,
    }

    results: list[SessionBenchResult] = []
    for i, (session_id, wav_path, turns_path) in enumerate(targets):
        if not turns_path.exists():
            print(f"スキップ: {turns_path} が見つかりません", file=sys.stderr)
            continue

        turns = load_turns(turns_path)
        sr = wav_sample_rate(wav_path) if wav_path.exists() else None
        if sr is not None and sr != 16000:
            print(
                f"警告: {wav_path} のサンプルレートは {sr}Hz です（16kHz想定と異なる）",
                file=sys.stderr,
            )

        mock_path = from_json_paths[i] if i < len(from_json_paths) else None
        if mock_path is not None:
            payload = json.loads(mock_path.read_text(encoding="utf-8"))
            segments = parse_pyannote_segments(payload)
        elif args.dry_run:
            segments = build_dry_run_mock(turns)
        else:
            if not wav_path.exists():
                print(f"スキップ: {wav_path} が見つかりません", file=sys.stderr)
                continue
            output = run_pyannote_diarization(
                wav_path,
                api_key,  # type: ignore[arg-type]
                session_id=session_id,
                num_speakers=args.num_speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                model=args.model,
                poll_interval_s=args.poll_interval,
                poll_timeout_s=args.poll_timeout,
            )
            segments = parse_pyannote_segments(output)

        result = compare_session(
            session_id, wav_path, turns_path, turns, segments, params, args.unknown_label
        )
        results.append(result)
        print_summary(result)

        out_path = args.out_dir / f"{session_id}.pyannote_bench.json"
        write_detail_json(result, out_path)
        print(f"  詳細JSON: {out_path}")

    print_overall_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
