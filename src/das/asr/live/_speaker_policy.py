"""リアルタイム介入で使う話者ラベルの扱い."""
from __future__ import annotations

from typing import Any

from ._constants import _BACKCHANNEL_RE, AGENT_SPEAKER, UNSURE_SPEAKER
from ._speaker_keys import is_provisional_key

_GENERIC_SPEAKER = "発話者"
_UNSURE_SPEAKER_LABELS = {UNSURE_SPEAKER, "未確定"}


def is_intervention_signal(record: dict[str, Any]) -> bool:
    """AI介入の判断材料にしてよい発話かを返す.

    話者が怪しくても、内容のある発話は「場の発言」として使う。相づちや未確定の
    短発話は、議論の流れを動かす材料にしない。
    """
    text = str(record.get("text") or "").strip()
    if not text:
        return False
    if record.get("bc") or record.get("speaker") in _UNSURE_SPEAKER_LABELS:
        return False
    return not _BACKCHANNEL_RE.match(text)


def is_triage_signal(record: dict[str, Any]) -> bool:
    """triage（呼びかけ検出）の対象にしてよい発話かを返す.

    ``is_intervention_signal`` から**未確定話者の除外だけを外した**判定。
    ファシリテーターへの呼びかけ検出は話者の同一性に依存しない操作なので、
    声紋未登録の未確定話者の発話も分類対象に含める。fact / drift / invite の
    ように「誰が言ったか」が効く用途は、従来どおり ``is_intervention_signal``
    側で未確定を除外して制限される。
    """
    text = str(record.get("text") or "").strip()
    if not text:
        return False
    if record.get("bc"):
        return False
    return not _BACKCHANNEL_RE.match(text)


def is_reliable_human_speaker(record: dict[str, Any]) -> bool:
    """AIが個人として扱ってよい話者かを返す.

    低信頼なSTT fallbackや未確定は、議論内容としては使うが、個人名の声かけ・
    発言量判定には使わない。誤った個人名でAIが介入するリスクを下げる。
    """
    speaker = record.get("speaker")
    if speaker in {None, AGENT_SPEAKER, "パートナー", *_UNSURE_SPEAKER_LABELS}:
        return False
    if record.get("speaker_source") == "stt_fallback":
        return False
    # 「暫定キーか」の判定は _speaker_keys に一本化（従来ここに3つ目の
    # 同じ述語の写しがあった）。
    return not is_provisional_key(speaker)


def intervention_speaker_name(state, record: dict[str, Any]) -> str:
    """AIエージェントへ渡す話者名。低信頼なら汎用名に落とす."""
    if is_reliable_human_speaker(record):
        return state.disp_name(record.get("speaker", ""))
    return _GENERIC_SPEAKER


def reliable_human_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if is_intervention_signal(r) and is_reliable_human_speaker(r)]


def intervention_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if is_intervention_signal(r)]


def triage_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """triage（呼びかけ検出）用フィルタ。未確定話者も含める（``is_triage_signal``）."""
    return [r for r in records if is_triage_signal(r)]
