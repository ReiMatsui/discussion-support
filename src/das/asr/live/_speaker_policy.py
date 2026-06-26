"""リアルタイム介入で使う話者ラベルの扱い."""
from __future__ import annotations

from typing import Any

from ._constants import AGENT_SPEAKER, UNSURE_SPEAKER

_GENERIC_SPEAKER = "発話者"


def is_reliable_human_speaker(record: dict[str, Any]) -> bool:
    """AIが個人として扱ってよい話者かを返す.

    低信頼なSTT fallbackや未確定は、議論内容としては使うが、個人名の声かけ・
    発言量判定には使わない。誤った個人名でAIが介入するリスクを下げる。
    """
    speaker = record.get("speaker")
    if speaker in {None, AGENT_SPEAKER, "パートナー", UNSURE_SPEAKER}:
        return False
    if record.get("speaker_source") == "stt_fallback":
        return False
    return not str(speaker).startswith(("#", "@diar:"))


def intervention_speaker_name(state, record: dict[str, Any]) -> str:
    """AIエージェントへ渡す話者名。低信頼なら汎用名に落とす."""
    if is_reliable_human_speaker(record):
        return state.disp_name(record.get("speaker", ""))
    return _GENERIC_SPEAKER


def reliable_human_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if is_reliable_human_speaker(r)]
