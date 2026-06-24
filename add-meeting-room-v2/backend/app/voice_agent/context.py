from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TranscriptTurn:
    speaker_label: str | None
    display_name: str | None
    text: str
    server_timestamp_ms: int


@dataclass(frozen=True, slots=True)
class VoiceAgentContext:
    meeting_id: str
    recent_turns: list[TranscriptTurn]

    def latest_text(self) -> str:
        if not self.recent_turns:
            return ""
        return self.recent_turns[-1].text

    def prompt_text(self) -> str:
        lines = [
            "以下は会議室の直近会話です。",
            "沈黙が発生したため、必要なら短く自然にファシリテーションしてください。",
            "",
        ]
        for turn in self.recent_turns:
            speaker = (
                turn.display_name
                or (f"Speaker {turn.speaker_label}" if turn.speaker_label else None)
                or "Speaker ?"
            )
            lines.append(f"{speaker}: {turn.text}")
        lines.extend(
            [
                "",
                "返答条件:",
                "- 日本語で1文、長くても2文。",
                "- 直近の内容に沿って、確認・整理・次の一歩のどれかだけを話す。",
                "- 新しい論点を勝手に増やさない。",
            ]
        )
        return "\n".join(lines)
