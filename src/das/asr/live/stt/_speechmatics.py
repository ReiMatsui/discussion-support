"""Speechmatics リアルタイムSTTバックエンド."""
from __future__ import annotations

import json

from .._constants import SR

SM_WS_URL = "wss://eu.rt.speechmatics.com/v2/"


class SpeechmaticsBackend:
    """Speechmatics WebSocket STTプロバイダ.

    受信メッセージをSoniox互換の内部トークン形式に変換する。
    話者ラベル: S1→"1"（表示は話者1）、不明UUはそのまま。
    """

    def __init__(self, api_key: str):
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "speechmatics"

    def ws_url(self) -> str:
        return SM_WS_URL

    def ws_headers(self) -> dict[str, str] | None:
        return {"Authorization": f"Bearer {self._api_key}"}

    def start_message(self, model: str, lang: str) -> dict:
        return {
            "message": "StartRecognition",
            "audio_format": {
                "type": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": SR,
            },
            "transcription_config": {
                "language": lang,
                "operating_point": "enhanced",
                "diarization": "speaker",
                "enable_partials": True,
                "max_delay": 1.2,
                "conversation_config": {"end_of_utterance_silence_trigger": 0.8},
            },
        }

    def parse_message(self, raw: dict, lang: str) -> dict:
        """SpeechmaticsのRTメッセージを内部トークン形式に変換."""
        m = raw.get("message")
        if m == "Error":
            return {"error_code": raw.get("type"),
                    "error_message": raw.get("reason")}
        if m == "EndOfTranscript":
            return {"finished": True, "tokens": []}
        if m == "EndOfUtterance":
            return {"tokens": [{"text": "<end>", "is_final": True}]}
        if m in ("AddTranscript", "AddPartialTranscript"):
            final = m == "AddTranscript"
            toks: list[dict] = []
            for r in raw.get("results", []):
                alts = r.get("alternatives") or []
                content = alts[0].get("content", "") if alts else ""
                if not content:
                    continue
                spk = alts[0].get("speaker") or "UU"
                if spk.startswith("S") and spk[1:].isdigit():
                    spk = spk[1:]
                if (lang not in ("ja", "zh", "cmn", "yue") and toks
                        and r.get("type") == "word"):
                    content = " " + content
                toks.append({
                    "text": content,
                    "speaker": spk,
                    "start_ms": int(r["start_time"] * 1000),
                    "end_ms": int(r["end_time"] * 1000),
                    "is_final": final,
                })
            return {"tokens": toks}
        return {"tokens": []}

    def make_end_message(self, seq: int) -> str | bytes:
        return json.dumps({"message": "EndOfStream", "last_seq_no": seq})
