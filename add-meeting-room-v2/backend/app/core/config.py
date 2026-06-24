from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    meeting_id: str = Field(default="meeting_001", min_length=1)

    openai_api_key: str | None = None
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_realtime_websocket_url: str = "wss://api.openai.com/v1/realtime"
    openai_realtime_model: str = "gpt-realtime-2"
    openai_realtime_voice: str = "marin"
    openai_realtime_reasoning_effort: Literal["minimal", "low", "medium", "high"] = (
        "low"
    )
    openai_realtime_output_sample_rate: int = Field(default=24000, gt=0)

    voice_agent_enabled: bool = False
    voice_agent_silence_ms: int = Field(default=1500, ge=100)
    voice_agent_cooldown_ms: int = Field(default=15000, ge=0)
    voice_agent_post_agent_mute_ms: int = Field(default=5000, ge=0)
    voice_agent_min_transcript_chars: int = Field(default=8, ge=1)
    voice_agent_realtime_timeout_ms: int = Field(default=30000, ge=1000)
    voice_agent_barge_in_enabled: bool = True
    voice_agent_barge_in_min_chars: int = Field(default=2, ge=1)
    speaker_intro_window_ms: int = Field(default=8000, ge=1000)

    speaker_verification_backend: Literal[
        "off",
        "speechbrain",
        "wespeaker",
        "both",
    ] = "off"
    speaker_verification_ring_buffer_ms: int = Field(default=120000, ge=10000)
    speaker_verification_min_enrollment_ms: int = Field(default=2000, ge=500)
    speaker_verification_min_evaluation_ms: int = Field(default=1500, ge=500)
    speaker_verification_queue_size: int = Field(default=32, ge=1)
    speaker_verification_device: str = "cpu"
    speaker_verification_speechbrain_model: str = (
        "speechbrain/spkrec-ecapa-voxceleb"
    )
    speaker_verification_wespeaker_model: str = "campplus"
    speaker_verification_results_path: str = (
        "app/experiments/experiment-speaker-verification/results/evaluations.jsonl"
    )

    soniox_api_key: str | None = None
    soniox_websocket_url: str = "wss://stt-rt.soniox.com/transcribe-websocket"
    soniox_model: str = "stt-rt-v5"
    soniox_language_hints: str = "ja"
    soniox_enable_speaker_diarization: bool = True
    soniox_enable_endpoint_detection: bool = True
    soniox_max_endpoint_delay_ms: int = Field(default=1000, ge=500, le=3000)

    audio_format: Literal["pcm_s16le"] = "pcm_s16le"
    audio_sample_rate: int = Field(default=16000, gt=0)
    audio_channels: int = Field(default=1, gt=0)
    audio_frame_ms: int = Field(default=100, gt=0)
    audio_input_device: int | str | None = None

    def language_hints_list(self) -> list[str]:
        return [
            item.strip()
            for item in self.soniox_language_hints.split(",")
            if item.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
