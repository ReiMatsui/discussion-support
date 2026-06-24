import base64
import binascii
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PCM_S16LE_BYTES_PER_SAMPLE = 2


def current_timestamp_ms() -> int:
    return time.time_ns() // 1_000_000


class AudioConfig(BaseModel):
    format: Literal["pcm_s16le"]
    sample_rate: Literal[24000]
    channels: Literal[1]
    chunk_duration_ms: int = Field(gt=0)


class VisualConfig(BaseModel):
    enabled: bool
    feature_rate_hz: int = Field(ge=0)
    sends_raw_video: Literal[False]
    features: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_feature_rate(self) -> "VisualConfig":
        if self.enabled and self.feature_rate_hz <= 0:
            raise ValueError("visual.enabled=trueの場合はfeature_rate_hzが必要です")
        return self


class ClockInfo(BaseModel):
    client_wall_time_ms: int = Field(ge=0)
    client_monotonic_time_ns: int = Field(ge=0)


class DeviceJoinMessage(BaseModel):
    type: Literal["device.join"]
    protocol_version: Literal[1]
    meeting_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    stream_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    audio: AudioConfig
    visual: VisualConfig
    clock: ClockInfo


class AudioPayload(BaseModel):
    format: Literal["pcm_s16le"]
    sample_rate: Literal[24000]
    channels: Literal[1]
    base64: str = Field(min_length=1)

    @field_validator("base64")
    @classmethod
    def validate_base64(cls, value: str) -> str:
        try:
            base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("audio.base64は有効なbase64である必要があります") from exc
        return value

    def decoded_bytes(self) -> bytes:
        return base64.b64decode(self.base64, validate=True)


class AudioFeatures(BaseModel):
    rms_dbfs: float | None = None
    peak_dbfs: float | None = None
    local_voice_activity_score: float | None = Field(default=None, ge=0.0, le=1.0)
    clipping_detected: bool | None = None


class AudioChunkMessage(BaseModel):
    type: Literal["audio.chunk"]
    protocol_version: Literal[1]
    meeting_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    stream_id: str = Field(min_length=1)
    seq: int = Field(ge=0)
    client_capture_time_ms: int = Field(ge=0)
    client_monotonic_time_ns: int = Field(ge=0)
    duration_ms: int = Field(gt=0)
    audio: AudioPayload
    audio_features: AudioFeatures | None = None

    @model_validator(mode="after")
    def validate_pcm_byte_length(self) -> "AudioChunkMessage":
        expected_bytes = (
            self.audio.sample_rate
            * self.audio.channels
            * PCM_S16LE_BYTES_PER_SAMPLE
            * self.duration_ms
        ) // 1000
        actual_bytes = len(self.audio.decoded_bytes())
        if actual_bytes != expected_bytes:
            raise ValueError(
                "audio.base64のbyte数がduration_ms、sample_rate、channelsと"
                f"一致しません expected_bytes={expected_bytes} actual_bytes={actual_bytes}"
            )
        return self


class HeadPose(BaseModel):
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None


class VisualFeatureValues(BaseModel):
    model_config = ConfigDict(extra="allow")

    face_detected: bool | None = None
    face_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    face_count: int | None = Field(default=None, ge=0)
    mouth_open_score: float | None = Field(default=None, ge=0.0, le=1.0)
    mouth_motion_score: float | None = Field(default=None, ge=0.0, le=1.0)
    head_pose: HeadPose | None = None
    body_pose_detected: bool | None = None
    hand_motion_score: float | None = Field(default=None, ge=0.0, le=1.0)


class VisualFeaturesMessage(BaseModel):
    type: Literal["visual.features"]
    protocol_version: Literal[1]
    meeting_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    stream_id: str = Field(min_length=1)
    seq: int = Field(ge=0)
    client_capture_time_ms: int = Field(ge=0)
    client_monotonic_time_ns: int = Field(ge=0)
    window_ms: int = Field(gt=0)
    features: VisualFeatureValues


class DeviceConnectedEvent(BaseModel):
    type: Literal["device.connected"] = "device.connected"
    meeting_id: str
    device_id: str
    stream_id: str
    user_id: str
    display_name: str
    server_timestamp_ms: int


class DeviceAudioReceivedEvent(BaseModel):
    type: Literal["device.audio_received"] = "device.audio_received"
    meeting_id: str
    device_id: str
    stream_id: str
    user_id: str
    display_name: str
    seq: int
    client_capture_time_ms: int
    client_monotonic_time_ns: int
    server_received_ms: int
    duration_ms: int
    seq_status: Literal["ok", "gap", "duplicate", "out_of_order"]
    expected_seq: int | None = None
    audio_features: AudioFeatures | None = None


class DeviceVisualReceivedEvent(BaseModel):
    type: Literal["device.visual_received"] = "device.visual_received"
    meeting_id: str
    device_id: str
    stream_id: str
    user_id: str
    display_name: str
    seq: int
    client_capture_time_ms: int
    client_monotonic_time_ns: int
    server_received_ms: int
    window_ms: int
    seq_status: Literal["ok", "gap", "duplicate", "out_of_order"]
    expected_seq: int | None = None
    features: VisualFeatureValues | None = None


class DeviceDisconnectedEvent(BaseModel):
    type: Literal["device.disconnected"] = "device.disconnected"
    meeting_id: str
    device_id: str
    stream_id: str
    user_id: str
    display_name: str
    server_timestamp_ms: int


class TranscriptionConnectedEvent(BaseModel):
    type: Literal["transcription.connected"] = "transcription.connected"
    meeting_id: str
    device_id: str
    stream_id: str
    user_id: str
    display_name: str
    model: str
    server_timestamp_ms: int = Field(default_factory=current_timestamp_ms)


class TranscriptionErrorEvent(BaseModel):
    type: Literal["transcription.error"] = "transcription.error"
    meeting_id: str
    device_id: str
    stream_id: str
    user_id: str
    display_name: str
    message: str
    server_timestamp_ms: int = Field(default_factory=current_timestamp_ms)


class TranscriptDeltaEvent(BaseModel):
    type: Literal["transcript.delta"] = "transcript.delta"
    meeting_id: str
    device_id: str
    stream_id: str
    user_id: str
    display_name: str
    segment_id: str
    text: str
    is_final: Literal[False] = False
    start_ms: int | None = None
    end_ms: int | None = None
    server_timestamp_ms: int = Field(default_factory=current_timestamp_ms)


class TranscriptFinalEvent(BaseModel):
    type: Literal["transcript.final"] = "transcript.final"
    meeting_id: str
    device_id: str
    stream_id: str
    user_id: str
    display_name: str
    segment_id: str
    text: str
    is_final: Literal[True] = True
    start_ms: int | None = None
    end_ms: int | None = None
    server_timestamp_ms: int = Field(default_factory=current_timestamp_ms)


class TranscriptRefinedEvent(BaseModel):
    type: Literal["transcript.refined"] = "transcript.refined"
    meeting_id: str
    device_id: str
    stream_id: str
    user_id: str
    display_name: str
    refined_full_text: str
    server_timestamp_ms: int = Field(default_factory=current_timestamp_ms)


class TranscriptionDisconnectedEvent(BaseModel):
    type: Literal["transcription.disconnected"] = "transcription.disconnected"
    meeting_id: str
    device_id: str
    stream_id: str
    user_id: str
    display_name: str
    server_timestamp_ms: int = Field(default_factory=current_timestamp_ms)


class TranscriptionRawEvent(BaseModel):
    type: Literal["transcription.raw_event"] = "transcription.raw_event"
    meeting_id: str
    device_id: str
    stream_id: str
    event_type: str
    server_timestamp_ms: int = Field(default_factory=current_timestamp_ms)


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str


class DeviceSpeakerEntry(BaseModel):
    device_id: str
    user_id: str
    display_name: str
    audio_score: float = Field(ge=0.0, le=1.0)
    visual_score: float = Field(ge=0.0, le=1.0)
    face_detected: bool
    speaker_confidence: float = Field(ge=0.0, le=1.0)


class ActiveSpeaker(BaseModel):
    device_id: str
    user_id: str
    display_name: str
    speaker_confidence: float = Field(ge=0.0, le=1.0)


class SpeakerEstimationEvent(BaseModel):
    type: Literal["speaker.estimation"] = "speaker.estimation"
    meeting_id: str
    server_timestamp_ms: int = Field(default_factory=current_timestamp_ms)
    window_ms: int = Field(gt=0)
    devices: list[DeviceSpeakerEntry]
    active_speaker: ActiveSpeaker | None = None
    raw_active_speaker: ActiveSpeaker | None = None
