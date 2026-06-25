"""STT送信済みPCMバッファのテスト."""
from __future__ import annotations

import datetime
from typing import Any

from das.asr.live._session_state import SessionState
from das.asr.live._workers import _run_sender


class _Backend:
    def make_end_message(self, seq: int) -> str:
        return f"END:{seq}"


class _WS:
    def __init__(self, *, fail_first: bool = False) -> None:
        self.fail_first = fail_first
        self.sent: list[bytes | str] = []

    def send(self, payload: bytes | str) -> None:
        if self.fail_first:
            self.fail_first = False
            raise RuntimeError("send failed")
        self.sent.append(payload)


class _DiarizationProvider:
    name = "fake"

    def __init__(self) -> None:
        self.audio: list[bytes] = []
        self.drains = 0

    def start(self) -> None:
        pass

    def send_audio(self, pcm16k: bytes) -> None:
        self.audio.append(pcm16k)

    def drain_events(self) -> list[Any]:
        self.drains += 1
        return []

    def close(self) -> None:
        pass


def _make_state() -> SessionState:
    return SessionState(
        args=object(),
        started=datetime.datetime(2026, 1, 1),
        out_path="/tmp/o.md",
        html_path="/tmp/o.html",
        diag_path="/tmp/o.diag",
        turns_path="/tmp/o.turns",
        wav_path="/tmp/o.wav",
        serve=False,
    )


def test_sender_keeps_only_successfully_sent_audio_for_asr_buffer() -> None:
    state = _make_state()
    provider = _DiarizationProvider()
    state.diarization_provider = provider
    ws = _WS(fail_first=True)
    state.stt_ws = ws  # type: ignore[assignment]
    first = b"\x01\x00" * 1600
    second = b"\x02\x00" * 1600

    state.audio_q.put(first)
    state.audio_q.put(second)
    state.audio_q.put(None)
    _run_sender(state, _Backend())  # type: ignore[arg-type]

    assert state.pcm_buf == first + second
    assert state.pcm_total_bytes == len(first) + len(second)
    assert state.asr_pcm_buf == second
    assert state.asr_pcm_total_bytes == len(second)
    assert ws.sent == [second, "END:1"]
    assert provider.audio == [second]
    assert provider.drains == 1


def test_open_wav_resets_asr_buffer_offsets() -> None:
    state = _make_state()
    state.asr_pcm_buf.extend(b"abc")
    state.asr_pcm_buf_offset = 10
    state.asr_pcm_total_bytes = 3

    state.open_wav()  # type: ignore[no-untyped-call]

    assert state.asr_pcm_buf == bytearray()
    assert state.asr_pcm_buf_offset == 0
    assert state.asr_pcm_total_bytes == 0
