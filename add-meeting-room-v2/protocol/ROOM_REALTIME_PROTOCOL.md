# ROOM_REALTIME_PROTOCOL.md

## Purpose

This document defines the initial real-time protocol for `meeting-room-v2`.

The first version is intentionally small. It covers only the backend-to-viewer transcript events produced from Soniox real-time speech-to-text with speaker diarization.

## Scope

The initial runtime shape is:

```text
Mac room microphone
  -> backend audio capture
  -> Soniox realtime WebSocket
  -> backend normalized transcript events
  -> frontend viewer
```

Raw Soniox payloads are not part of the frontend protocol. The backend normalizes Soniox tokens into project-owned events before broadcasting them.

The initial viewer transport is a backend WebSocket endpoint:

```text
GET /ws/viewer
```

This endpoint is for the initial local viewer. It may be replaced by a different frontend or transport in production, but the normalized event shapes in this document should remain project-owned.

## Non-Goals

- iPhone device clients.
- Browser microphone capture.
- Raw audio recording storage.
- Raw Soniox event forwarding to the frontend.
- Participant identity confirmation.
- Visual features, face images, raw video, or meeting video.
- Device-based active speaker estimation.

## Common Fields

All backend-to-viewer events include:

```text
type
meeting_id
server_timestamp_ms
```

Transcript events additionally include:

```text
meeting_id
segment_id
text
speaker_label
participant_id
display_name
speaker_status
is_final
endpoint_detected
server_timestamp_ms
```

`server_timestamp_ms` is the backend wall-clock time in milliseconds when the normalized event is emitted.

`speaker_label` is the provider-derived diarization label, such as `"1"` or `"2"`. It is not a participant identity. A later assignment flow may map it to a participant, but the mapping must remain editable.

`participant_id`, `display_name`, and `speaker_status` are added by the backend speaker registry. Unknown Soniox clusters must not create new participants automatically.

## session.started

Sent when the backend starts a room microphone transcription session.

```json
{
  "type": "session.started",
  "meeting_id": "meeting_001",
  "soniox_model": "stt-rt-v5",
  "sample_rate": 16000,
  "frame_ms": 100,
  "server_timestamp_ms": 1710000000000
}
```

## session.ended

Sent when the backend stops the room microphone transcription session.

```json
{
  "type": "session.ended",
  "meeting_id": "meeting_001",
  "reason": "shutdown",
  "server_timestamp_ms": 1710000000000
}
```

## transcription.error

Sent when microphone capture, Soniox connection, or transcription processing fails.

```json
{
  "type": "transcription.error",
  "meeting_id": "meeting_001",
  "message": "SONIOX_API_KEY is not set",
  "server_timestamp_ms": 1710000000000
}
```

## transcript.delta

Partial transcript for a segment.

```json
{
  "type": "transcript.delta",
  "meeting_id": "meeting_001",
  "segment_id": "seg_000001",
  "speaker_label": "1",
  "participant_id": "p_001",
  "display_name": "田中",
  "speaker_status": "mapped",
  "text": "今日の議題は",
  "is_final": false,
  "endpoint_detected": false,
  "start_ms": 1200,
  "end_ms": 2400,
  "server_timestamp_ms": 1710000000000
}
```

Fields:

```text
type: "transcript.delta"
meeting_id: meeting/session id
segment_id: backend-assigned stable segment id
speaker_label: Soniox speaker label normalized as a string
participant_id: mapped participant id, or null
display_name: mapped participant display name, or null
speaker_status: "mapped" or "unassigned"
text: partial text for this segment
is_final: false
endpoint_detected: whether Soniox endpoint detection appeared in this response
start_ms: optional audio-relative start time from provider
end_ms: optional audio-relative end time from provider
server_timestamp_ms: backend emission time
```

The frontend should replace the current display for the same `segment_id` when a newer `transcript.delta` arrives.

## transcript.final

Final transcript for a segment.

```json
{
  "type": "transcript.final",
  "meeting_id": "meeting_001",
  "segment_id": "seg_000001",
  "speaker_label": "1",
  "participant_id": "p_001",
  "display_name": "田中",
  "speaker_status": "mapped",
  "text": "今日の議題は予算についてです。",
  "is_final": true,
  "endpoint_detected": true,
  "start_ms": 1200,
  "end_ms": 4300,
  "server_timestamp_ms": 1710000001200
}
```

Fields:

```text
type: "transcript.final"
meeting_id: meeting/session id
segment_id: backend-assigned stable segment id
speaker_label: Soniox speaker label normalized as a string
participant_id: mapped participant id, or null
display_name: mapped participant display name, or null
speaker_status: "mapped" or "unassigned"
text: final text for this segment
is_final: true
endpoint_detected: whether Soniox endpoint detection appeared in this response
start_ms: optional audio-relative start time from provider
end_ms: optional audio-relative end time from provider
server_timestamp_ms: backend emission time
```

When `transcript.final` arrives, the frontend should replace any existing delta for the same `segment_id` and mark the segment as final.

Soniox may return a special `<end>` token when endpoint detection decides that an utterance has ended. The backend must not include `<end>` in transcript `text`. Instead, it exposes the signal through `endpoint_detected`. This field is intended for later voice agent turn-taking logic.

## Speaker Label Rules

- `speaker_label` comes from Soniox diarization output.
- The label must be treated as a temporary speaker cluster, not as identity proof.
- The label is represented as a string even if the provider returns a number.
- Unknown speaker labels should be represented as `null` only if Soniox does not provide a label.
- Unknown speaker clusters must not automatically create participants.
- A speaker label can be mapped to a participant only by manual bind or a confirmed self-introduction candidate.

## Speaker Setup Commands

The experimental viewer may send setup commands over the same `/ws/viewer` socket.

### participant.list.update

```json
{
  "type": "participant.list.update",
  "meeting_id": "meeting_001",
  "participants": [
    {"participant_id": "p_001", "display_name": "田中", "role": "human"},
    {"participant_id": "agent", "display_name": "AI agent", "role": "agent"}
  ]
}
```

### speaker.intro.start

Starts a self-introduction window. The backend snapshots currently observed Soniox speaker labels. New labels observed during the window are emitted as candidates, not automatically bound.

```json
{
  "type": "speaker.intro.start",
  "meeting_id": "meeting_001",
  "participant_id": "p_001"
}
```

### speaker.bind

Confirms a candidate or manually maps a Soniox speaker label to a participant.

```json
{
  "type": "speaker.bind",
  "meeting_id": "meeting_001",
  "speaker_label": "1",
  "participant_id": "p_001"
}
```

## Speaker Setup Events

```json
{
  "type": "speaker.intro.started",
  "meeting_id": "meeting_001",
  "participant_id": "p_001",
  "display_name": "田中",
  "role": "human",
  "known_speaker_labels": [],
  "expires_at_ms": 1710000008000,
  "server_timestamp_ms": 1710000000000
}
```

```json
{
  "type": "speaker.intro.candidate_detected",
  "meeting_id": "meeting_001",
  "participant_id": "p_001",
  "speaker_label": "1",
  "candidates": ["1"],
  "server_timestamp_ms": 1710000003000
}
```

```json
{
  "type": "speaker.map.updated",
  "meeting_id": "meeting_001",
  "speaker_map": {
    "1": {
      "participant_id": "p_001",
      "display_name": "田中",
      "role": "human",
      "source": "self_intro"
    }
  },
  "server_timestamp_ms": 1710000004000
}
```

## Speaker Stats

`speaker.stats.updated` is emitted after `transcript.final` events outside the
self-introduction setup flow. Deltas are not counted.

```json
{
  "type": "speaker.stats.updated",
  "meeting_id": "meeting_001",
  "stats": [
    {
      "speaker_key": "participant:p_001",
      "participant_id": "p_001",
      "display_name": "田中",
      "speaker_label": "1",
      "utterance_count": 12,
      "text_chars": 430,
      "estimated_speech_ms": 85000,
      "last_spoke_at_ms": 1710000000000
    }
  ],
  "server_timestamp_ms": 1710000000000
}
```

Fields:

```text
utterance_count: count of final transcript segments
text_chars: total character count from final transcript text
estimated_speech_ms: sum of end_ms - start_ms when both are available
```

## Voice Agent Barge-In

Voice agent interruption is backend-internal and does not forward raw OpenAI
events to the viewer.

The backend interrupts an active agent response when:

```text
voice agent is speaking
speaker_label is mapped to a human participant
transcript text reaches the configured minimum character count
```

The backend must:

```text
send response.cancel to the Realtime session
abort local Mac audio playback
keep the human transcript for subsequent voice agent context
not apply the normal post-agent audio mute delay
```

Mapped AI speakers and unassigned Soniox clusters must not trigger interruption.

## Speaker Verification Evaluation Events

Speaker verification is an optional evaluation-only backend feature. It must not
block Soniox streaming or automatically change speaker mapping.

`speaker.verification.profile.ready` is emitted when self-introduction or early
mapped audio has produced a participant profile for one backend.

`speaker.verification.evaluated` contains per-backend scores and latency:

```json
{
  "type": "speaker.verification.evaluated",
  "meeting_id": "meeting_001",
  "segment_ids": ["seg_000120"],
  "speaker_label": "1",
  "expected_participant_id": "p_001",
  "audio_duration_ms": 3200,
  "backends": {
    "speechbrain": {
      "status": "ok",
      "scores": {"p_001": 0.86, "p_002": 0.42},
      "top1_participant_id": "p_001",
      "top1_score": 0.86,
      "margin": 0.44,
      "load_ms": 0.0,
      "inference_ms": 170.0,
      "correct": true
    },
    "wespeaker": {
      "status": "ok",
      "scores": {"p_001": 0.90, "p_002": 0.47},
      "top1_participant_id": "p_001",
      "top1_score": 0.90,
      "margin": 0.43,
      "load_ms": 0.0,
      "inference_ms": 90.0,
      "correct": true
    }
  },
  "server_timestamp_ms": 1710000000000
}
```

For an unassigned Soniox cluster, `expected_participant_id` and `correct` are
`null`. The event is also written to development JSONL when evaluation is
enabled.

## Raw Provider Payloads

The frontend protocol must not depend on raw Soniox payload shape.

If raw provider debugging becomes necessary later, add a separate development-only event that includes only safe metadata, not audio and not full provider payloads.

## Storage Policy

Initial storage policy:

```text
raw audio: not stored
normalized transcript events: memory only for viewer replay
development JSONL logs: written only when an explicit CLI option requests it
```

The experimental frontend must not add storage. Persistent transcript storage is a later product decision.
