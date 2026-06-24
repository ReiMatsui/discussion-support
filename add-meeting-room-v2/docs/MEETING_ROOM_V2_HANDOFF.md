# meeting-room-v2 handoff

## Purpose

`meeting-room-v2` is a new repository for a room-microphone based real-time meeting transcript system.

The v1 iPhone, visual feature, and device-based speaker estimation design is not carried forward as initial architecture.

## Initial Scope

- Use one room microphone first. A Mac built-in microphone is acceptable for the first PoC.
- Use Soniox as the primary real-time speech-to-text and speaker diarization provider.
- Do not store raw audio in the initial version.
- Show live transcripts with Soniox speaker labels such as `Speaker 1`.
- Build the voice agent on top of Soniox speaker-separated transcript history, after the Soniox path works.

## Out Of Scope Initially

- iPhone device clients.
- Raw video, face images, or meeting video.
- Visual features such as mouth motion or face detection.
- Device-based active speaker estimation.
- Participant identity confirmation.
- Long-term audio recording storage.
- Provider abstraction beyond Soniox.

## Core Data Model

Keep speaker label and participant identity separate.

```text
speaker_label: provider label such as "1" or "2"
participant_id: app-level participant id, optional
display_name: human-readable participant name, optional
assignment_source: unknown | self_intro | manual
```

Soniox speaker labels are not identity proof. A later self-introduction or manual assignment flow may map `Speaker 1` to a participant name, but the mapping must remain editable.

## Initial Audio Capture Choice

For the first Soniox PoC, prefer backend-side Mac microphone capture.

Reason:

- It removes the browser-to-backend audio hop.
- It avoids browser audio encoding and WebSocket forwarding work.
- It is easier to feed raw PCM frames directly to Soniox.
- It should have lower end-to-end latency for a local Mac room-mic demo.

Tradeoffs:

- macOS microphone permission and local audio dependencies must be handled.
- It is less portable than browser capture.
- It assumes the backend process runs on the machine with the microphone.

Browser microphone capture can be added later when remote participants, web-only operation, or easier deployment matter more than the shortest local path.

## Minimal Runtime Shape

```text
Mac microphone
  -> backend audio capture
  -> Soniox realtime WebSocket
  -> backend normalized transcript events
  -> frontend viewer
```

The backend should normalize Soniox tokens before sending events to the frontend. The frontend should not depend directly on Soniox raw response shapes.

## First Milestone

M0: Soniox realtime room-mic transcript.

- Start a backend process on Mac.
- Capture microphone audio without saving it.
- Stream audio to Soniox realtime STT with speaker diarization enabled.
- Convert Soniox token results into minimal transcript events.
- Show live `Speaker N` transcript in the viewer.
- Keep enough speaker-labeled transcript history for the later voice agent.

## Carry From v1

- Keep protocol-first thinking.
- Keep backend-to-viewer real-time event delivery.
- Keep the concept of `transcript.delta` and `transcript.final`.
- Reuse code only after the v2 core shape is proven.

## Do Not Carry From v1

- `device_id` as a speaker candidate.
- `visual.features`.
- `speaker.estimation` based on audio plus mouth movement.
- iOS-first milestones.
- Calibration logic for visual speaker estimation.
