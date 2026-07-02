# Prompt: closed roster speaker policy

You are implementing a focused speaker-identification UX fix in this repository.

## Goal

Make the voice-profile "名簿を確定" mode behave like a closed roster:

- If users pre-register 3 voices and lock the roster, future human utterances should be attributed only to those registered names when voiceprint confidence is sufficient.
- If confidence is not sufficient, the utterance should become "未確定", not a new anonymous participant such as "参加者A" or "人物N".
- Do not force every utterance into one of the registered names. A wrong confident-looking attribution is worse than "未確定".

In short:

> Closed roster means "registered people or unknown", not "registered people plus new anonymous speakers".

## Current Behavior To Verify

Relevant files:

- `src/das/asr/live/_voice_profiles.py`
- `src/das/asr/live/_recv_loop.py`
- `src/das/asr/live/_session_state.py`
- `src/das/asr/live/_ui.py`
- `tests/unit/live/test_voice_shortturn.py`
- `tests/unit/live/test_voice_unconfirmed.py`
- `tests/unit/live/test_ui_api.py`

Important current mechanics:

- `/api/roster` in `_ui.py` toggles `s.tracker.auto = not locked`.
- `VoiceProfiles.auto == False` currently disables unknown-speaker auto-enrollment.
- But when `VoiceProfiles.classify()` cannot confidently match an active profile, it can still fall back to the STT label key like `#1`, which displays as `参加者A`.
- `SessionState.constrain_human_speaker_key()` only enforces the manually configured `diarization_max_speakers` limit. It does not know that the active roster size should become the closed participant set.
- `diarization_max_speakers` is a separate setting, currently an anonymous display slot cap. It is not automatically synchronized with the active voice-profile roster.

## Desired Semantics

### 1. Open roster

When roster is unlocked:

- Existing behavior should remain as much as possible.
- Unknown voices may become anonymous participants and may be auto-registered as `人物N`.
- `diarization_max_speakers` continues to cap anonymous human slots.

### 2. Closed roster

When roster is locked (`tracker.auto == False`):

- Active named profiles are the roster.
- High-confidence voiceprint match -> that registered name.
- Low-confidence / no match / short ambiguous turn / overlapped speech -> `UNSURE_SPEAKER` (`?`, displayed as `未確定`), unless there is already a clearly justified same-speaker continuity rule.
- Do not create or display new anonymous people (`#N`, `@diar:N`, `人物N`, `参加者A`) after roster lock.
- Do not use `diarization_max_speakers` as the source of truth for this closed roster. The source of truth is `tracker.active_profile_names()`.

### 3. Participant count consistency

The "参加人数" setting and closed roster should be made coherent in the UI/logic:

- If the roster is locked and active profile count is 3, the effective human roster is 3 people.
- The UI should not make users think they also need to set "参加人数 = 3" for the roster to work.
- Preferred behavior: closed roster takes precedence over participant-count anonymous slot logic.
- If `diarization_max_speakers` is also set, it should not allow new anonymous participants beyond the closed roster.
- Consider exposing a snapshot field such as `vp.closed_roster_size` or `diarization.effective_max_speakers` only if it helps clarify UI state. Keep this minimal.

## Implementation Guidance

Start by adding failing tests that encode the desired behavior.

Suggested test cases:

1. `VoiceProfiles` closed roster, long unknown voice:
   - Active profiles: `A`, `B`, `C`.
   - `auto = False`.
   - Incoming long voice that does not match any active profile.
   - Expected: `UNSURE_SPEAKER`, not `#1` and not a new `人物N`.

2. `VoiceProfiles` closed roster, short ambiguous voice:
   - Active profiles: `A`, `B`.
   - `auto = False`.
   - No reliable match.
   - Expected: `UNSURE_SPEAKER` unless there is a very explicit same-speaker continuity case already tested and justified.
   - Revisit existing `test_short_turn_closed_roster_no_unsure`; it likely reflects the old behavior and may need to change.

3. `RecvLoop` / `SessionState` integration:
   - If tracker returns a fallback anonymous key under closed roster, final recorded speaker should still be `UNSURE_SPEAKER`.
   - This can be implemented either inside `VoiceProfiles.classify()` or as a final guard in `SessionState.constrain_human_speaker_key()`.

4. Participant count consistency:
   - With 3 active profiles and roster locked, new anonymous keys should not appear even if `diarization_max_speakers` is unset or greater than 3.
   - With roster unlocked, existing participant count behavior should remain.

## Likely Design Choice

The cleanest approach is probably:

- Make `VoiceProfiles.classify()` return `UNSURE_SPEAKER` whenever `auto == False` and no active profile match is confident enough.
- Keep confident registered matches unchanged.
- Keep AI echo profiles unchanged.
- Avoid introducing new "closed roster speaker" state outside `VoiceProfiles` unless needed.
- Add a defensive final guard in `SessionState.constrain_human_speaker_key()` only if tests show fallback paths outside `VoiceProfiles` can still create anonymous participants while roster is locked.

Be careful with continuity:

- In open roster, continuity/fallback to previous anonymous labels is useful.
- In closed roster, continuity to a registered speaker should only happen when the current voice still has enough evidence for that registered speaker.
- Do not let a new unknown speaker inherit the previous registered name just because STT reused the same raw label.

## Non-Goals

Do not do these in this task:

- Replace the voiceprint model.
- Change ReDimNet / ECAPA / Resemblyzer thresholds globally.
- Redesign the full speaker diarization pipeline.
- Remove the participant count setting.
- Change intervention/facilitation logic.
- Add LLM-based speaker attribution.

## Validation

Run at least:

```bash
uv run pytest tests/unit/live/test_voice_shortturn.py tests/unit/live/test_voice_unconfirmed.py tests/unit/live/test_ui_api.py -q
uv run pytest tests/unit/live/test_diarization.py tests/unit/live/test_asr_audio_buffer.py -q
uv run ruff check src/das/asr/live tests/unit/live
```

If the full live suite is feasible, run:

```bash
uv run pytest tests/unit/live -q
```

## Expected User-Facing Result

After the change:

- Register three people.
- Turn on "名簿を確定".
- Start the meeting.
- The transcript should show registered names when confident.
- Unrecognized, ambiguous, overlapped, or fourth-person speech should show as "未確定".
- It should not create "参加者A" / "人物N" while the roster is locked.

