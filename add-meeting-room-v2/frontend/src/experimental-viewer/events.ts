export type SessionStartedEvent = {
  type: "session.started";
  meeting_id: string;
  soniox_model: string;
  sample_rate: number;
  frame_ms: number;
  server_timestamp_ms: number;
};

export type SessionEndedEvent = {
  type: "session.ended";
  meeting_id: string;
  reason: string;
  server_timestamp_ms: number;
};

export type TranscriptionErrorEvent = {
  type: "transcription.error";
  meeting_id: string;
  message: string;
  server_timestamp_ms: number;
};

export type TranscriptEvent = {
  type: "transcript.delta" | "transcript.final";
  meeting_id: string;
  segment_id: string;
  speaker_label: string | null;
  participant_id?: string | null;
  display_name?: string | null;
  speaker_status?: "mapped" | "unassigned";
  text: string;
  is_final: boolean;
  endpoint_detected?: boolean;
  start_ms?: number | null;
  end_ms?: number | null;
  server_timestamp_ms: number;
};

export type Participant = {
  participant_id: string;
  display_name: string;
  role: "human" | "agent";
};

export type ParticipantListUpdatedEvent = {
  type: "participant.list.updated";
  meeting_id: string;
  participants: Participant[];
  server_timestamp_ms: number;
};

export type SpeakerMapEntry = {
  participant_id: string;
  display_name: string;
  role: "human" | "agent";
  source: string;
};

export type SpeakerMapUpdatedEvent = {
  type: "speaker.map.updated";
  meeting_id: string;
  speaker_map: Record<string, SpeakerMapEntry>;
  server_timestamp_ms: number;
};

export type SpeakerUnassignedDetectedEvent = {
  type: "speaker.unassigned_detected";
  meeting_id: string;
  speaker_label: string;
  server_timestamp_ms: number;
};

export type SpeakerIntroStartedEvent = {
  type: "speaker.intro.started";
  meeting_id: string;
  participant_id: string;
  display_name: string;
  role: "human" | "agent";
  known_speaker_labels: string[];
  expires_at_ms: number;
  server_timestamp_ms: number;
};

export type SpeakerIntroCandidateDetectedEvent = {
  type: "speaker.intro.candidate_detected";
  meeting_id: string;
  participant_id: string;
  display_name: string | null;
  speaker_label: string;
  candidates: string[];
  server_timestamp_ms: number;
};

export type SpeakerIntroCompletedEvent = {
  type: "speaker.intro.completed";
  meeting_id: string;
  participant_id: string;
  speaker_label: string;
  server_timestamp_ms: number;
};

export type SpeakerIntroExpiredEvent = {
  type: "speaker.intro.expired";
  meeting_id: string;
  participant_id: string;
  candidates: string[];
  server_timestamp_ms: number;
};

export type SpeakerIntroCancelledEvent = {
  type: "speaker.intro.cancelled";
  meeting_id: string;
  participant_id: string;
  server_timestamp_ms: number;
};

export type SpeakerStatsEntry = {
  speaker_key: string;
  participant_id: string | null;
  display_name: string | null;
  speaker_label: string | null;
  utterance_count: number;
  text_chars: number;
  estimated_speech_ms: number;
  last_spoke_at_ms: number | null;
};

export type SpeakerStatsUpdatedEvent = {
  type: "speaker.stats.updated";
  meeting_id: string;
  stats: SpeakerStatsEntry[];
  server_timestamp_ms: number;
};

export type ViewerEvent =
  | SessionStartedEvent
  | SessionEndedEvent
  | TranscriptionErrorEvent
  | TranscriptEvent
  | ParticipantListUpdatedEvent
  | SpeakerMapUpdatedEvent
  | SpeakerUnassignedDetectedEvent
  | SpeakerIntroStartedEvent
  | SpeakerIntroCandidateDetectedEvent
  | SpeakerIntroCompletedEvent
  | SpeakerIntroExpiredEvent
  | SpeakerIntroCancelledEvent
  | SpeakerStatsUpdatedEvent;

export function parseViewerEvent(value: unknown): ViewerEvent | null {
  if (!isRecord(value) || typeof value.type !== "string") {
    return null;
  }

  switch (value.type) {
    case "session.started":
    case "session.ended":
    case "transcription.error":
    case "transcript.delta":
    case "transcript.final":
    case "participant.list.updated":
    case "speaker.map.updated":
    case "speaker.unassigned_detected":
    case "speaker.intro.started":
    case "speaker.intro.candidate_detected":
    case "speaker.intro.completed":
    case "speaker.intro.expired":
    case "speaker.intro.cancelled":
    case "speaker.stats.updated":
      return value as ViewerEvent;
    default:
      return null;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
