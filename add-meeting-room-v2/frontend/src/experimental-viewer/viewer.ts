import {
  parseViewerEvent,
  type Participant,
  type SpeakerMapEntry,
  type SpeakerStatsEntry,
  type TranscriptEvent,
  type ViewerEvent,
} from "./events";

type ConnectionStatus = "connecting" | "connected" | "closed" | "error";
type ActiveTab = "setup" | "live";
type IntroStatus = "listening" | "candidate" | "failed";

type IntroState = {
  status: IntroStatus;
  candidates: string[];
  expiresAtMs?: number;
};

type ViewerState = {
  status: ConnectionStatus;
  activeTab: ActiveTab;
  meetingId: string | null;
  model: string | null;
  error: string | null;
  segments: Map<string, TranscriptEvent>;
  order: string[];
  participants: Participant[];
  speakerMap: Record<string, SpeakerMapEntry>;
  speakerStats: SpeakerStatsEntry[];
  intros: Record<string, IntroState>;
  unassignedSpeakers: Set<string>;
};

const viewerWsUrl =
  import.meta.env.VITE_VIEWER_WS_URL ?? "ws://localhost:8000/ws/viewer";
const renderedSetupHtml = new WeakMap<HTMLElement, string>();

export function mountExperimentalViewer(root: HTMLElement): void {
  const state: ViewerState = {
    status: "connecting",
    activeTab: "setup",
    meetingId: null,
    model: null,
    error: null,
    segments: new Map(),
    order: [],
    participants: [],
    speakerMap: {},
    speakerStats: [],
    intros: {},
    unassignedSpeakers: new Set(),
  };

  root.innerHTML = layout();
  const socket = new WebSocket(viewerWsUrl);

  attachControls(root, state, (payload) => {
    if (socket.readyState !== WebSocket.OPEN) {
      state.error = "viewer websocket is not connected";
      render(state);
      return;
    }
    socket.send(
      JSON.stringify({
        meeting_id: state.meetingId ?? "meeting_001",
        ...payload,
      }),
    );
  });
  render(state);

  socket.addEventListener("open", () => {
    state.status = "connected";
    render(state);
  });

  socket.addEventListener("close", () => {
    state.status = "closed";
    render(state);
  });

  socket.addEventListener("error", () => {
    state.status = "error";
    state.error = "viewer websocket connection failed";
    render(state);
  });

  socket.addEventListener("message", (message) => {
    const event = parseMessage(message.data);
    if (event === null) {
      return;
    }
    applyEvent(state, event);
    render(state);
  });

  window.setInterval(() => {
    if (Object.values(state.intros).some(isExpiredListeningIntro)) {
      render(state);
    }
  }, 1000);
}

function parseMessage(data: unknown): ViewerEvent | null {
  if (typeof data !== "string") {
    return null;
  }
  try {
    return parseViewerEvent(JSON.parse(data));
  } catch {
    return null;
  }
}

function applyEvent(state: ViewerState, event: ViewerEvent): void {
  if ("meeting_id" in event) {
    state.meetingId = event.meeting_id;
  }

  if (event.type === "session.started") {
    state.model = event.soniox_model;
    state.error = null;
    return;
  }

  if (event.type === "transcription.error") {
    state.error = event.message;
    return;
  }

  if (event.type === "transcript.delta" || event.type === "transcript.final") {
    if (!state.segments.has(event.segment_id)) {
      state.order.push(event.segment_id);
    }
    state.segments.set(event.segment_id, event);
    return;
  }

  if (event.type === "participant.list.updated") {
    state.participants = event.participants;
    state.intros = {};
    return;
  }

  if (event.type === "speaker.map.updated") {
    state.speakerMap = event.speaker_map;
    return;
  }

  if (event.type === "speaker.unassigned_detected") {
    state.unassignedSpeakers.add(event.speaker_label);
    return;
  }

  if (event.type === "speaker.intro.started") {
    state.intros[event.participant_id] = {
      status: "listening",
      candidates: [],
      expiresAtMs: event.expires_at_ms,
    };
    return;
  }

  if (event.type === "speaker.intro.candidate_detected") {
    state.intros[event.participant_id] = {
      status: "candidate",
      candidates: event.candidates,
    };
    return;
  }

  if (event.type === "speaker.intro.completed") {
    delete state.intros[event.participant_id];
    return;
  }

  if (event.type === "speaker.intro.expired") {
    state.intros[event.participant_id] = {
      status: event.candidates.length > 0 ? "candidate" : "failed",
      candidates: event.candidates,
    };
    return;
  }

  if (event.type === "speaker.intro.cancelled") {
    delete state.intros[event.participant_id];
    return;
  }

  if (event.type === "speaker.stats.updated") {
    state.speakerStats = event.stats;
  }
}

function render(state: ViewerState): void {
  setText("status-value", statusLabel(state.status));
  setText("meeting-value", state.meetingId ?? "waiting");
  setText("model-value", state.model ?? "waiting");
  setText("error-value", state.error ?? "");

  document
    .querySelectorAll<HTMLButtonElement>(".tab")
    .forEach((tab) =>
      tab.classList.toggle("active", tab.dataset.tab === state.activeTab),
    );

  const setupPanel = document.querySelector<HTMLElement>("#setup-panel");
  const livePanel = document.querySelector<HTMLElement>("#live-panel");
  if (setupPanel !== null && livePanel !== null) {
    setupPanel.hidden = state.activeTab !== "setup";
    livePanel.hidden = state.activeTab !== "live";
  }

  renderSetup(state);
  renderStats(state);
  renderTranscript(state);
}

function renderSetup(state: ViewerState): void {
  const participants = document.querySelector<HTMLDivElement>("#participants");
  const speakerMap = document.querySelector<HTMLDivElement>("#speaker-map");
  const unassigned = document.querySelector<HTMLDivElement>("#unassigned-speakers");
  if (participants === null || speakerMap === null || unassigned === null) {
    return;
  }

  updateSetupHtml(
    participants,
    state.participants.length === 0
      ? `<p class="empty compact">Save participants to begin.</p>`
      : state.participants
          .map((participant) => renderParticipant(state, participant))
          .join(""),
  );

  const mapEntries = Object.entries(state.speakerMap);
  updateSetupHtml(
    speakerMap,
    mapEntries.length === 0
      ? `<p class="empty compact">No speaker mappings yet.</p>`
      : mapEntries
          .map(
            ([speakerLabel, assignment]) => `
              <div class="map-row">
                <b>Speaker ${escapeHtml(speakerLabel)}</b>
                <span>${escapeHtml(assignment.display_name)} · ${escapeHtml(
                  assignment.source,
                )}</span>
              </div>
            `,
          )
          .join(""),
  );

  const unmapped = [...state.unassignedSpeakers].filter(
    (speakerLabel) => state.speakerMap[speakerLabel] === undefined,
  );
  const unassignedParticipants = state.participants.filter(
    (participant) =>
      speakerForParticipant(state, participant.participant_id) === null,
  );
  updateSetupHtml(
    unassigned,
    unmapped.length === 0
      ? `<p class="empty compact">No unassigned speakers.</p>`
      : unmapped
          .map(
            (speakerLabel) => `
              <div class="map-row unassigned">
                <div>
                  <b>Speaker ${escapeHtml(speakerLabel)}</b>
                  <span>Unassigned</span>
                </div>
                ${renderManualAssignment(speakerLabel, unassignedParticipants)}
              </div>
            `,
          )
          .join(""),
  );
}

function updateSetupHtml(element: HTMLElement, html: string): void {
  if (renderedSetupHtml.get(element) === html) {
    return;
  }
  element.innerHTML = html;
  renderedSetupHtml.set(element, html);
}

function renderManualAssignment(
  speakerLabel: string,
  participants: Participant[],
): string {
  if (participants.length === 0) {
    return `<span>No unassigned participants</span>`;
  }

  return `
    <div class="manual-assignment">
      <select
        data-manual-participant
        aria-label="Participant for Speaker ${escapeHtml(speakerLabel)}"
      >
        <option value="">Select participant</option>
        ${participants
          .map(
            (participant) => `
              <option value="${escapeHtml(participant.participant_id)}">
                ${escapeHtml(participant.display_name)}
              </option>
            `,
          )
          .join("")}
      </select>
      <button
        type="button"
        data-bind-speaker="${escapeHtml(speakerLabel)}"
        disabled
      >Assign</button>
    </div>
  `;
}

function renderParticipant(state: ViewerState, participant: Participant): string {
  const boundSpeaker = speakerForParticipant(state, participant.participant_id);
  if (boundSpeaker !== null) {
    return `
      <div class="participant-row">
        <div>
          <b>${escapeHtml(participant.display_name)}</b>
          <span>Speaker ${escapeHtml(boundSpeaker)}</span>
        </div>
        <button type="button" data-intro="${escapeHtml(
          participant.participant_id,
        )}">Re-intro</button>
      </div>
    `;
  }

  const intro = state.intros[participant.participant_id];
  if (intro?.status === "listening" && isExpiredListeningIntro(intro)) {
    return `
      <div class="participant-row failed">
        <div>
          <b>${escapeHtml(participant.display_name)}</b>
          <span>No new speaker detected</span>
        </div>
        <button type="button" data-intro="${escapeHtml(
          participant.participant_id,
        )}">Retry</button>
      </div>
    `;
  }

  if (intro?.status === "listening") {
    return `
      <div class="participant-row active-intro">
        <div>
          <b>${escapeHtml(participant.display_name)}</b>
          <span>Listening for a new speaker...</span>
        </div>
        <button type="button" data-cancel-intro="${escapeHtml(
          participant.participant_id,
        )}">Cancel</button>
      </div>
    `;
  }

  if (intro?.status === "candidate") {
    const candidate = intro.candidates[0] ?? "";
    const extra =
      intro.candidates.length > 1
        ? ` +${intro.candidates.length - 1} more`
        : "";
    return `
      <div class="participant-row candidate">
        <div>
          <b>${escapeHtml(participant.display_name)}</b>
          <span>Candidate: Speaker ${escapeHtml(candidate)}${escapeHtml(extra)}</span>
        </div>
        <div class="row-actions">
          <button type="button" data-confirm-intro="${escapeHtml(
            participant.participant_id,
          )}" data-speaker-label="${escapeHtml(candidate)}">Confirm</button>
          <button type="button" data-intro="${escapeHtml(
            participant.participant_id,
          )}">Retry</button>
        </div>
      </div>
    `;
  }

  if (intro?.status === "failed") {
    return `
      <div class="participant-row failed">
        <div>
          <b>${escapeHtml(participant.display_name)}</b>
          <span>No new speaker detected</span>
        </div>
        <button type="button" data-intro="${escapeHtml(
          participant.participant_id,
        )}">Retry</button>
      </div>
    `;
  }

  return `
    <div class="participant-row">
      <div>
        <b>${escapeHtml(participant.display_name)}</b>
        <span>${participant.role}</span>
      </div>
      <button type="button" data-intro="${escapeHtml(
        participant.participant_id,
      )}">Start Intro</button>
    </div>
  `;
}

function isExpiredListeningIntro(intro: IntroState): boolean {
  return (
    intro.status === "listening" &&
    intro.expiresAtMs !== undefined &&
    Date.now() >= intro.expiresAtMs
  );
}

function renderTranscript(state: ViewerState): void {
  const transcript = document.querySelector<HTMLDivElement>("#transcript");
  if (transcript === null) {
    return;
  }

  const segments = state.order
    .map((id) => state.segments.get(id))
    .filter((event): event is TranscriptEvent => event !== undefined);

  if (segments.length === 0) {
    transcript.innerHTML = `<p class="empty">Waiting for speech...</p>`;
    return;
  }

  transcript.innerHTML = segments.map(renderSegment).join("");
  transcript.scrollTop = transcript.scrollHeight;
}

function renderStats(state: ViewerState): void {
  const stats = document.querySelector<HTMLDivElement>("#speaker-stats");
  if (stats === null) {
    return;
  }

  if (state.speakerStats.length === 0) {
    stats.innerHTML = `<p class="empty compact">No speaking stats yet.</p>`;
    return;
  }

  stats.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Speaker</th>
          <th>Turns</th>
          <th>Chars</th>
          <th>Time</th>
        </tr>
      </thead>
      <tbody>
        ${state.speakerStats.map(renderStatsRow).join("")}
      </tbody>
    </table>
  `;
}

function renderStatsRow(entry: SpeakerStatsEntry): string {
  const speaker =
    entry.display_name ??
    (entry.speaker_label === null ? "Unknown" : `Speaker ${entry.speaker_label}`);
  return `
    <tr>
      <td>${escapeHtml(speaker)}</td>
      <td>${entry.utterance_count}</td>
      <td>${entry.text_chars}</td>
      <td>${formatDuration(entry.estimated_speech_ms)}</td>
    </tr>
  `;
}

function renderSegment(event: TranscriptEvent): string {
  const fallbackSpeaker =
    event.speaker_label === null ? "Speaker ?" : `Speaker ${event.speaker_label}`;
  const speaker = event.display_name ?? fallbackSpeaker;
  const finalClass = event.is_final ? "final" : "delta";
  const status = event.is_final ? "final" : "live";
  const text = stripEndpointToken(event.text);
  const rawSpeaker =
    event.display_name && event.speaker_label !== null
      ? `<small>Speaker ${escapeHtml(event.speaker_label)}</small>`
      : "";
  return `
    <section class="segment ${finalClass}">
      <div class="speaker">${escapeHtml(speaker)}${rawSpeaker}</div>
      <p>${escapeHtml(text)}</p>
      <span>${status}</span>
    </section>
  `;
}

function attachControls(
  root: HTMLElement,
  state: ViewerState,
  sendCommand: (payload: Record<string, unknown>) => void,
): void {
  root.addEventListener("change", (event) => {
    const target = event.target;
    if (
      !(target instanceof HTMLSelectElement) ||
      target.dataset.manualParticipant === undefined
    ) {
      return;
    }
    const assignButton =
      target.parentElement?.querySelector<HTMLButtonElement>(
        "[data-bind-speaker]",
      );
    if (assignButton !== null && assignButton !== undefined) {
      assignButton.disabled = target.value === "";
    }
  });

  root.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }

    const tab = target.dataset.tab;
    if (tab === "setup" || tab === "live") {
      state.activeTab = tab;
      render(state);
      return;
    }

    if (target.id === "save-participants") {
      sendCommand(participantUpdatePayload());
      return;
    }

    const introParticipantId = target.dataset.intro;
    if (introParticipantId) {
      sendCommand({
        type: "speaker.intro.start",
        participant_id: introParticipantId,
      });
      return;
    }

    const cancelParticipantId = target.dataset.cancelIntro;
    if (cancelParticipantId) {
      sendCommand({
        type: "speaker.intro.cancel",
        participant_id: cancelParticipantId,
      });
      return;
    }

    const confirmParticipantId = target.dataset.confirmIntro;
    const speakerLabel = target.dataset.speakerLabel;
    if (confirmParticipantId && speakerLabel) {
      sendCommand({
        type: "speaker.bind",
        speaker_label: speakerLabel,
        participant_id: confirmParticipantId,
      });
      return;
    }

    const manualSpeakerLabel = target.dataset.bindSpeaker;
    if (manualSpeakerLabel) {
      const participantSelect =
        target.parentElement?.querySelector<HTMLSelectElement>(
          "[data-manual-participant]",
        );
      const participantId = participantSelect?.value;
      if (participantId) {
        sendCommand({
          type: "speaker.bind",
          speaker_label: manualSpeakerLabel,
          participant_id: participantId,
        });
      }
    }
  });
}

function participantUpdatePayload(): Record<string, unknown> {
  const rawNames = getInputValue("participant-names");
  const aiName = getInputValue("agent-name") || "AI agent";
  const humanParticipants = rawNames
    .split("\n")
    .map((name) => name.trim())
    .filter(Boolean)
    .map((name, index) => ({
      participant_id: `p_${String(index + 1).padStart(3, "0")}`,
      display_name: name,
      role: "human",
    }));

  return {
    type: "participant.list.update",
    participants: [
      ...humanParticipants,
      {
        participant_id: "agent",
        display_name: aiName,
        role: "agent",
      },
    ],
  };
}

function speakerForParticipant(
  state: ViewerState,
  participantId: string,
): string | null {
  for (const [speakerLabel, assignment] of Object.entries(state.speakerMap)) {
    if (assignment.participant_id === participantId) {
      return speakerLabel;
    }
  }
  return null;
}

function getInputValue(id: string): string {
  const element = document.querySelector<HTMLInputElement | HTMLTextAreaElement>(
    `#${id}`,
  );
  return element?.value.trim() ?? "";
}

function stripEndpointToken(text: string): string {
  return text.replaceAll("<end>", "").trim();
}

function formatDuration(ms: number): string {
  const totalSeconds = Math.round(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes === 0) {
    return `${seconds}s`;
  }
  return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
}

function setText(id: string, value: string): void {
  const element = document.querySelector<HTMLElement>(`#${id}`);
  if (element !== null) {
    element.textContent = value;
  }
}

function statusLabel(status: ConnectionStatus): string {
  switch (status) {
    case "connecting":
      return "connecting";
    case "connected":
      return "connected";
    case "closed":
      return "closed";
    case "error":
      return "error";
  }
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function layout(): string {
  return `
    <div class="viewer-shell">
      <header>
        <h1>Meeting Room Viewer</h1>
        <dl>
          <div><dt>Status</dt><dd id="status-value">connecting</dd></div>
          <div><dt>Meeting</dt><dd id="meeting-value">waiting</dd></div>
          <div><dt>Model</dt><dd id="model-value">waiting</dd></div>
        </dl>
      </header>
      <nav class="tabs">
        <button class="tab active" type="button" data-tab="setup">Setup</button>
        <button class="tab" type="button" data-tab="live">Live</button>
      </nav>
      <p id="error-value" class="error"></p>
      <section id="setup-panel" class="setup-panel">
        <div class="setup-grid">
          <section>
            <h2>Participants</h2>
            <textarea id="participant-names" rows="4">田中
佐藤
鈴木</textarea>
            <label>
              AI
              <input id="agent-name" value="AI agent" />
            </label>
            <button id="save-participants" type="button">Save</button>
          </section>
          <section>
            <h2>Self Intro</h2>
            <div id="participants"></div>
          </section>
          <section>
            <h2>Speaker Map</h2>
            <div id="speaker-map"></div>
          </section>
          <section>
            <h2>Unassigned</h2>
            <div id="unassigned-speakers"></div>
          </section>
        </div>
      </section>
      <section id="live-panel" class="live-panel" hidden>
        <section class="stats-panel">
          <h2>Speaking Stats</h2>
          <div id="speaker-stats"></div>
        </section>
        <div id="transcript" class="transcript" aria-live="polite"></div>
      </section>
    </div>
  `;
}
