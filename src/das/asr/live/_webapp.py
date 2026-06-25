"""ライブセッションのフロントエンド（単一ファイルSPA, F4）.

バックエンドAPI（/api/state, /api/stream(SSE), /api/stop, /api/mode）を使う
依存なしのWebアプリ。`_UIHandler` が GET / でこの INDEX_HTML を配信する。
"""
from __future__ import annotations

INDEX_HTML = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>議論支援</title>
<style>
  :root {
    --bg: #f6f7f9; --card: #fff; --line: #e5e7eb; --ink: #1f2937;
    --muted: #6b7280; --accent: #2563eb; --accent-soft: #eff6ff;
    --facil: #0369a1; --facil-bg: #f0f9ff; --danger: #dc2626;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
    font-family: -apple-system, "Hiragino Sans", "Segoe UI", sans-serif;
    line-height: 1.6; }
  .wrap { max-width: 1040px; margin: 0 auto; padding: 16px; }

  header { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
  header h1 { font-size: 1.05rem; margin: 0; font-weight: 700; }
  .status { display: flex; align-items: center; gap: 6px; font-size: .85rem;
    color: var(--muted); }
  .dot { width: 9px; height: 9px; border-radius: 50%; background: #16a34a; }
  .status.stopped .dot { background: #9ca3af; }
  .spacer { flex: 1; }
  .btn { border: 1px solid var(--line); background: var(--card); color: var(--ink);
    border-radius: 8px; padding: .4em .8em; font-size: .85rem; cursor: pointer; }
  .btn:hover { background: #f3f4f6; }
  .btn-stop { border-color: #fecaca; color: var(--danger); }
  .btn-stop:hover { background: #fef2f2; }
  .btn:disabled { opacity: .5; cursor: default; }

  /* モードセレクタ */
  .modes { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;
    margin-bottom: 14px; }
  .mode { border: 1px solid var(--line); background: var(--card); border-radius: 10px;
    padding: 10px 12px; cursor: pointer; transition: all .15s; text-align: left; }
  .mode:hover { border-color: #93c5fd; }
  .mode.active { border-color: var(--accent); background: var(--accent-soft); }
  .mode .name { font-weight: 600; font-size: .92rem; }
  .mode .desc { font-size: .76rem; color: var(--muted); margin-top: 2px; }
  .mode.busy { opacity: .6; }

  .agenda-bar { display: flex; align-items: center; gap: 8px; margin-bottom: 14px;
    background: var(--card); border: 1px solid var(--line); border-radius: 10px;
    padding: 8px 12px; }
  .agenda-label { font-size: .82rem; color: var(--muted); font-weight: 600;
    flex-shrink: 0; }
  .agenda-bar input { flex: 1; min-width: 0; border: 1px solid var(--line);
    border-radius: 7px; padding: .4em .6em; font-size: .9rem; }
  .cols { display: flex; gap: 14px; align-items: flex-start; }
  .main { flex: 1; min-width: 0; }
  .side { width: 248px; flex-shrink: 0; }
  @media (max-width: 760px) {
    .cols { flex-direction: column; } .side { width: 100%; }
    .modes { grid-template-columns: 1fr; }
  }

  /* 議事録 */
  .transcript { background: var(--card); border: 1px solid var(--line);
    border-radius: 12px; padding: 10px 12px; max-height: 72vh; overflow-y: auto; }
  .u { margin: .45rem 0; }
  .u .who { font-weight: 700; margin-right: .5em; }
  .u .ts { color: #9ca3af; font-size: .78rem; margin-right: .5em;
    font-variant-numeric: tabular-nums; }
  .u.facil { background: var(--facil-bg); border: 1px solid #bae6fd;
    border-radius: 10px; padding: .5em .7em; }
  .u.facil .who { color: var(--facil); }
  .u .badge { background: #fef3c7; color: #92400e; font-size: .68rem;
    border-radius: 5px; padding: .05em .4em; margin-left: .4em; }
  .vp-banner { margin: 0 0 .6rem; padding: .5rem .75rem; border-radius: 6px;
    font-size: .85rem; background: #fdecea; color: #8a1c12; border: 1px solid #f3b4ab; }
  .vp-banner.ok { background: #eef7ee; color: #2c6e2c; border-color: #bfe0bf; }
  .enroll-hint { font-size: .78rem; color: var(--muted); margin: 0 0 .5rem; }
  .roster { display: flex; flex-wrap: wrap; gap: .3rem; margin-bottom: .5rem; }
  .roster .chip { background: #eef2ff; color: #3730a3; border-radius: 999px;
    padding: .12rem .55rem; font-size: .8rem; }
  .enroll-row { display: flex; gap: .35rem; }
  .enroll-row input { flex: 1; min-width: 0; border: 1px solid var(--line);
    border-radius: 6px; padding: .35rem .5rem; }
  .enroll-status { font-size: .8rem; color: var(--muted); margin-top: .4rem; min-height: 1.1em; }
  .enroll-status.rec { color: #b91c1c; font-weight: 600; }
  .roster-lock { display: flex; align-items: center; gap: .4rem; font-size: .8rem;
    margin-top: .6rem; cursor: pointer; }
  .setting-row { display: flex; align-items: center; gap: .45rem; }
  .setting-row label { flex: 1; font-size: .8rem; color: var(--muted); }
  .setting-row select { border: 1px solid var(--line); border-radius: 6px;
    padding: .3rem .45rem; background: var(--card); }
  .setting-note { font-size: .72rem; color: var(--muted); margin-top: .35rem; }
  .u.bc { opacity: .5; font-size: .85em; }            /* 相槌は薄く小さく */
  .u.unsure { opacity: .75; }
  .u.unsure .who { font-style: italic; }              /* 未確定は話者名をイタリックに */
  .u.partial { opacity: .55; font-style: italic; }
  .u.partial::after { content: "…"; }
  .sys { text-align: center; color: var(--muted); font-size: .78rem; margin: .4rem 0; }
  .empty { color: var(--muted); font-size: .9rem; padding: 1rem; text-align: center; }

  /* サイドパネル */
  .panel { background: var(--card); border: 1px solid var(--line);
    border-radius: 12px; padding: 10px 12px; margin-bottom: 12px; }
  .panel h2 { font-size: .8rem; color: #9ca3af; font-weight: 500;
    margin: 0 0 .5rem; }
  .bar-row { display: flex; align-items: center; gap: 6px; font-size: .8rem;
    margin-bottom: .35em; }
  .bar-name { width: 4.5em; flex-shrink: 0; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap; }
  .bar-bg { flex: 1; height: 9px; background: #eef0f2; border-radius: 5px;
    overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 5px; background: var(--accent); }
  .bar-pct { width: 2.6em; text-align: right; color: var(--muted);
    font-size: .74rem; font-variant-numeric: tabular-nums; }
  .stat-group { margin-bottom: .7em; }
  .stat-label { font-size: .72rem; color: #9ca3af; margin-bottom: .25em; }
  .topic { font-size: .8rem; padding: .35em .5em; margin-bottom: .3em;
    background: #fff; border: 1px solid var(--line); border-left: 3px solid #8b5cf6;
    border-radius: 6px; }
  .topic .by { font-size: .68rem; color: #9ca3af; }
  .spk-row { display: flex; align-items: center; gap: 5px; margin-bottom: .4em; }
  .spk-name { width: 4em; flex-shrink: 0; font-weight: 600; font-size: .82rem;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .spk-input { flex: 1; min-width: 0; font-size: .8rem; border: 1px solid #d1d5db;
    border-radius: 5px; padding: .2em .4em; }
  .spk-btn { font-size: .74rem; background: var(--accent); color: #fff; border: none;
    border-radius: 5px; padding: .25em .55em; cursor: pointer; white-space: nowrap; }
  .spk-btn:hover { background: #1d4ed8; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>議論支援</h1>
    <span class="status" id="status"><span class="dot"></span><span id="status-text">接続中…</span></span>
    <span class="spacer"></span>
    <button class="btn" id="reset">新しい会議</button>
    <button class="btn btn-stop" id="end">終了</button>
  </header>

  <div class="modes" id="modes"></div>

  <div class="vp-banner" id="vp-banner" hidden></div>

  <div class="agenda-bar">
    <span class="agenda-label">議題</span>
    <input id="agenda" placeholder="この会議のテーマ（脱線判定の基準。空欄なら自動推定）">
    <button class="btn" id="agenda-set">設定</button>
  </div>

  <div class="cols">
    <div class="main">
      <div class="transcript" id="transcript"><div class="empty">まだ発話がありません</div></div>
    </div>
    <div class="side">
      <div class="panel" id="enroll-panel" hidden>
        <h2>声の事前登録</h2>
        <p class="enroll-hint">会議の前に、各人が順に20秒ほど一人で話して登録すると精度が上がります。</p>
        <div id="roster" class="roster"></div>
        <div class="enroll-row">
          <input id="enroll-name" placeholder="名前">
          <button class="btn" id="enroll-btn">20秒録音して登録</button>
        </div>
        <div id="enroll-status" class="enroll-status"></div>
        <label class="roster-lock">
          <input type="checkbox" id="roster-lock"> 名簿を確定（登録済みの人だけで進める）
        </label>
      </div>
      <div class="panel" id="diar-panel">
        <h2>話者分離</h2>
        <div class="setting-row">
          <label for="speaker-count">想定話者数</label>
          <select id="speaker-count"></select>
        </div>
        <div class="setting-note">次の会議から反映</div>
      </div>
      <div class="panel" id="spk-panel" hidden>
        <h2>話者の名前を登録</h2><div id="speakers"></div>
      </div>
      <div class="panel" id="part-panel" hidden>
        <h2>発言量</h2><div id="participation"></div>
      </div>
      <div class="panel" id="topic-panel" hidden>
        <h2>論点</h2><div id="topics"></div>
      </div>
    </div>
  </div>
</div>

<script>
"use strict";

// 3モードの定義（UI表示用）
const MODES = [
  { id: "transcribe", name: "議事録のみ", desc: "文字起こし＋話者分離だけ" },
  { id: "converse",   name: "AIと会話",   desc: "AIと話しつつ進行も手伝う" },
  { id: "facilitate", name: "人間に介入", desc: "人同士の議論を進行役が支援" },
];
const $ = (id) => document.getElementById(id);
let busyMode = null;   // 切替中のモードid
let stopped = false;

function fmtTs(ms) {
  if (ms == null) return "--:--";
  const s = Math.floor(ms / 1000);
  return String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
}

function renderModes(active) {
  const el = $("modes");
  el.innerHTML = "";
  for (const m of MODES) {
    const div = document.createElement("button");
    div.className = "mode" + (m.id === active ? " active" : "")
      + (m.id === busyMode ? " busy" : "");
    div.innerHTML = `<div class="name">${m.name}</div><div class="desc">${m.desc}</div>`;
    div.onclick = () => setMode(m.id);
    el.appendChild(div);
  }
}

function renderTranscript(records, partial) {
  const box = $("transcript");
  // 末尾付近にいるなら自動スクロールを維持
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
  const parts = [];
  for (const r of records) {
    if (r.type === "sys") { parts.push(`<div class="sys">⚙ ${esc(r.text)}</div>`); continue; }
    const facil = r.speaker === "ファシリテーター";
    const color = facil ? "var(--facil)" : (r.unsure ? "var(--muted)" : (r.color || "#444"));
    const badge = r.corrected ? '<span class="badge">声紋補正</span>' : "";
    const cls = "u" + (facil ? " facil" : "") + (r.bc ? " bc" : "") + (r.unsure ? " unsure" : "");
    parts.push(`<div class="${cls}">`
      + `<span class="ts">${fmtTs(r.ms)}</span>`
      + `<span class="who" style="color:${color}">${esc(r.speaker)}</span>`
      + `${esc(r.text)}${badge}</div>`);
  }
  // 認識途中（partial）を薄字で末尾に表示（課題①）
  if (partial && partial.text) {
    parts.push(`<div class="u partial">`
      + `<span class="who" style="color:var(--muted)">${esc(partial.speaker)}</span>`
      + `${esc(partial.text)}</div>`);
  }
  box.innerHTML = parts.length ? parts.join("")
    : '<div class="empty">まだ発話がありません</div>';
  if (atBottom) box.scrollTop = box.scrollHeight;
}

function bars(list, key) {
  return list.map((p) => {
    const pct = Math.round((p[key] || 0) * 100);
    return `<div class="bar-row"><span class="bar-name" title="${esc(p.speaker)}">${esc(p.speaker)}</span>`
      + `<span class="bar-bg"><span class="bar-fill" style="width:${pct}%;background:${p.color || "#888"}"></span></span>`
      + `<span class="bar-pct">${pct}%</span></div>`;
  }).join("");
}

function renderParticipation(list) {
  const panel = $("part-panel");
  if (!list || list.length < 2) { panel.hidden = true; return; }
  panel.hidden = false;
  let html = "";
  if (list.some((p) => p.has_time)) {
    html += `<div class="stat-group"><div class="stat-label">発話時間</div>${bars(list, "time_share")}</div>`;
  }
  html += `<div class="stat-group"><div class="stat-label">文字数</div>${bars(list, "char_share")}</div>`;
  html += `<div class="stat-group"><div class="stat-label">発話回数</div>${bars(list, "turn_share")}</div>`;
  $("participation").innerHTML = html;
}

function renderSpeakers(list) {
  const panel = $("spk-panel");
  const items = (list || []).filter((s) => s.renameable);
  if (!items.length) { panel.hidden = true; return; }
  panel.hidden = false;
  // 編集中（入力にフォーカス）なら作り直さない（課題②: ライブ更新で入力が消えるのを防ぐ）
  if ($("speakers").contains(document.activeElement)) return;
  $("speakers").innerHTML = items.map((s) =>
    `<div class="spk-row">`
    + `<span class="spk-name" style="color:${s.color || "#444"}" title="${esc(s.name)}">${esc(s.name)}</span>`
    + `<input class="spk-input" data-label="${esc(s.label)}" placeholder="名前">`
    + `<button class="spk-btn">登録</button></div>`
  ).join("");
  for (const btn of $("speakers").querySelectorAll(".spk-btn")) {
    btn.onclick = () => rename(btn.previousElementSibling);
  }
}

async function rename(input) {
  const label = input.dataset.label, name = input.value.trim();
  if (!name) return;
  try {
    await fetch("/rename", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label, name }),
    });
    input.value = "";  // 反映はSSEで届く
  } catch (e) { alert("名前の登録に失敗しました"); }
}

function renderTopics(list) {
  const panel = $("topic-panel");
  if (!list || !list.length) { panel.hidden = true; return; }
  panel.hidden = false;
  $("topics").innerHTML = list.map((t) =>
    `<div class="topic">${esc(t.topic)}<div class="by">${esc(t.speaker || "")}</div></div>`
  ).join("");
}

function renderAgenda(agenda) {
  const inp = $("agenda");
  if (document.activeElement !== inp) inp.value = agenda || "";  // 編集中は上書きしない
}

async function setAgenda() {
  const topic = $("agenda").value.trim();
  try {
    await fetch("/api/topic", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic }),
    });
    $("agenda").blur();
  } catch (e) { alert("議題の設定に失敗しました"); }
}

function renderVpBanner(vp) {
  const el = $("vp-banner");
  if (!vp) { el.hidden = true; return; }
  el.hidden = false;
  if (vp.enabled) {
    el.className = "vp-banner ok";
    el.textContent = `声紋オン（${vp.model || "?"}）：話者の取り違えを補正します`;
  } else {
    el.className = "vp-banner";
    el.textContent = "⚠ 声紋オフ：話者の取り違えは補正されません。"
      + "依存を導入して再起動してください（uv sync --extra soniox）";
  }
}

function renderEnroll(vp) {
  const panel = $("enroll-panel");
  if (!vp || !vp.enabled) { panel.hidden = true; return; }
  panel.hidden = false;
  const roster = vp.roster || [];
  $("roster").innerHTML = roster.length
    ? roster.map((n) => `<span class="chip">${esc(n)}</span>`).join("")
    : '<span class="enroll-hint">まだ登録された人はいません</span>';
  const lock = $("roster-lock");
  if (document.activeElement !== lock) lock.checked = !!vp.locked;
}

function renderDiarization(config) {
  const sel = $("speaker-count");
  if (!sel.options.length) {
    sel.innerHTML = '<option value="">未指定</option>'
      + Array.from({ length: 10 }, (_, i) => `<option value="${i + 1}">${i + 1}人</option>`).join("");
  }
  if (document.activeElement !== sel) sel.value = config && config.max_speakers ? String(config.max_speakers) : "";
}

let enrolling = false;
async function enrollPerson() {
  if (enrolling) return;
  const name = $("enroll-name").value.trim();
  if (!name) { $("enroll-name").focus(); return; }
  enrolling = true;
  const btn = $("enroll-btn"), st = $("enroll-status");
  btn.disabled = true;
  const secs = 20;
  for (let t = secs; t > 0; t--) {
    st.className = "enroll-status rec";
    st.textContent = `● 録音中… ${esc(name)} さんに一人で話してもらってください（残り${t}秒）`;
    await new Promise((r) => setTimeout(r, 1000));
  }
  st.className = "enroll-status";
  st.textContent = "登録中…";
  try {
    const res = await fetch("/api/enroll", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: name, seconds: secs }),
    });
    const d = await res.json();
    if (res.ok) { st.textContent = `✓ ${esc(name)} を登録しました（${d.seconds}秒）`; $("enroll-name").value = ""; }
    else { st.textContent = "登録失敗: " + (d.error || ""); }
  } catch (e) { st.textContent = "登録に失敗しました"; }
  btn.disabled = false; enrolling = false;
}

async function toggleRoster() {
  const locked = $("roster-lock").checked;
  try {
    await fetch("/api/roster", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ locked: locked }),
    });
  } catch (e) { /* SSEで状態が届く */ }
}

async function setSpeakerCount() {
  const v = $("speaker-count").value;
  try {
    await fetch("/api/diarization", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ max_speakers: v ? Number(v) : null }),
    });
  } catch (e) { alert("話者数の設定に失敗しました"); }
}

function render(state) {
  renderModes(state.mode);
  renderVpBanner(state.vp);
  renderEnroll(state.vp);
  renderDiarization(state.diarization);
  renderAgenda(state.agenda);
  renderTranscript(state.records || [], state.partial);
  renderSpeakers(state.speakers);
  renderParticipation(state.participation);
  renderTopics(state.topics);
  setStatus(state.running, state.resetting);
}

function setStatus(running, resetting) {
  const el = $("status"), txt = $("status-text");
  if (!running || stopped) {
    el.classList.add("stopped"); txt.textContent = "終了しました";
    $("end").disabled = true; $("reset").disabled = true;
  } else if (resetting) {
    el.classList.remove("stopped"); txt.textContent = "リセット中…";
    $("reset").disabled = true;
  } else {
    el.classList.remove("stopped"); txt.textContent = "ライブ";
    $("reset").disabled = false;
  }
}

async function resetMeeting() {
  if (stopped) return;
  if (!confirm("今の会議を終了して、新しい会議を始めますか？（声紋・話者名は引き継ぎます）")) return;
  try {
    await fetch("/api/reset", { method: "POST" });  // クリア後の状態はSSEで届く
  } catch (e) { alert("会議の切り替えに失敗しました"); }
}

async function setMode(mode) {
  if (busyMode || stopped) return;
  busyMode = mode; renderModes(mode);
  try {
    const res = await fetch("/api/mode", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    if (!res.ok) {
      const d = await res.json().catch(() => ({}));
      alert("モード切替に失敗: " + (d.error || res.status));
    }
  } catch (e) { alert("モード切替に失敗しました"); }
  finally { busyMode = null; }
}

async function stopSession() {
  if (stopped) return;
  if (!confirm("アプリを終了しますか？（次の会議に移るだけなら「新しい会議」を使ってください）")) return;
  stopped = true; setStatus(false);
  try { await fetch("/api/stop", { method: "POST" }); } catch (e) {}
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function connect() {
  const src = new EventSource("/api/stream");
  src.onmessage = (ev) => { try { render(JSON.parse(ev.data)); } catch (e) {} };
  src.addEventListener("end", () => { stopped = true; setStatus(false); src.close(); });
  src.onerror = () => { /* EventSourceが自動再接続する */ };
}

$("end").onclick = stopSession;
$("reset").onclick = resetMeeting;
$("agenda-set").onclick = setAgenda;
$("agenda").addEventListener("keydown", (e) => { if (e.key === "Enter") setAgenda(); });
$("enroll-btn").onclick = enrollPerson;
$("roster-lock").addEventListener("change", toggleRoster);
$("speaker-count").addEventListener("change", setSpeakerCount);
// 初期描画 + ライブ接続
fetch("/api/state").then((r) => r.json()).then(render).catch(() => {});
connect();
</script>
</body>
</html>
"""
