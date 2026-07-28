"""アノテーション画面（HTML/CSS/JS）。`annotate.py` が読み込んで配信する.

画面と配信を分けているのは、UI をいじるたびにサーバ側を読み直さずに済む
ようにするため。ここには埋め込み変数は無く、データは全て fetch で取りに行く
（音声も /audio から Range 付きで配信されるので、ファイル選択が要らない）。
"""

PAGE = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>話者アノテーション</title>
<style>
  :root {
    --bg:#f6f7f9; --card:#fff; --line:#e3e6ea; --ink:#1a1d21; --sub:#6b7280;
    --accent:#2563eb;
  }
  * { box-sizing:border-box; }
  body { font-family:-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;
         margin:0; background:var(--bg); color:var(--ink); font-size:14px; }

  header { position:sticky; top:0; z-index:20; background:var(--card);
           border-bottom:1px solid var(--line); padding:8px 14px 6px; }
  .titlebar { display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; }
  .titlebar h1 { font-size:14px; margin:0; font-weight:600; }
  .titlebar .meta { font-size:12px; color:var(--sub); }
  #saved { font-size:12px; color:var(--sub); margin-left:auto; }

  /* 全体波形（クリックで移動、色は話者） */
  #wave { width:100%; height:64px; display:block; margin:6px 0 4px;
          background:#fff; border:1px solid var(--line); border-radius:8px;
          cursor:pointer; }

  .controls { display:flex; gap:10px; align-items:center; flex-wrap:wrap;
              font-size:12px; color:var(--sub); }
  button { font:inherit; padding:5px 11px; border:1px solid var(--line);
           border-radius:8px; background:#fff; cursor:pointer; color:var(--ink); }
  button:hover { background:#eef3ff; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  button.primary:hover { filter:brightness(1.08); }
  button.on { background:#111827; border-color:#111827; color:#fff; }
  #clock { font-variant-numeric:tabular-nums; min-width:86px; }
  .spk { display:flex; align-items:center; gap:4px; }
  .spk .chip { width:14px; height:14px; border-radius:4px; display:inline-block; }
  .spk input { width:76px; padding:3px 6px; border:1px solid var(--line);
               border-radius:6px; font:inherit; }
  .spk .num { color:var(--sub); font-size:11px; }

  #list { max-width:1000px; margin:10px auto 96px; padding:0 12px; }
  .seg { background:var(--card); border:1px solid var(--line); border-left:5px solid transparent;
         border-radius:10px; padding:8px 12px; margin-bottom:6px;
         display:flex; gap:12px; align-items:center; }
  .seg.cur { border-color:var(--accent); box-shadow:0 0 0 3px #2563eb22; }
  .seg .t { font-size:11px; color:var(--sub); width:76px; flex-shrink:0;
            font-variant-numeric:tabular-nums; }
  .seg .txt { flex:1; line-height:1.5; word-break:break-word; }
  .seg .txt.empty { color:#9ca3af; }
  .seg .who { width:96px; flex-shrink:0; text-align:center; font-weight:600;
              border-radius:7px; padding:4px 0; background:#f1f3f5; color:#9ca3af;
              font-size:12px; }
  .seg .play { flex-shrink:0; }

  footer { position:fixed; left:0; right:0; bottom:0; background:var(--card);
           border-top:1px solid var(--line); padding:8px 14px;
           display:flex; gap:14px; align-items:center; flex-wrap:wrap; font-size:12px; }
  footer .keys { color:var(--sub); }
  kbd { background:#f1f3f5; border:1px solid var(--line); border-bottom-width:2px;
        border-radius:5px; padding:1px 5px; font-family:inherit; font-size:11px; }
  #bar { flex:1; height:8px; background:#eceff3; border-radius:99px; overflow:hidden;
         min-width:120px; }
  #bar i { display:block; height:100%; background:var(--accent); width:0; }
</style>
</head>
<body>
<header>
  <div class="titlebar">
    <h1 id="title">読み込み中…</h1>
    <span class="meta" id="meta"></span>
    <span id="saved"></span>
  </div>
  <canvas id="wave"></canvas>
  <div class="controls">
    <button id="auto" class="primary">▶ 自動送り</button>
    <button id="cont">連続再生</button>
    <span id="clock">0:00.0</span>
    <label>速度 <select id="rate">
      <option>0.75</option><option selected>1</option><option>1.25</option>
      <option>1.5</option><option>2</option></select></label>
    <label>前後の余白 <select id="pad">
      <option>0</option><option selected>0.3</option><option>0.6</option>
      <option>1</option></select>秒</label>
    <span id="spks" style="display:flex;gap:10px;flex-wrap:wrap"></span>
    <button id="addspk">＋話者</button>
  </div>
</header>

<div id="list"></div>

<footer>
  <span id="progress">—</span>
  <span id="bar"><i></i></span>
  <span class="keys">
    <kbd>1</kbd>…<kbd>9</kbd>話者　<kbd>S</kbd>直前と同じ　<kbd>M</kbd>複数人
    <kbd>0</kbd>不明　<kbd>Backspace</kbd>取消　<kbd>Space</kbd>再生/停止
    <kbd>Enter</kbd>もう一度　<kbd>↑↓</kbd>移動　<kbd>←→</kbd>3秒
  </span>
  <button id="download">JSONを保存</button>
</footer>

<script>
const COLORS = ["#2563eb","#dc2626","#16a34a","#d97706","#7c3aed",
                "#0891b2","#db2777","#65a30d","#475569"];
const MULTI = "MULTI", UNK = "UNK";

let SEGS = [], NAMES = {}, labels = {}, cur = 0;
let audio = null, mode = "idle", stopTimer = null, peaks = [], dur = 0;
let saveTimer = null, dirty = false;

const $ = id => document.getElementById(id);
const fmt = s => Math.floor(s/60) + ":" + (s%60).toFixed(1).padStart(4,"0");
const codes = () => Object.keys(NAMES).filter(c => c.startsWith("S"))
                          .sort((a,b) => +a.slice(1) - +b.slice(1));
function colorOf(code){
  if (code === MULTI) return "#94a3b8";
  if (code === UNK) return "#cbd5e1";
  const i = +String(code).slice(1) - 1;
  return COLORS[i % COLORS.length] || "#94a3b8";
}
function nameOf(code){
  if (!code) return "—";
  if (code === MULTI) return "複数人";
  if (code === UNK) return "不明";
  return NAMES[code] || code;
}

// ---------- 起動 ----------
async function boot(){
  const info = await (await fetch("/info")).json();
  SEGS = info.segments; dur = info.duration; peaks = info.peaks;
  NAMES = info.speaker_names;
  labels = info.labels || {};
  $("title").textContent = info.title;
  $("meta").textContent =
    `${SEGS.length}区間・${(dur/60).toFixed(1)}分　保存先 ${info.gt_path}`;
  audio = new Audio("/audio");
  audio.preload = "auto";
  audio.addEventListener("timeupdate", onTime);
  audio.addEventListener("ended", () => setMode("idle"));
  renderSpeakers();
  renderList();
  drawWave();
  updateProgress();
}

// ---------- 話者チップ ----------
function renderSpeakers(){
  const box = $("spks"); box.innerHTML = "";
  codes().forEach((c, i) => {
    const el = document.createElement("span");
    el.className = "spk";
    el.innerHTML = `<span class="num">${i+1}</span>`
      + `<span class="chip" style="background:${colorOf(c)}"></span>`;
    const inp = document.createElement("input");
    inp.value = NAMES[c];
    inp.addEventListener("input", () => {
      NAMES[c] = inp.value; markDirty(); renderList(); });
    el.appendChild(inp);
    box.appendChild(el);
  });
}
$("addspk").addEventListener("click", () => {
  const n = codes().length + 1;
  if (n > 9) return;
  NAMES["S" + n] = "話者" + n;
  markDirty(); renderSpeakers(); renderList();
});

// ---------- 一覧 ----------
function renderList(){
  const list = $("list");
  list.innerHTML = "";
  SEGS.forEach((s, i) => {
    const d = document.createElement("div");
    d.className = "seg" + (i === cur ? " cur" : "");
    d.dataset.i = i;
    d.innerHTML =
      `<div class="t">#${s.id}<br>${fmt(s.start)}<br>${(s.end-s.start).toFixed(1)}s</div>`
      + `<div class="txt${s.text ? "" : " empty"}">${s.text || "（文字起こしなし）"}</div>`
      + `<div class="who"></div>`
      + `<button class="play">▶</button>`;
    d.querySelector(".play").addEventListener("click", e => {
      e.stopPropagation(); select(i); playSeg(); });
    d.addEventListener("click", () => select(i));
    list.appendChild(d);
    paintRow(i);
  });
}
function paintRow(i){
  const el = $("list").children[i];
  if (!el) return;
  const code = labels[SEGS[i].id];
  const who = el.querySelector(".who");
  who.textContent = nameOf(code);
  if (code){
    who.style.background = colorOf(code) + "22";
    who.style.color = colorOf(code);
    el.style.borderLeftColor = colorOf(code);
  } else {
    who.style.background = "#f1f3f5"; who.style.color = "#9ca3af";
    el.style.borderLeftColor = "transparent";
  }
}
function select(i, scroll = true){
  if (i < 0 || i >= SEGS.length) return;
  $("list").children[cur]?.classList.remove("cur");
  cur = i;
  const el = $("list").children[cur];
  el.classList.add("cur");
  if (scroll) el.scrollIntoView({block:"center", behavior:"smooth"});
  drawWave();
}

// ---------- 波形 ----------
function drawWave(){
  const c = $("wave"), w = c.clientWidth, h = 64;
  if (c.width !== w * devicePixelRatio){
    c.width = w * devicePixelRatio; c.height = h * devicePixelRatio;
  }
  const g = c.getContext("2d");
  g.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  g.clearRect(0, 0, w, h);
  // 背景の波形
  g.fillStyle = "#e5e7eb";
  const n = peaks.length;
  for (let x = 0; x < w; x++){
    const p = peaks[Math.floor(x / w * n)] || 0;
    const bh = Math.max(1, p * (h - 14));
    g.fillRect(x, (h - 14) / 2 - bh / 2 + 2, 1, bh);
  }
  // 区間の帯（ラベル色）
  SEGS.forEach((s, i) => {
    const x0 = s.start / dur * w, x1 = s.end / dur * w;
    const code = labels[s.id];
    g.fillStyle = code ? colorOf(code) : "#cbd5e1";
    g.globalAlpha = code ? 0.9 : 0.5;
    g.fillRect(x0, h - 11, Math.max(1.2, x1 - x0), 8);
    g.globalAlpha = 1;
    if (i === cur){
      g.strokeStyle = "#111827"; g.lineWidth = 1;
      g.strokeRect(x0 - 0.5, h - 12.5, Math.max(2, x1 - x0) + 1, 11);
    }
  });
  // 再生位置
  if (audio){
    const x = audio.currentTime / dur * w;
    g.fillStyle = "#111827";
    g.fillRect(x - 1, 0, 2, h);
  }
}
$("wave").addEventListener("click", e => {
  const r = e.currentTarget.getBoundingClientRect();
  const t = (e.clientX - r.left) / r.width * dur;
  if (audio) audio.currentTime = t;
  let best = 0;
  SEGS.forEach((s, i) => { if (s.start <= t + 0.2) best = i; });
  select(best);
});
addEventListener("resize", drawWave);

// ---------- 再生 ----------
const pad = () => parseFloat($("pad").value);
function setMode(m){
  mode = m;
  $("auto").classList.toggle("on", m === "auto");
  $("cont").classList.toggle("on", m === "cont");
  if (m === "idle"){ clearTimeout(stopTimer); audio && audio.pause(); }
}
/** いまの区間だけ鳴らして止める。自動送りでは止まった所でキー入力を待つ。 */
function playSeg(){
  if (!audio) return;
  clearTimeout(stopTimer);
  const s = SEGS[cur];
  audio.playbackRate = parseFloat($("rate").value);
  audio.currentTime = Math.max(0, s.start - pad());
  audio.play();
  const len = (s.end - s.start + pad() * 2) * 1000 / audio.playbackRate;
  stopTimer = setTimeout(() => audio.pause(), len);
}
$("auto").addEventListener("click", () => {
  if (mode === "auto"){ setMode("idle"); return; }
  setMode("auto"); playSeg();
});
$("cont").addEventListener("click", () => {
  if (mode === "cont"){ setMode("idle"); return; }
  setMode("cont");
  clearTimeout(stopTimer);
  audio.playbackRate = parseFloat($("rate").value);
  audio.currentTime = Math.max(0, SEGS[cur].start - 0.2);
  audio.play();
});
$("rate").addEventListener("change", () => {
  if (audio) audio.playbackRate = parseFloat($("rate").value); });

let lastFollow = 0;
function onTime(){
  $("clock").textContent = fmt(audio.currentTime);
  drawWave();
  if (mode !== "cont") return;
  const now = performance.now();
  if (now - lastFollow < 200) return;
  lastFollow = now;
  let best = cur;
  for (let i = 0; i < SEGS.length; i++){
    if (SEGS[i].start <= audio.currentTime + 0.2) best = i; else break;
  }
  if (best !== cur) select(best);
}

// ---------- ラベル付け ----------
function assign(code){
  const s = SEGS[cur];
  if (code === null) delete labels[s.id]; else labels[s.id] = code;
  paintRow(cur); drawWave(); updateProgress(); markDirty();
  if (code === null) return;
  if (cur < SEGS.length - 1){
    select(cur + 1);
    if (mode === "auto") playSeg();
  } else if (mode === "auto") setMode("idle");
}
function updateProgress(){
  let done = 0, sec = 0, total = 0;
  SEGS.forEach(s => {
    total += s.end - s.start;
    if (labels[s.id]){ done++; sec += s.end - s.start; }
  });
  $("progress").textContent =
    `${done} / ${SEGS.length} 区間（${(sec/60).toFixed(1)} / ${(total/60).toFixed(1)}分）`;
  $("bar").firstElementChild.style.width = (done / SEGS.length * 100) + "%";
}

// ---------- 保存（自動） ----------
function markDirty(){
  dirty = true;
  $("saved").textContent = "保存中…";
  clearTimeout(saveTimer);
  saveTimer = setTimeout(save, 600);
}
async function save(){
  const body = JSON.stringify({speaker_names: NAMES, labels: labels});
  try {
    const r = await fetch("/labels", {method:"POST",
      headers:{"Content-Type":"application/json"}, body});
    const j = await r.json();
    dirty = false;
    $("saved").textContent = "保存済み " + new Date().toLocaleTimeString("ja-JP")
                             + "（" + j.path + "）";
  } catch (e) {
    $("saved").textContent = "保存できません: " + e;
  }
}
addEventListener("beforeunload", e => { if (dirty){ save(); e.preventDefault(); }});
$("download").addEventListener("click", async () => {
  await save();
  location.href = "/labels";
});

// ---------- キー ----------
addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  const k = e.key;
  if (k >= "1" && k <= "9"){
    const c = "S" + k;
    if (!NAMES[c]) { NAMES[c] = "話者" + k; renderSpeakers(); }
    assign(c); e.preventDefault();
  } else if (k === "0"){ assign(UNK); e.preventDefault(); }
  else if (k === "m" || k === "M"){ assign(MULTI); e.preventDefault(); }
  else if (k === "s" || k === "S"){
    for (let i = cur - 1; i >= 0; i--){
      if (labels[SEGS[i].id]){ assign(labels[SEGS[i].id]); break; }
    }
    e.preventDefault();
  }
  else if (k === "Backspace"){
    if (cur > 0 && !labels[SEGS[cur].id]) select(cur - 1);
    assign(null); e.preventDefault();
  }
  else if (k === " "){
    e.preventDefault();
    if (mode === "idle"){ setMode("auto"); playSeg(); } else setMode("idle");
  }
  else if (k === "Enter"){ e.preventDefault(); playSeg(); }
  else if (k === "ArrowDown"){ e.preventDefault(); select(cur + 1);
                               if (mode === "auto") playSeg(); }
  else if (k === "ArrowUp"){ e.preventDefault(); select(cur - 1);
                             if (mode === "auto") playSeg(); }
  else if (k === "ArrowLeft"){ e.preventDefault();
    if (audio) audio.currentTime = Math.max(0, audio.currentTime - 3); }
  else if (k === "ArrowRight"){ e.preventDefault();
    if (audio) audio.currentTime += 3; }
});

boot();
</script>
</body>
</html>
"""
