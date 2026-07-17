#!/usr/bin/env python3
"""話者正解アノテーターHTMLを新セッション用に生成する.

従来は既存の gt_annotator_*.html を手で複製して TURNS 埋め込みを差し替えて
いた（handoff §12 の「再生成」）。ヘッダの直し漏れが実際に起きていた
（gt_annotator_2026-06-25_1520.html のタイトルが 142016 のまま）ため、
テンプレートを本スクリプトに固定してコマンド1発にする。テンプレートは
2026-06-25_1520 版（連続再生・ズレ補正・再生余白つきの最新JS）と同一で、
アノテーション形式・出力GT（labels: {turn_id: S1|S2|S3|MULTI|UNK}）は不変。

使い方:
    uv run python eval/make_gt_annotator.py <セッション名>
    → eval/gt_annotator_<セッション名>.html を生成
    → ブラウザで開き、transcripts/<セッション名>.wav を選択して
      発話ごとに S1/S2/S3/MULTI/UNK を付け、書き出したJSONを
      eval/gt_<セッション名>.json に保存
    → uv run python eval/eval_speaker_gt.py eval/gt_<セッション名>.json で採点
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # eval/（_gtlib 用）

from _gtlib import read_jsonl

ROOT = Path(__file__).resolve().parent.parent

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>話者正解アノテーション — __SESSION__</title>
<style>
  :root { --bg:#fafafa; --card:#fff; --border:#ddd; --accent:#2563eb; --done:#16a34a; }
  body { font-family: -apple-system, "Hiragino Sans", sans-serif; margin:0; background:var(--bg); color:#222; }
  header { position:sticky; top:0; background:var(--card); border-bottom:1px solid var(--border); padding:10px 16px; z-index:10; }
  header h1 { font-size:15px; margin:0 0 8px; }
  .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-bottom:6px; }
  .row label { font-size:12px; color:#555; }
  input[type=text] { padding:4px 8px; border:1px solid var(--border); border-radius:6px; font-size:13px; width:100px; }
  input[type=number] { padding:4px 6px; border:1px solid var(--border); border-radius:6px; font-size:13px; width:60px; }
  button { padding:6px 12px; border:1px solid var(--border); border-radius:8px; background:#fff; cursor:pointer; font-size:13px; }
  button:hover { background:#f0f4ff; }
  button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
  button.playing { background:#dc2626; color:#fff; border-color:#dc2626; }
  #progress { font-size:13px; color:#555; }
  #list { max-width:860px; margin:12px auto 120px; padding:0 12px; }
  .turn { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:10px 14px; margin-bottom:8px;
          display:flex; gap:12px; align-items:center; }
  .turn.current { border-color:var(--accent); box-shadow:0 0 0 2px #2563eb33; background:#f0f6ff; }
  .turn.labeled { border-left:5px solid var(--done); }
  .tid { font-size:11px; color:#888; width:70px; flex-shrink:0; }
  .txt { flex:1; font-size:14px; line-height:1.5; }
  .sys { font-size:11px; color:#999; display:none; }
  body.showsys .sys { display:block; }
  .gt { font-weight:600; font-size:13px; width:72px; text-align:center; flex-shrink:0;
        border-radius:6px; padding:3px 0; background:#eee; color:#666; }
  .gt.set { background:#dcfce7; color:#166534; }
  .btns { display:flex; gap:4px; flex-shrink:0; }
  .btns button { padding:4px 8px; font-size:12px; }
  footer { position:fixed; bottom:0; left:0; right:0; background:var(--card); border-top:1px solid var(--border);
           padding:10px 16px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  .kbd { font-size:11px; color:#888; margin-left:auto; }
  #dropzone { border:2px dashed var(--border); border-radius:8px; padding:4px 10px; font-size:12px; color:#777; }
  #timeinfo { font-variant-numeric:tabular-nums; font-size:12px; color:#555; min-width:90px; }
</style>
</head>
<body>
<header>
  <h1>話者正解アノテーション — transcripts/__SESSION__（__NTURNS__発話・約__MINUTES__分）</h1>
  <div class="row">
    <span id="dropzone">① 音声: <input type="file" id="wav" accept=".wav,audio/*">（transcripts/__SESSION__.wav を選択）</span>
    <label>話者名: <input type="text" id="n1" value="話者1"></label>
    <label><input type="text" id="n2" value="話者2"></label>
    <label><input type="text" id="n3" value="話者3"></label>
    <label><input type="checkbox" id="showsys"> システムのラベルも表示</label>
  </div>
  <div class="row">
    <button id="cont" class="primary">▶ 連続再生（推奨）</button>
    <span id="timeinfo">0:00.0</span>
    <label>ズレ補正 <input type="number" id="offset" value="0" step="0.5"> 秒
      （表示より音が<b>遅れて</b>聞こえるならプラスに）</label>
    <label>個別再生の余白 <input type="number" id="pad" value="0.7" step="0.1" min="0"> 秒</label>
    <label>速度 <input type="number" id="rate" value="1.0" step="0.25" min="0.5" max="2"></label>
    <span id="progress"></span>
  </div>
</header>
<div id="list"></div>
<footer>
  <button class="primary" id="export">② 正解をJSONで保存</button>
  <button id="clear">全クリア</button>
  <span id="status" style="font-size:12px;color:#555"></span>
  <span class="kbd">キー: <b>1/2/3</b>=話者 <b>0</b>=不明 <b>9</b>=複数人 <b>Space</b>=連続再生/停止
    <b>←</b>=3秒戻す <b>→</b>=3秒送る <b>Enter</b>=この行だけ再生 <b>↑↓</b>=行移動</span>
</footer>
<script>
const TURNS = __TURNS__;
const SESSION = "__SESSION__";
const KEY = "gt_" + SESSION;
let labels = {};
try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) {}
let cur = 0, audio = null, stopTimer = null, continuous = false;

const list = document.getElementById("list");
const num = id => parseFloat(document.getElementById(id).value) || 0;
function spName(code){
  if (code === "S1") return document.getElementById("n1").value || "話者1";
  if (code === "S2") return document.getElementById("n2").value || "話者2";
  if (code === "S3") return document.getElementById("n3").value || "話者3";
  if (code === "MULTI") return "複数人";
  if (code === "UNK") return "不明";
  return "—";
}
function fmt(ms){ const s = ms/1000; return Math.floor(s/60)+":"+String((s%60).toFixed(1)).padStart(4,"0"); }

function render(){
  list.innerHTML = "";
  TURNS.forEach((t,i)=>{
    const div = document.createElement("div");
    div.className = "turn" + (i===cur?" current":"") + (labels[t.turn_id]?" labeled":"");
    div.innerHTML = `
      <div class="tid">#${t.turn_id}<br>${fmt(t.ms)}</div>
      <div class="txt">${t.text}<div class="sys">システム: ${t.speaker}</div></div>
      <div class="gt ${labels[t.turn_id]?"set":""}">${spName(labels[t.turn_id])}</div>
      <div class="btns">
        <button data-i="${i}" data-c="S1">1</button>
        <button data-i="${i}" data-c="S2">2</button>
        <button data-i="${i}" data-c="S3">3</button>
        <button data-i="${i}" data-c="MULTI">複</button>
        <button data-i="${i}" data-c="UNK">?</button>
        <button data-i="${i}" data-c="PLAY">▶</button>
      </div>`;
    div.addEventListener("click", e => {
      if (e.target.tagName === "BUTTON") {
        const c = e.target.dataset.c;
        if (c === "PLAY") { stopContinuous(); select(parseInt(e.target.dataset.i)); playOne(); }
        else assign(parseInt(e.target.dataset.i), c);
      } else select(i);
    });
    list.appendChild(div);
  });
  const done = TURNS.filter(t=>labels[t.turn_id]).length;
  document.getElementById("progress").textContent = `進捗: ${done} / ${TURNS.length}`;
}
// render() は全再構築で重いので、ハイライト移動だけ軽量に行う
function highlight(i, scroll=true){
  if (i === cur && list.children[cur]?.classList.contains("current")) return;
  list.children[cur]?.classList.remove("current");
  cur = Math.max(0, Math.min(TURNS.length-1, i));
  const el = list.children[cur];
  el.classList.add("current");
  if (scroll) el.scrollIntoView({block:"center", behavior:"smooth"});
}
function select(i, scroll=true){ highlight(i, scroll); }
function updateRow(i){
  const t = TURNS[i], el = list.children[i];
  el.classList.toggle("labeled", !!labels[t.turn_id]);
  const gt = el.querySelector(".gt");
  gt.textContent = spName(labels[t.turn_id]);
  gt.classList.toggle("set", !!labels[t.turn_id]);
  const done = TURNS.filter(x=>labels[x.turn_id]).length;
  document.getElementById("progress").textContent = `進捗: ${done} / ${TURNS.length}`;
}
function assign(i, code){
  labels[TURNS[i].turn_id] = code;
  localStorage.setItem(KEY, JSON.stringify(labels));
  updateRow(i);
  if (continuous) return;          // 連続再生中: 音声は止めず、追従に任せる
  select(i+1);
  playOne();
}

// --- 再生 ---
function ensureAudio(){
  if (!audio) document.getElementById("status").textContent = "先に wav を選択してください";
  return !!audio;
}
// 個別再生: 前後に余白を付け、次の発話の頭までは食い込んでも再生する
function playOne(){
  if (!ensureAudio()) return;
  stopContinuous();
  const t = TURNS[cur], pad = num("pad"), off = num("offset");
  clearTimeout(stopTimer);
  audio.playbackRate = num("rate");
  audio.currentTime = Math.max(0, t.ms/1000 + off - pad);
  audio.play();
  const dur = (t.end_ms - t.ms)/1000 + pad*2;
  stopTimer = setTimeout(()=>audio.pause(), dur*1000 / audio.playbackRate);
}
// 連続再生: 流しっぱなしで currentTime に該当する発話を自動ハイライト
function startContinuous(fromCur=true){
  if (!ensureAudio()) return;
  clearTimeout(stopTimer);
  continuous = true;
  audio.playbackRate = num("rate");
  if (fromCur) audio.currentTime = Math.max(0, TURNS[cur].ms/1000 + num("offset") - 0.3);
  audio.play();
  const b = document.getElementById("cont");
  b.textContent = "⏸ 停止"; b.classList.add("playing");
}
function stopContinuous(){
  if (!continuous) return;
  continuous = false;
  audio && audio.pause();
  const b = document.getElementById("cont");
  b.textContent = "▶ 連続再生（推奨）"; b.classList.remove("playing");
}
function turnAt(sec){
  // offset 補正後の再生位置に「鳴っている or 直近に始まった」発話を返す
  const ms = (sec - num("offset")) * 1000;
  let best = 0;
  for (let i = 0; i < TURNS.length; i++){
    if (TURNS[i].ms <= ms + 200) best = i; else break;
  }
  return best;
}
let lastFollow = 0;
function onTime(){
  document.getElementById("timeinfo").textContent = fmt(audio.currentTime*1000);
  if (!continuous) return;
  const now = performance.now();
  if (now - lastFollow < 150) return;
  lastFollow = now;
  highlight(turnAt(audio.currentTime));
}

document.getElementById("wav").addEventListener("change", e=>{
  const f = e.target.files[0];
  if (!f) return;
  audio = new Audio(URL.createObjectURL(f));
  audio.addEventListener("timeupdate", onTime);
  audio.addEventListener("ended", stopContinuous);
  document.getElementById("status").textContent = "音声OK: " + f.name + " — 「▶ 連続再生」でどうぞ";
});
document.getElementById("cont").addEventListener("click", ()=> continuous ? stopContinuous() : startContinuous());
document.getElementById("showsys").addEventListener("change", e=>{
  document.body.classList.toggle("showsys", e.target.checked);
});
["n1","n2","n3"].forEach(id=>document.getElementById(id).addEventListener("input", render));
document.getElementById("rate").addEventListener("input", ()=>{ if (audio) audio.playbackRate = num("rate"); });
document.getElementById("export").addEventListener("click", ()=>{
  const done = TURNS.filter(t=>labels[t.turn_id]).length;
  const out = {
    session: SESSION,
    created: new Date().toISOString(),
    speaker_names: {S1: spName("S1"), S2: spName("S2"), S3: spName("S3")},
    labeled: done, total: TURNS.length,
    labels: labels
  };
  const blob = new Blob([JSON.stringify(out, null, 1)], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `gt_${SESSION}.json`;
  a.click();
  document.getElementById("status").textContent = `保存しました（${done}/${TURNS.length}件）。eval/ フォルダに置いてください`;
});
document.getElementById("clear").addEventListener("click", ()=>{
  if (confirm("全てのラベルを消しますか？")) { labels = {}; localStorage.removeItem(KEY); render(); }
});
document.addEventListener("keydown", e=>{
  if (e.target.tagName === "INPUT") return;
  if (e.key === "1") assign(cur, "S1");
  else if (e.key === "2") assign(cur, "S2");
  else if (e.key === "3") assign(cur, "S3");
  else if (e.key === "9") assign(cur, "MULTI");
  else if (e.key === "0") assign(cur, "UNK");
  else if (e.key === " ") { e.preventDefault(); if (!ensureAudio()) return;
    continuous ? stopContinuous() : startContinuous(); }
  else if (e.key === "Enter") { e.preventDefault(); playOne(); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); if (audio) audio.currentTime = Math.max(0, audio.currentTime - 3); }
  else if (e.key === "ArrowRight") { e.preventDefault(); if (audio) audio.currentTime += 3; }
  else if (e.key === "ArrowDown") { e.preventDefault(); stopContinuous(); select(cur+1); }
  else if (e.key === "ArrowUp") { e.preventDefault(); stopContinuous(); select(cur-1); }
});
render();
</script>
</body>
</html>
"""


def main(session: str) -> None:
    turns_path = ROOT / "transcripts" / f"{session}.turns.jsonl"
    if not turns_path.exists():
        sys.exit(f"{turns_path} がありません（セッション名を確認してください）")
    turns = read_jsonl(turns_path)
    keep = [{"turn_id": t["turn_id"], "ms": t["ms"], "end_ms": t["end_ms"],
             "speaker": t.get("speaker", ""), "text": t.get("text", "")}
            for t in turns]
    minutes = max((t["end_ms"] for t in keep), default=0) / 60000
    html = (_TEMPLATE
            .replace("__TURNS__", json.dumps(keep, ensure_ascii=False))
            .replace("__SESSION__", session)
            .replace("__NTURNS__", str(len(keep)))
            .replace("__MINUTES__", f"{minutes:.1f}"))
    out = ROOT / "eval" / f"gt_annotator_{session}.html"
    out.write_text(html, encoding="utf-8")
    print(f"{out.relative_to(ROOT)}: {len(keep)}発話・約{minutes:.1f}分 "
          f"(音声: transcripts/{session}.wav をブラウザで選択)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
