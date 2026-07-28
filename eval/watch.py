#!/usr/bin/env python3
"""録音を再生しながら、議事録がどう出ていたかをその場で見る.

    uv run python eval/watch.py 2026-07-20_1623

ブラウザが開き、音声の再生に合わせて発話が1件ずつ現れる——**そのセッションで
実際に画面に出ていた姿**の再現。話者名は `*.turns.jsonl`（保存された議事録）を
そのまま使うので、「聞こえている声」と「そう判定された名前」を突き合わせられる。

正解（`eval/gt_<セッション>.json`）があれば、行ごとに ○（正解）/ ×（誤帰属）/
―（未確定）を出し、そこまでの累計も上に表示する。どこで崩れ、どこで持ち直すかが
耳で追える。

新しく録音するわけでも、STTを叩き直すわけでもない（APIコストは0）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _gtlib  # noqa: E402
from annotate import peaks_of, prepare_audio  # noqa: E402

from das.asr.live._constants import UNSURE_SPEAKER  # noqa: E402

GT_CODES = ("S1", "S2", "S3", "S4", "S5", "S6")
UNSURE_NAMES = {"未確定", UNSURE_SPEAKER}


def load_turns(session: str) -> list[dict]:
    path = ROOT / "transcripts" / f"{session}.turns.jsonl"
    if not path.exists():
        sys.exit(f"# {path} がありません")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def attach_truth(session: str, turns: list[dict]) -> dict:
    """正解があれば、行ごとの ○/×/― を付ける（無ければ何もしない）.

    表示名（参加者A・実名）と正解のコード（S1…）に共通の名前空間は無いので、
    最も当たる1:1対応を取ってから突き合わせる——採点側（`_gtlib.best_mapping`）
    と同じ規則にして、画面の印象と数字が食い違わないようにする。
    """
    gt_path = ROOT / "eval" / f"gt_{session}.json"
    if not gt_path.exists():
        return {"has_gt": False, "names": {}, "labels": {}}
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    labels = gt.get("labels") or {}
    pairs = []
    for t in turns:
        code = labels.get(str(t["turn_id"]))
        if code in GT_CODES:
            pairs.append((str(t["speaker"]), code))
    _a, mapping = _gtlib.best_mapping(pairs, GT_CODES, unsure="未確定")
    for t in turns:
        code = labels.get(str(t["turn_id"]))
        if code not in GT_CODES:
            t["mark"] = ""          # 正解が付いていない行（相槌・複数人など）
        elif str(t["speaker"]) in UNSURE_NAMES:
            t["mark"] = "unsure"
        elif mapping.get(str(t["speaker"])) == code:
            t["mark"] = "ok"
        else:
            t["mark"] = "ng"
        t["gt"] = code or ""
    return {"has_gt": True, "names": gt.get("speaker_names") or {},
            "labels": labels}


def attach_today(session: str, turns: list[dict], labels: dict) -> bool:
    """「今日の実装ならこう出る」を各行に足す（声紋モデルを読むので少し待つ）.

    **なぜ要るのか**: `turns.jsonl` は収録した日の判定である。2026-07-20 の
    録音を再生すると、§27-§28 の改善が入る前の姿（未確定だらけ）が見える。
    それを今日の実力と誤解しないよう、同じ画面で並べて見られるようにする。

    対象は正解が付いた発話（相槌を除く）だけ。それ以外は空欄になる。
    """
    import _pipeline as pipe

    from das.asr.live._voice_profiles import VoiceProfiles

    print("# 今日の実装で判定し直しています（声紋モデルを読みます）…", flush=True)
    keys = pipe.current_keys(session, VoiceProfiles(model="redimnet"))
    if not keys:
        print("# 今日の実装は計算できませんでした（wav か diag が足りない）")
        return False
    # 内部キー（人物1 / @diar:2）を、出てきた順に 参加者A・B… へ置き換える。
    order: dict[str, str] = {}
    for t in turns:
        k = keys.get(int(t["ms"]))
        if k is None or k == UNSURE_SPEAKER:
            continue
        if k not in order:
            order[k] = f"参加者{chr(ord('A') + len(order))}"
    pairs = []
    for t in turns:
        k = keys.get(int(t["ms"]))
        code = labels.get(str(t["turn_id"]))
        if k is not None and code in GT_CODES:
            pairs.append((order.get(k, "未確定"), code))
    _a, mapping = _gtlib.best_mapping(pairs, GT_CODES, unsure="未確定")
    for t in turns:
        k = keys.get(int(t["ms"]))
        if k is None:
            t["today"], t["today_mark"] = "", ""
            continue
        name = order.get(k, "未確定")
        t["today"] = name
        code = labels.get(str(t["turn_id"]))
        if code not in GT_CODES:
            t["today_mark"] = ""
        elif name == "未確定":
            t["today_mark"] = "unsure"
        else:
            t["today_mark"] = "ok" if mapping.get(name) == code else "ng"
    return True


PAGE = r"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8"><title>議事録の再生</title>
<style>
  :root{--bg:#f6f7f9;--card:#fff;--line:#e3e6ea;--ink:#1a1d21;--sub:#6b7280;}
  *{box-sizing:border-box}
  body{font-family:-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;
       margin:0;background:var(--bg);color:var(--ink);font-size:14px}
  header{position:sticky;top:0;z-index:5;background:var(--card);
         border-bottom:1px solid var(--line);padding:8px 14px}
  .row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
  h1{font-size:14px;margin:0;font-weight:600}
  button{font:inherit;padding:5px 12px;border:1px solid var(--line);
         border-radius:8px;background:#fff;cursor:pointer}
  button.primary{background:#2563eb;border-color:#2563eb;color:#fff}
  #wave{width:100%;height:52px;display:block;margin:6px 0 2px;background:#fff;
        border:1px solid var(--line);border-radius:8px;cursor:pointer}
  #clock{font-variant-numeric:tabular-nums;min-width:82px;color:var(--sub)}
  .tally{font-size:12px;color:var(--sub);margin-left:auto}
  .tally b{font-variant-numeric:tabular-nums;color:var(--ink)}
  #list{max-width:900px;margin:12px auto 60vh;padding:0 12px}
  .t{background:var(--card);border:1px solid var(--line);border-left:4px solid #cbd5e1;
     border-radius:10px;padding:8px 12px;margin-bottom:6px;display:flex;gap:12px;
     align-items:baseline;opacity:0;transform:translateY(6px);
     transition:opacity .25s,transform .25s}
  .t.shown{opacity:1;transform:none}
  .t.now{box-shadow:0 0 0 3px #2563eb22;border-color:#2563eb}
  .t .ts{font-size:11px;color:var(--sub);width:52px;flex-shrink:0;
         font-variant-numeric:tabular-nums}
  .t .who{width:104px;flex-shrink:0;font-weight:600;font-size:13px}
  .t .tx{flex:1;line-height:1.55}
  .t .mk{width:20px;text-align:center;flex-shrink:0;font-weight:700}
  .ok{color:#16a34a}.ng{color:#dc2626}.unsure{color:#9ca3af}
  .t.ng{border-left-color:#dc2626;background:#fff5f5}
  .t.unsure{border-left-color:#cbd5e1}
  .legend{font-size:12px;color:var(--sub)}
</style></head><body>
<header>
  <div class="row">
    <h1 id="title">…</h1>
    <span class="legend" id="meta"></span>
    <span class="tally" id="tally"></span>
  </div>
  <canvas id="wave"></canvas>
  <div class="row">
    <button id="play" class="primary">▶ 再生</button>
    <span id="clock">0:00</span>
    <label class="legend">速度 <select id="rate">
      <option>0.75</option><option selected>1</option><option>1.25</option>
      <option>1.5</option><option>2</option></select></label>
    <label class="legend"><input type="checkbox" id="all"> 最初から全部見せる</label>
    <label class="legend" id="todaybox" style="display:none">
      <input type="checkbox" id="today"> 今日の実装で見る</label>
    <span class="legend">○ 正解 / × 誤帰属 / ― 未確定（「今日の実装」では相槌など採点対象外は — になります）</span>
  </div>
</header>
<div id="list"></div>
<script>
let T=[], dur=0, peaks=[], audio=null, hasGt=false, cur=-1;
const $=id=>document.getElementById(id);
const fmt=s=>Math.floor(s/60)+":"+String(Math.floor(s%60)).padStart(2,"0");
const COLORS=["#2563eb","#dc2626","#16a34a","#d97706","#7c3aed","#0891b2"];
const useToday=()=>$("today")&&$("today").checked;
const who=t=>useToday()?(t.today||"—"):t.speaker;   // 空欄＝採点対象外
const markOf=t=>useToday()?(t.today_mark||""):(t.mark||"");
const colorOf=(()=>{const m={};let i=0;return n=>{
  if(n==="未確定"||n==="?"||n==="—")return "#9ca3af";
  if(!(n in m))m[n]=COLORS[i++%COLORS.length];return m[n];};})();

async function boot(){
  const info=await (await fetch("/info")).json();
  T=info.turns; dur=info.duration; peaks=info.peaks; hasGt=info.has_gt;
  $("title").textContent=info.title;
  $("meta").textContent=`${T.length}発話・${(dur/60).toFixed(1)}分`
    + (hasGt?"（正解つき）":"（正解なし）");
  audio=new Audio("/audio"); audio.preload="auto";
  audio.addEventListener("timeupdate",onTime);
  audio.addEventListener("ended",()=>$("play").textContent="▶ 再生");
  if(info.has_today) $("todaybox").style.display="";
  render(); draw();
}
$("today").addEventListener("change",()=>{ render(); draw(); tally(cur); });
function render(){
  const L=$("list"); L.innerHTML="";
  T.forEach((t,i)=>{
    const d=document.createElement("div");
    const mark=markOf(t);
    d.className="t"+(mark==="ng"?" ng":mark==="unsure"?" unsure":"");
    d.id="t"+i;
    const mk=mark==="ok"?'<span class="mk ok">○</span>'
            :mark==="ng"?'<span class="mk ng">×</span>'
            :mark==="unsure"?'<span class="mk unsure">―</span>'
            :'<span class="mk"></span>';
    d.innerHTML=`<div class="ts">${fmt(t.ms/1000)}</div>`
      +`<div class="who" style="color:${colorOf(who(t))}">${who(t)}</div>`
      +`<div class="tx">${t.text}</div>`+mk;
    d.style.borderLeftColor=colorOf(who(t));
    L.appendChild(d);
  });
  if($("all").checked) T.forEach((_,i)=>$("t"+i).classList.add("shown"));
}
function draw(){
  const c=$("wave"),w=c.clientWidth,h=52;
  if(c.width!==w*devicePixelRatio){c.width=w*devicePixelRatio;c.height=h*devicePixelRatio;}
  const g=c.getContext("2d");
  g.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0); g.clearRect(0,0,w,h);
  g.fillStyle="#e5e7eb";
  for(let x=0;x<w;x++){const p=peaks[Math.floor(x/w*peaks.length)]||0;
    const bh=Math.max(1,p*(h-12));g.fillRect(x,(h-12)/2-bh/2+2,1,bh);}
  T.forEach(t=>{const x0=t.ms/1000/dur*w,x1=t.end_ms/1000/dur*w;
    g.fillStyle=colorOf(who(t));g.globalAlpha=.9;
    g.fillRect(x0,h-9,Math.max(1.2,x1-x0),7);g.globalAlpha=1;});
  if(audio){const x=audio.currentTime/dur*w;g.fillStyle="#111827";g.fillRect(x-1,0,2,h);}
}
function tally(upto){
  let ok=0,ng=0,un=0;
  for(let i=0;i<=upto&&i<T.length;i++){
    const m=markOf(T[i]);
    if(m==="ok")ok++; else if(m==="ng")ng++; else if(m==="unsure")un++;}
  const n=ok+ng+un;
  $("tally").innerHTML = (hasGt && n)
    ? `ここまで <b>${n}</b>件 — 正解 <b>${(ok/n*100).toFixed(0)}%</b>`
      + ` / 誤帰属 <b>${(ng/n*100).toFixed(0)}%</b> / 未確定 <b>${(un/n*100).toFixed(0)}%</b>`
    : "";
}
function onTime(){
  const now=audio.currentTime;
  $("clock").textContent=fmt(now);
  let idx=-1;
  for(let i=0;i<T.length;i++){ if(T[i].ms/1000<=now+0.05) idx=i; else break; }
  if(idx!==cur){
    if(cur>=0)$("t"+cur)?.classList.remove("now");
    for(let i=Math.max(cur,0);i<=idx;i++) $("t"+i)?.classList.add("shown");
    cur=idx;
    if(cur>=0){const el=$("t"+cur);el.classList.add("now");
      el.scrollIntoView({block:"center",behavior:"smooth"});}
    tally(cur);
  }
  draw();
}
$("play").addEventListener("click",()=>{
  if(audio.paused){audio.playbackRate=parseFloat($("rate").value);audio.play();
    $("play").textContent="⏸ 停止";}
  else{audio.pause();$("play").textContent="▶ 再生";}});
$("rate").addEventListener("change",()=>{audio.playbackRate=parseFloat($("rate").value);});
$("all").addEventListener("change",()=>{ if($("all").checked)
  T.forEach((_,i)=>$("t"+i).classList.add("shown")); else render(); });
$("wave").addEventListener("click",e=>{
  const r=e.currentTarget.getBoundingClientRect();
  audio.currentTime=(e.clientX-r.left)/r.width*dur; cur=-1;
  if($("all").checked===false) render();
});
addEventListener("resize",draw);
addEventListener("keydown",e=>{ if(e.key===" "){e.preventDefault();$("play").click();}});
boot();
</script></body></html>
"""


class _State:
    def __init__(self, title, wav, turns, peaks, duration, has_gt,
                 has_today=False):
        self.title, self.wav, self.turns = title, wav, turns
        self.peaks, self.duration, self.has_gt = peaks, duration, has_gt
        self.has_today = has_today


class _Handler(BaseHTTPRequestHandler):
    state: _State

    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype: str, extra=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        s = self.state
        if self.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path.startswith("/info"):
            body = {"title": s.title, "turns": s.turns, "peaks": s.peaks,
                    "duration": s.duration, "has_gt": s.has_gt,
                    "has_today": s.has_today}
            self._send(200, json.dumps(body, ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")
        elif self.path.startswith("/audio"):
            self._send_audio()
        else:
            self._send(404, b"not found", "text/plain")

    def _send_audio(self):
        """Range 付きで返す（付けないとブラウザが頭出しできない）."""
        data = self.state.wav.read_bytes()
        m = re.match(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
        if not m:
            self._send(200, data, "audio/wav", [("Accept-Ranges", "bytes")])
            return
        a = int(m.group(1))
        b = min(int(m.group(2)) if m.group(2) else len(data) - 1, len(data) - 1)
        chunk = data[a:b + 1]
        self.send_response(206)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("session", help="収録セッション名（例 2026-07-20_1623）")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--today", action="store_true",
                   help="今日の実装ならどう出るかも並べる（声紋モデルを読む）")
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args(argv)

    wav_src = ROOT / "transcripts" / f"{args.session}.wav"
    if not wav_src.exists():
        sys.exit(f"# {wav_src} がありません")
    turns = load_turns(args.session)
    truth = attach_truth(args.session, turns)
    has_today = False
    if args.today:
        has_today = attach_today(args.session, turns, truth["labels"])
    wav, y = prepare_audio(wav_src, None)
    state = _State(title=args.session, wav=wav, turns=turns,
                   peaks=peaks_of(y), duration=len(y) / 16000,
                   has_gt=truth["has_gt"], has_today=has_today)
    _Handler.state = state
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"# {args.session}: {len(turns)}発話・{state.duration / 60:.1f}分"
          f"{'（正解つき）' if truth['has_gt'] else ''}")
    print(f"# {url} を開いてください（Ctrl-C で終了）")
    if not args.no_open:
        with __import__("contextlib").suppress(Exception):
            webbrowser.open(url)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\n# 終了しました")


if __name__ == "__main__":
    main()
