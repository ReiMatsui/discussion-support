#!/usr/bin/env python3
"""話者判定用のブラインド採点シート（HTML＋音声クリップ）を生成する.

pyannote ベンチマーク（scripts/benchmark_pyannote.py）が出力した
transcripts/<session>.pyannote_bench.json の不一致ターンから無作為抽出し、
「誰の声か」を人間が盲検で判定できる HTML シートを作る。
システム（現行/pyannote）の判定は画面に出さず、採点ボタンで自動照合する。

使い方:
  uv run python scripts/make_adjudication_sheet.py --session 2026-06-25_1554
  uv run python scripts/make_adjudication_sheet.py --session 2026-06-25_1614 --n 30 --seed 1

出力:
  transcripts/clips/<session>/clipNN.wav と checklist.html
  → checklist.html をブラウザで開き、判定→採点→結果テキストをコピー
"""
from __future__ import annotations

import argparse
import html
import json
import random
import wave
from pathlib import Path

TRANSCRIPTS = Path("transcripts")


def load_speakers(session: str) -> list[str]:
    """turns.jsonl から実在の話者ラベル一覧を作る（合成・未確定を除く）."""
    skip = {"未確定", "AI", "パートナー", "ファシリテーター", "[Partner]", "AIファシリテーター"}
    seen: dict[str, int] = {}
    for line in (TRANSCRIPTS / f"{session}.turns.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sp = json.loads(line).get("speaker", "")
        if sp and sp not in skip:
            seen[sp] = seen.get(sp, 0) + 1
    return sorted(seen, key=lambda s: -seen[s])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", required=True)
    ap.add_argument("--n", type=int, default=20, help="抽出クリップ数（既定20）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pad-ms", type=int, default=400, help="クリップ前後の余白")
    args = ap.parse_args()

    sess = args.session
    bench_path = TRANSCRIPTS / f"{sess}.pyannote_bench.json"
    wav_path = TRANSCRIPTS / f"{sess}.wav"
    if not bench_path.exists():
        raise SystemExit(f"{bench_path} がありません。先に benchmark_pyannote.py を実行してください")
    d = json.loads(bench_path.read_text(encoding="utf-8"))
    mism = [m for m in d["mismatches"] if m.get("pyannote_speaker_mapped")]
    if not mism:
        raise SystemExit("不一致ターンがありません")
    random.seed(args.seed)
    sample = random.sample(mism, min(args.n, len(mism)))
    sample.sort(key=lambda m: m["ms"])

    out_dir = TRANSCRIPTS / "clips" / sess
    out_dir.mkdir(parents=True, exist_ok=True)
    speakers = load_speakers(sess) + ["その他/不明"]

    items = []
    with wave.open(str(wav_path), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        for i, m in enumerate(sample, 1):
            s = max(0, m["ms"] - args.pad_ms)
            e = m["end_ms"] + args.pad_ms
            w.setpos(int(s * sr / 1000))
            frames = w.readframes(int((e - s) * sr / 1000))
            fn = f"clip{i:02d}.wav"
            with wave.open(str(out_dir / fn), "wb") as o:
                o.setnchannels(ch)
                o.setsampwidth(sw)
                o.setframerate(sr)
                o.writeframes(frames)
            t = m["ms"] // 1000
            items.append({
                "n": i, "file": fn, "time": f"{t // 60}:{t % 60:02d}",
                "text": m["text_head"][:30],
                "cur": m["current_speaker"],
                "py": m["pyannote_speaker_mapped"],
            })

    meta = json.dumps(items, ensure_ascii=False)
    opts = "".join(f'<option value="{s}">{s}</option>' for s in speakers)
    rows = "".join(
        f'<tr><td>{it["n"]}</td><td>{it["time"]}</td>'
        f'<td><audio controls preload="none" src="{it["file"]}"></audio></td>'
        f'<td class="tx">{html.escape(it["text"])}</td>'
        f'<td><select data-n="{it["n"]}"><option value="">--</option>{opts}</select></td></tr>'
        for it in items
    )
    page = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>話者判定シート {sess}</title>
<style>body{{font-family:-apple-system,'Hiragino Sans',sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:.4em .6em;font-size:.9rem}}
.tx{{color:#555}}#result{{margin-top:1rem;padding:1rem;background:#f0f9ff;border-radius:8px;white-space:pre-wrap;font-size:.95rem}}
button{{margin-top:.5rem;padding:.4em 1em}}</style></head><body>
<h2>話者判定シート（ブラインド） {sess}</h2>
<p>各クリップを再生し「実際に誰の声か」を選んでください。システムの判定は採点時に自動照合されます。</p>
<table><tr><th>#</th><th>時刻</th><th>音声</th><th>文字起こし</th><th>誰の声？</th></tr>{rows}</table>
<button onclick="tally()">採点する</button>
<div id="result"></div>
<script>
const META={meta};
function tally(){{
  let cur=0,py=0,none=0,both=0,ans=0,detail=[];
  for(const it of META){{
    const sel=document.querySelector(`select[data-n="${{it.n}}"]`);
    if(!sel||!sel.value)continue; ans++;
    const v=sel.value,c=(v===it.cur),p=(v===it.py);
    if(c&&!p)cur++;else if(p&&!c)py++;else if(c&&p)both++;else none++;
    detail.push(`#${{it.n}} 正解=${{v}} / 現行=${{it.cur}}${{c?"○":"×"}} / pyannote=${{it.py}}${{p?"○":"×"}}`);
  }}
  document.getElementById("result").textContent=
    `セッション {sess}\\n回答 ${{ans}}/${{META.length}}件\\n現行のみ正解: ${{cur}}\\npyannoteのみ正解: ${{py}}\\n両方不正解: ${{none}}\\n\\n`+
    detail.join("\\n")+"\\n\\n↑この結果全体をコピーしてClaudeに貼ってください";
}}
</script></body></html>"""
    (out_dir / "checklist.html").write_text(page, encoding="utf-8")
    print(f"{len(items)}クリップ → {out_dir}/checklist.html をブラウザで開いてください")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
