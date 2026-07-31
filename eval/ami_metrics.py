"""AMI会議（英語）のランを標準指標（DER / cpWER）で採点する——外部と同一データ比較.

`standard_metrics.py`（千葉・日本語・cpCER）の英語版。AMI は SA-ASR 研究の
標準ベンチマークで、ここで出す cpWER は published な数字（例: DNCASR の
AMI cpWER 31.5%、オフライン・MDM条件）と**同一コーパス・同一指標**で並ぶ。
条件の差（本システムはストリーミング・Mix-Headset 1ch・ゼロショット商用STT）
は読み手が判断できるよう明記する。

正解: data/ami/manual/words/<meeting>.<A-D>.words.xml（AMI公式の単語時刻）。
仮説: transcripts/<run>.turns.jsonl（--lang en で流したラン）。

使い方:
  uv run python eval/ami_metrics.py <run名> --meeting ES2004a
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from meeteval.wer.wer.cp import cp_word_error_rate  # noqa: E402
from pyannote.core import Annotation, Segment  # noqa: E402
from pyannote.metrics.diarization import DiarizationErrorRate  # noqa: E402

MERGE_GAP_SEC = 0.5
UNSURE_DISPLAY = "未確定"
_PUNCT = re.compile(r"[^a-z0-9' ]")


def norm_en(text: str) -> str:
    """英語の慣例的な正規化（小文字化・アポストロフィ以外の記号除去）."""
    return _PUNCT.sub(" ", text.lower()).strip()


def load_ami_tokens(meeting: str) -> list[tuple[float, float, str, str]]:
    rows = []
    for p in sorted((ROOT / "data" / "ami" / "manual" / "words").glob(
            f"{meeting}.*.words.xml")):
        who = p.name.split(".")[1]
        for w in ET.parse(p).getroot():
            if w.get("punc") == "true":
                continue
            try:
                s, e = float(w.get("starttime")), float(w.get("endtime"))
            except (TypeError, ValueError):
                continue
            text = norm_en(w.text or "")
            if e > s and text:
                rows.append((s, e, who, text))
    rows.sort(key=lambda x: (x[0], x[1]))
    return rows


def reference(meeting: str, until_sec: float):
    ann = Annotation()
    texts: dict[str, list[str]] = {}
    ordered: list[tuple[float, str]] = []
    cur: dict[str, list] = {}
    for s, e, who, text in load_ami_tokens(meeting):
        if s >= until_sec:
            continue
        e = min(e, until_sec)
        texts.setdefault(who, []).append(text)
        ordered.append((s, text))
        c = cur.get(who)
        if c is not None and s - c[1] <= MERGE_GAP_SEC:
            c[1] = max(c[1], e)
        else:
            if c is not None:
                ann[Segment(c[0], c[1]), f"r-{who}-{len(texts[who])}"] = who
            cur[who] = [s, e]
    for who, c in cur.items():
        ann[Segment(c[0], c[1]), f"r-{who}-t"] = who
    return (ann, {w: " ".join(t) for w, t in texts.items()},
            " ".join(x for _s, x in sorted(ordered)))


def hypothesis(run: str, *, drop_unsure: bool = False):
    ann = Annotation()
    texts: dict[str, list[str]] = {}
    ordered: list[tuple[int, str]] = []
    end = 0.0
    with open(ROOT / "transcripts" / f"{run}.turns.jsonl", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            if t.get("ms") is None or t.get("end_ms") is None:
                continue
            sp = str(t["speaker"])
            if sp == "ファシリテーター":
                continue
            if drop_unsure and sp == UNSURE_DISPLAY:
                continue
            end = max(end, t["end_ms"] / 1000)
            text = norm_en(str(t.get("text") or ""))
            ann[Segment(t["ms"] / 1000, t["end_ms"] / 1000), f"h{t['turn_id']}"] = sp
            texts.setdefault(sp, []).append(text)
            ordered.append((t["ms"], text))
    return (ann, {s: " ".join(x) for s, x in texts.items()},
            " ".join(x for _m, x in sorted(ordered)), end)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run", help="採点するラン（transcripts/<run>.turns.jsonl）")
    p.add_argument("--meeting", default="ES2004a")
    args = p.parse_args(argv)

    hyp_ann, hyp_txt, hyp_ordered, hyp_end = hypothesis(args.run)
    ref_end = max(e for _s, e, _w, _t in load_ami_tokens(args.meeting))
    dur = min(ref_end, hyp_end)
    if hyp_end < ref_end * 0.97:
        print(f"# 注意: 仮説が {hyp_end:.0f}s で終わる（正解 {ref_end:.0f}s）。短い側で評価")
    ref_ann, ref_txt, ref_ordered = reference(args.meeting, dur)
    uem = Segment(0, dur)

    der = DiarizationErrorRate(collar=0.5)
    d = der(ref_ann, hyp_ann, uem=uem, detailed=True)
    print(f"## {args.meeting} × {args.run}（{dur/60:.1f}分・話者{len(ref_txt)}人）")
    print(f"DER (collar±0.25s, 重なり込み): {d['diarization error rate']:.1%}")
    tot = d["total"] or 1
    print(f"  miss {d['missed detection']/tot:.1%} / FA {d['false alarm']/tot:.1%}"
          f" / confusion {d['confusion']/tot:.1%}")
    d2 = DiarizationErrorRate(collar=0.5)(
        ref_ann, hypothesis(args.run, drop_unsure=True)[0], uem=uem, detailed=True)
    print(f"  未確定を外した confusion: {d2['confusion']/(d2['total'] or 1):.1%}")
    dso = DiarizationErrorRate(collar=0.5, skip_overlap=True)(ref_ann, hyp_ann, uem=uem)
    print(f"DER (重なり除く): {abs(dso):.1%}")

    cp = cp_word_error_rate(ref_txt, hyp_txt)
    asr = cp_word_error_rate({"all": ref_ordered}, {"all": hyp_ordered})
    print(f"cpWER: {cp.error_rate:.1%}  （参考: DNCASR オフラインMDMで31.5%）")
    print(f"  話者無視のWER（STT品質のみ）: {asr.error_rate:.1%}"
          f" → 帰属による上乗せ ≈ {cp.error_rate - asr.error_rate:+.1%}")


if __name__ == "__main__":
    main()
