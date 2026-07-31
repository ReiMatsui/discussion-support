"""AMI会議（英語）のランを標準指標（DER / cpWER）で採点する——外部と同一データ比較.

`standard_metrics.py`（千葉・日本語・cpCER）の英語版。AMI は SA-ASR 研究の
標準ベンチマークで、ここで出す cpWER は published な数字（例: DNCASR の
AMI cpWER 31.5%、オフライン・MDM条件）と**同一コーパス・同一指標**で並ぶ。
条件の差（本システムはストリーミング・Mix-Headset 1ch・ゼロショット商用STT）
は読み手が判断できるよう明記する。

正解: data/ami/manual/words/<meeting>.<A-D>.words.xml（AMI公式の単語時刻）。
仮説: transcripts/<run>.turns.jsonl（--lang en で流したラン）。

変種「＋検出区間」: diag の diar_seg（pyannote の全区間, §48.2）で、仮説音声に
覆われていない時間を「発話あり」として穴埋めした DER も出す。文字は増えない
（cpWER 不変）が、「誰が・いつ」の申告を完全にした条件で、区間のみを出力する
比較先（LS-EEND 等）と同じ課題定義になる（§48.3。恣意性の議論は §48.2）。
ラベルは pyannote クラスタ名なので、切断でクラスタが分断されたランでは
効果が薄れる。

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


def fill_with_detected(hyp_ann, run: str):
    """仮説音声に覆われていない時間を diar_seg で穴埋めした仮説を返す.

    穴埋めのラベルは、そのクラスタと時間重なりが最大の**仮説話者**
    （参加者X）に写像する——システムは実行中にクラスタ→参加者の対応
    （diarization_speaker_keys / クラスタ名前付け）を持っており、ここでは
    それをログから多数決で復元している。生のクラスタ名のまま出すと、
    正解4人と対応済みの参加者ラベルに加えて別クラスタが立ち、正しく
    検出できた時間まで confusion に数えられる（クリーンランで実測:
    素 45.1% → 生ラベル穴埋め 47.9% と悪化。写像後の測定が本命）。
    対応が取れないクラスタは自ラベルのまま（安全側＝confusion行き）。

    区間が無ければ (None) を返す（diar_seg 未記録の古いラン）。
    """
    from collections import defaultdict

    from pyannote.core import Timeline
    diag = ROOT / "transcripts" / f"{run}.diag.jsonl"
    segs = []
    if diag.exists():
        with open(diag, encoding="utf-8") as f:
            for line in f:
                if '"diar_seg"' not in line:
                    continue
                d = json.loads(line)
                if d.get("end") is not None:
                    segs.append(d)
    if not segs:
        return None
    # クラスタ→仮説話者の写像（時間重なりの多数決）
    votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for sgm in segs:
        s0, s1 = sgm["ms"] / 1000, sgm["end"] / 1000
        for useg, _tr, spk in hyp_ann.itertracks(yield_label=True):
            ov = min(s1, useg.end) - max(s0, useg.start)
            if ov > 0:
                votes[sgm["spk"]][spk] += ov
    cluster_to = {c: max(v, key=v.get) for c, v in votes.items() if v}
    hyp_tl = hyp_ann.get_timeline().support()
    union = hyp_ann.copy()
    i = 0
    for sgm in segs:
        label = cluster_to.get(sgm["spk"], f"diar:{sgm['spk']}")
        if label == UNSURE_DISPLAY:
            label = f"diar:{sgm['spk']}"   # 未確定へ寄せても情報が無い
        span = Timeline([Segment(sgm["ms"] / 1000, sgm["end"] / 1000)])
        for gap in span.extrude(hyp_tl):
            if gap.duration > 0.05:
                union[gap, f"d{i}"] = label
                i += 1
    return union


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

    filled = fill_with_detected(hyp_ann, args.run)
    if filled is not None:
        df = DiarizationErrorRate(collar=0.0)(ref_ann, filled, uem=uem, detailed=True)
        tf = df["total"] or 1
        print(f"DER (collar 0, ＋検出区間の変種): {df['diarization error rate']:.1%}"
              f" (miss {df['missed detection']/tf:.1%} / FA {df['false alarm']/tf:.1%}"
              f" / conf {df['confusion']/tf:.1%})"
              f"  ※文字は増えない＝cpWERは下の値のまま")
    else:
        print("# diar_seg なし（旧ラン）: ＋検出区間の変種はスキップ")

    cp = cp_word_error_rate(ref_txt, hyp_txt)
    asr = cp_word_error_rate({"all": ref_ordered}, {"all": hyp_ordered})
    print(f"cpWER: {cp.error_rate:.1%}  （参考: DNCASR オフラインMDMで31.5%）")
    print(f"  話者無視のWER（STT品質のみ）: {asr.error_rate:.1%}"
          f" → 帰属による上乗せ ≈ {cp.error_rate - asr.error_rate:+.1%}")


if __name__ == "__main__":
    main()
