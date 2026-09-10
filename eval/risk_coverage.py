#!/usr/bin/env python3
"""棄権の閾値を振って risk–coverage（名前を出す割合 × 出したときの誤り）を出す.

**問い**: 選択的話者同定の本質は「確信の持てる発話にだけ名前を出す」ことだが、
成績はこれまで閾値1点（表3の被覆と名指し中の正しさ）でしか示していない。
閾値を動かしたとき、被覆（coverage）と名指し中の誤り（risk）がどう交換される
かを曲線にし、基準系（Soniox のラベル素通し＋長さで棄権）と同じ軸で並べる。
選択的予測（selective prediction）の標準の見せ方で、先行研究と比較できる形。

対象は**介入時点の判定**（介入層が見ている断面。遡及訂正前。§49.15）。
再生は `decision_time.py` と同じ `_pipeline`（replay_seats → apply_schedule の
予定表なし → 門番）で、席の貼り直しまで含む。確信度は声紋の最近傍との
コサイン類似度（記録の `sim`、席で決め直した発話は席の参照との内積）。
文字数で重み付けし、相槌は分母から除く（§13.2。`_pipeline.gt_rows` と同じ）。

`--mode record` は再生せず記録の判定だけを使う（クラウドで動くが、席の
貼り直しが入らないので介入時点の断面より低く出る。傾向を見る用）。

  本手法:   閾値 t を振る。sim < t の発話は未確定に落とす（sim の無い発話は不変）
  基準系:   Soniox のラベルをそのまま名前とみなし、文字数 L 未満を未確定にする
            （ラベルに確信度が無いので、棄権の手掛かりは長さだけ）
  併用:     本手法に長さの棄権 L も掛ける

Mac で回す（録音 wav と声紋モデルが要る。API キーは不要）:
  uv run python eval/risk_coverage.py
  uv run python eval/risk_coverage.py --csv eval/_risk_coverage.csv
  uv run python eval/risk_coverage.py --mode record   # 再生なし（クラウド可）
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _gtlib  # noqa: E402
import _pipeline as pipe  # noqa: E402
import decompose_attribution as dec  # noqa: E402

from das.asr.live._constants import UNSURE_SPEAKER  # noqa: E402

GROUPS = (("校正9本", "2026-07-20"), ("持ち越し4本", "2026-07-16"))
SIM_THRESHOLDS = [round(0.30 + 0.02 * i, 2) for i in range(21)]   # 0.30..0.70
LEN_THRESHOLDS = (0, 5, 10, 20, 30, 50)
COVERAGE_POINTS = (0.5, 0.7, 0.8, 0.9)


def _item(run: str, u: dict, code: str, ours: str, sim) -> dict:
    return {
        "run": run,
        "ours": ours,
        "label": (f"{run}:{u.get('label')}"
                  if u.get("label") not in (None, "", "?") else UNSURE_SPEAKER),
        "code": code,
        "chars": len(str(u.get("_text") or "")),
        "sim": float(sim) if sim not in (None, "", "None") else None,
    }


def load_record(runs: list[str]) -> list[dict]:
    """再生なし。記録の判定（席の貼り直しなし）と記録の sim を使う."""
    out = []
    for run in runs:
        for u, code in pipe.gt_rows(run, align="text") or []:
            out.append(_item(run, u, code, pipe.resolved_key(u), u.get("sim")))
    return out


def load_replay(runs: list[str], vp) -> list[dict]:
    """`decision_time.py` と同じ再生で介入時点の判定を作る（席の貼り直し込み）."""
    import numpy as np
    out = []
    for run in runs:
        data = pipe.replay_seats(run, vp, align="text")
        if data is None:
            continue
        steps = data["steps"]
        final = pipe.apply_impure_lowsim_guard(
            pipe.apply_schedule(steps, schedule=(), interval=float("inf")), steps)
        for st, f in zip(steps, final, strict=True):
            u = st["utt"]
            sim = u.get("sim")
            if st["revisable"] and st["emb"] is not None and len(st["refs"]) >= 2:
                sim = max(float(np.dot(st["emb"], v)) for v in st["refs"].values())
            out.append(_item(run, u, st["code"], f, sim))
    return out


def load(runs: list[str], mode: str) -> list[dict]:
    if mode == "record":
        return load_record(runs)
    try:
        from das.asr.live._voice_profiles import VoiceProfiles
        vp = VoiceProfiles(model="redimnet")
    except Exception as e:  # torch や重みが無い環境
        if mode == "replay":
            raise
        print(f"# 再生できないため記録の判定で代用します（{type(e).__name__}）。"
              "介入時点の断面より低く出ます", flush=True)
        return load_record(runs)
    return load_replay(runs, vp)


def tally(items: list[dict], key: str, keep) -> tuple[float, float, float]:
    """(被覆, 名指し中の誤り, 全体正解率) を文字数重みで返す.

    最適1:1対応はラン単位で取る（キーの名前空間がランごとに違うため）。
    """
    w = sum(it["chars"] for it in items) or 1
    named = Counter()
    by_run: dict[str, list] = {}
    for it in items:
        f = it[key] if keep(it) else UNSURE_SPEAKER
        by_run.setdefault(it["run"], []).append((f, it["code"], it["chars"]))
    for pairs in by_run.values():
        _a, m = _gtlib.best_mapping([(f, c) for f, c, _ in pairs], dec.GT_CODES,
                                    unsure=UNSURE_SPEAKER)
        for f, c, ch in pairs:
            if f == UNSURE_SPEAKER:
                continue
            named["all"] += ch
            named["ok" if m.get(f) == c else "ng"] += ch
    cov = named["all"] / w
    risk = named["ng"] / named["all"] if named["all"] else 0.0
    return cov, risk, named["ok"] / w


def curve_ours(items, length: int = 0):
    pts = []
    for t in SIM_THRESHOLDS:
        pts.append((f"t={t:.2f}", *tally(
            items, "ours",
            lambda it, t=t: it["chars"] >= length and (it["sim"] is None or it["sim"] >= t))))
    return pts


def curve_soniox(items):
    return [(f"L={ln}", *tally(items, "label", lambda it, ln=ln: it["chars"] >= ln))
            for ln in LEN_THRESHOLDS]


def risk_at(pts, cov_target: float) -> float | None:
    """被覆が cov_target 以上の点のうち最小の誤り（無ければ None）."""
    cands = [r for _n, c, r, _a in pts if c >= cov_target]
    return min(cands) if cands else None


def show(title: str, pts) -> None:
    print(f"\n## {title}")
    print(f"{'条件':<8}{'被覆':>8}{'名指し中の誤り':>14}{'全体正解率':>12}")
    for name, cov, risk, acc in pts:
        print(f"{name:<8}{cov * 100:7.1f}%{risk * 100:13.1f}%{acc * 100:11.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=None, help="全点を CSV に書く")
    ap.add_argument("--mode", default="auto", choices=["auto", "replay", "record"],
                    help="replay=再生（Mac）/ record=記録のみ / auto=可能なら再生")
    args = ap.parse_args()

    all_runs = sorted(dec.discover())
    rows_out = []
    for label, prefix in GROUPS:
        runs = [r for r in all_runs if r.startswith(prefix)]
        items = load(runs, args.mode)
        if not items:
            continue
        print(f"\n# {label}（{len(runs)}本・実質発話 {len(items)} 件・"
              f"{sum(i['chars'] for i in items)} 文字・sim あり {sum(1 for i in items if i['sim'] is not None)} 件）")
        base = tally(items, "ours", lambda it: True)
        print(f"現行の規則（閾値そのまま）: 被覆 {base[0] * 100:.1f}% / "
              f"名指し中の誤り {base[1] * 100:.1f}% / 全体正解率 {base[2] * 100:.1f}%")
        curves = {
            "本手法（sim の閾値）": curve_ours(items),
            "本手法（sim の閾値・20文字未満は棄権）": curve_ours(items, 20),
            "基準系 Soniox ラベル素通し（長さで棄権）": curve_soniox(items),
        }
        for name, pts in curves.items():
            show(name, pts)
            rows_out += [(label, name, *p) for p in pts]
        print("\n## 被覆を揃えたときの名指し中の誤り")
        print(f"{'被覆≥':<8}" + "".join(f"{n[:14]:>18}" for n in curves))
        for cv in COVERAGE_POINTS:
            cells = []
            for pts in curves.values():
                r = risk_at(pts, cv)
                cells.append("     届かない" if r is None else f"{r * 100:16.1f}%")
            print(f"{cv * 100:5.0f}%   " + "".join(f"{c:>18}" for c in cells))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["group", "system", "condition", "coverage", "risk", "accuracy"])
            w.writerows(rows_out)
        print(f"\n# 書き出し: {args.csv}")


if __name__ == "__main__":
    main()
