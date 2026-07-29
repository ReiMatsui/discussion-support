#!/usr/bin/env python3
"""Sonioxの区切りを真の話者境界で割り直したら、どこまで取り返せるかを測る.

**問い**（§36 の次）: 残る誤帰属のうち、1秒未満の分の68%は「またぎ」——
Sonioxが1区間に2人を入れてしまった発話である。切り出した音声に最初から2人
入っているので、声紋をどう工夫しても届かない。では**区切りを直せば**どこまで
届くのか。

真の境界（GT）で割るのは**上限を測るため**で、実装案ではない。実装するなら
pyannote の話者交代を使うことになるが、pyannote が正しい位置で切れなければ
この上限には届かない。上限が小さければ、その先を調べる必要も無い。

割り直しで返ってくるものは2種類ある。混同すると数字が二重に見える。

  取り返し1: **少数派の時間**。いまは1発話に1人しか割り当てないので、
             割り込んだ側の発話は**必ず落ちている**（採点にすら現れない）。
             割れば拾える——ただし断片は短いので、当たるとは限らない。
  取り返し2: **主発話の判定**。2人分が混ざった音声で声紋を取っているので、
             主たる話者の判定も汚れている。混ざりを除いて取り直せば当たる
             ようになるかもしれない。こちらは既存の採点（件数・文字数）に
             そのまま効く。

取り返し2は「席の音声で決め直す」経路の発話に限って測る。それ以外は声紋層が
判定しており、そちらを記録から再現することはできない（誤差を上乗せするより
測らないほうが誠実）。なお誤帰属の88%はこの経路に集中している（§35）。

重なり（2人以上が同時に喋っている区間）は、どちらの取り返しにも使わない。
区切りをどこに置いても声は分けられないので、区切りの問題ではない。

GT は採点にしか使わない。新規録音も STT の再課金も不要。

使い方:
    uv run python eval/segment_split_ceiling.py --prefix 2026-07-20
"""
from __future__ import annotations

import argparse
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _gtlib  # noqa: E402
import _pipeline as pipe  # noqa: E402
import _textgt  # noqa: E402
import decompose_attribution as dec  # noqa: E402

from das.asr.live._constants import SR, UNSURE_SPEAKER  # noqa: E402

MINORITY_MIN_RATIO = 0.10   # 少数派がこの割合を超えたら「またいでいる」
PIECE_MIN_SEC = 0.15        # これより短い断片は声紋を取らない（取れない）


# ------------------------------------------------ 発話を話者ごとに割る


def spans_of(run: str) -> list[tuple[int, int, str]]:
    """正解の (開始, 終了, コード) を時刻順で返す（コードは S1/S2/S3）."""
    src = _textgt.source_of(run)
    corpus = _textgt._corpus_turns(src) if src else None
    if not corpus:
        return []
    return sorted((int(c["ms"]), int(c["end_ms"]), c["code"]) for c in corpus)


def split_by_speaker(a: int, b: int, spans) -> tuple[dict, int]:
    """[a,b] を「1人だけが喋っている区間」に割る.

    戻り値: ({コード: [(開始, 終了), …]}, 重なりの合計ms)。

    重なりを別扱いにするのは、そこが区切りの問題ではないため——2人が同時に
    喋っている音声は、どこで切っても1人分にはならない。
    """
    edges = {a, b}
    for x, y, _c in spans:
        if y > a and x < b:
            edges.add(max(a, x))
            edges.add(min(b, y))
    pts = sorted(edges)
    out: dict[str, list[tuple[int, int]]] = {}
    overlap = 0
    for lo, hi in pairwise(pts):
        if hi <= lo:
            continue
        mid = (lo + hi) / 2
        active = {c for x, y, c in spans if x <= mid < y}
        if len(active) == 1:
            out.setdefault(active.pop(), []).append((lo, hi))
        elif len(active) > 1:
            overlap += hi - lo
    return out, overlap


def _audio(pcm: np.ndarray, base_ms: int, parts) -> np.ndarray:
    """発話の音声から、指定区間だけを抜いて繋ぐ."""
    chunks = [pcm[int((x - base_ms) / 1000 * SR):int((y - base_ms) / 1000 * SR)]
              for x, y in parts]
    chunks = [c for c in chunks if c.size]
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


class Splitter:
    """発話ごとに「主たる話者だけの音声」と「割り込んだ側の音声」を作る."""

    def __init__(self, spans) -> None:
        self.spans = spans
        self.info: dict[int, dict] = {}   # ms -> 割り方の内訳

    def __call__(self, u: dict, wav: np.ndarray,
                 revisable: bool) -> dict[str, np.ndarray]:
        a = int(u["ms"])
        b = int(u.get("end") or a)
        out = {"": wav} if revisable else {}
        parts, overlap = split_by_speaker(a, b, self.spans)
        secs = {c: sum(y - x for x, y in v) / 1000.0 for c, v in parts.items()}
        total = sum(secs.values())
        if not total:
            return out
        top = max(secs, key=secs.get)
        minority = total - secs[top]
        self.info[a] = {"top": top, "total_s": total, "minor_s": minority,
                        "secs": secs, "overlap_s": overlap / 1000.0,
                        "straddle": minority / total > MINORITY_MIN_RATIO}
        if not self.info[a]["straddle"]:
            return out
        if secs[top] >= PIECE_MIN_SEC:
            out["major"] = _audio(wav, a, parts[top])
        for c, v in parts.items():
            if c != top and secs[c] >= PIECE_MIN_SEC:
                out[f"minor:{c}"] = _audio(wav, a, v)
        return out


# ---------------------------------------------------------------- 採点


def _outcomes(steps, final, mapping) -> list[dict]:
    return [{"ms": int(st["ms"]),
             "chars": len(str(st["utt"].get("_text") or "")),
             "outcome": ("未確定" if f == UNSURE_SPEAKER
                         else "正解" if mapping.get(f) == st["code"]
                         else "誤帰属")}
            for f, st in zip(final, steps, strict=True)]


def measure(run: str, vp) -> dict | None:
    spans = spans_of(run)
    if not spans:
        return None
    sp = Splitter(spans)
    data = pipe.replay_seats(run, vp, align="text", query=sp)
    if data is None:
        return None
    steps = data["steps"]

    # 取り返し2: 主たる話者だけの音声で決め直す（席の経路の発話のみ）
    now = pipe.apply_schedule(steps)
    split = pipe.apply_schedule(steps, name="major")
    # 「major」を持たない発話は声紋が無い＝現行のまま。混ざりの無い発話まで
    # 未確定に落ちてしまうので、そこは現行の答えで埋め戻す。
    split = [s if "major" in st["embs"] else n
             for n, s, st in zip(now, split, steps, strict=True)]
    pairs = [(f, st["code"]) for f, st in zip(now, steps, strict=True)]
    _a, m = _gtlib.best_mapping(pairs, dec.GT_CODES, unsure=UNSURE_SPEAKER)

    # 取り返し1: 割り込んだ側の断片を、その時点の席の参照で当てにいく
    minor_s = minor_hit_s = minor_n = minor_hit = 0.0
    total_s = overlap_s = 0.0
    for st in steps:
        info = sp.info.get(int(st["ms"]))
        if not info:
            continue
        total_s += info["total_s"]
        overlap_s += info["overlap_s"]
        if not info["straddle"]:
            continue
        for name, emb in st["embs"].items():
            if not name.startswith("minor:"):
                continue
            code = name.split(":", 1)[1]
            got = pipe.pick_nearest(emb, st["refs"])
            # 断片ごとの秒数で数える（割り込みが2人いると二重に数えてしまう）
            sec = info["secs"].get(code, 0.0)
            minor_s += sec
            minor_n += 1
            if got is not None and m.get(got) == code:
                minor_hit_s += sec
                minor_hit += 1
    return {"run": run,
            "now": _outcomes(steps, now, m),
            "split": _outcomes(steps, split, m),
            "straddle": sum(1 for st in steps
                            if sp.info.get(int(st["ms"]), {}).get("straddle")),
            "overlapped": sum(
                1 for st in steps
                if (i := sp.info.get(int(st["ms"])))
                and i["overlap_s"] > 0.10 * (i["total_s"] + i["overlap_s"])),
            "n": len(steps), "total_s": total_s, "overlap_s": overlap_s,
            "minor_s": minor_s, "minor_hit_s": minor_hit_s,
            "minor_n": minor_n, "minor_hit": minor_hit}


def _rates(rows, weigh):
    tot = sum(weigh(r) for r in rows) or 1
    return tuple(sum(weigh(r) for r in rows if r["outcome"] == k) / tot
                 for k in ("正解", "誤帰属", "未確定"))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prefix", default="2026-07-20")
    p.add_argument("--model", default="redimnet")
    args = p.parse_args(argv)

    from das.asr.live._voice_profiles import VoiceProfiles
    vp = VoiceProfiles(model=args.model)
    runs = [r for r in sorted(dec.discover()) if r.startswith(args.prefix)]
    got = []
    for run in runs:
        r = measure(run, vp)
        if r:
            got.append(r)
            print(f"# {run} 済み（{r['n']}発話・またぎ {r['straddle']}）",
                  flush=True)
    if not got:
        raise SystemExit("# 測れるランが無い")

    n = sum(r["n"] for r in got)
    st = sum(r["straddle"] for r in got)
    ovn = sum(r["overlapped"] for r in got)
    print(f"\n# {len(got)}本 {n}発話")
    print(f"#   順番にまたぐ（1人ずつ喋っていて境界を跨いだ）  {st}件（{st / n:.0%}）")
    print(f"#   同時に喋っている（発話時間の1割以上が重なり）    {ovn}件"
          f"（{ovn / n:.0%}）← 区切りでは分けられない")

    print("\n## 取り返し2: 主たる話者だけの音声で決め直す")
    print(f"{'条件':<16}{'件数':>7}{'正解':>8}{'誤帰属':>8}{'未確定':>8}"
          f"{'  ':>2}{'文字:正解':>10}{'誤帰属':>8}{'未確定':>8}")
    for key, label in (("now", "いまの実装"), ("split", "真の境界で分割")):
        rows = [x for r in got for x in r[key]]
        c = _rates(rows, lambda r: 1)
        w = _rates(rows, lambda r: r["chars"])
        print(f"{label:<16}{len(rows):>7}{c[0]:>8.1%}{c[1]:>8.1%}{c[2]:>8.1%}"
              f"{'  ':>2}{w[0]:>10.1%}{w[1]:>8.1%}{w[2]:>8.1%}")

    tot_s = sum(r["total_s"] for r in got)
    ov_s = sum(r["overlap_s"] for r in got)
    mi_s = sum(r["minor_s"] for r in got)
    hit_s = sum(r["minor_hit_s"] for r in got)
    mi_n = sum(r["minor_n"] for r in got)
    hit_n = sum(r["minor_hit"] for r in got)
    print("\n## 取り返し1: 割り込んだ側の断片")
    print(f"  いま落としている時間  {mi_s:.0f}秒（発話時間の {mi_s / tot_s:.1%}）"
          f"／断片 {mi_n:.0f}件")
    if mi_n:
        print(f"  断片だけを当てにいくと {hit_n / mi_n:.0%} 当たる"
              f"（時間では {hit_s / tot_s:.1%} を取り返す）")
    print(f"  参考: 2人以上が同時に喋っている時間 {ov_s:.0f}秒"
          f"（{ov_s / tot_s:.1%}）は、区切りを直しても分けられない")

    print("\n読み方: 取り返し1と2は別のもの。1は「いま採点にすら現れない")
    print("  割り込みを拾えるか」、2は「混ざりを除けば主発話が当たるか」。")
    print("  どちらも真の境界を使った**上限**で、実装（pyannoteの話者交代）が")
    print("  この位置で切れなければ届かない。")


if __name__ == "__main__":
    main()
