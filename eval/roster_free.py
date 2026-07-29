#!/usr/bin/env python3
"""人数を教えなくても、自分で人をまとめられるか（§39 の続き）.

**問い**: 席の仕組みは「席を持つN人のうち誰か」を解いているだけで、その N は
外から貰っている。人数の入力が当てにならない現場では、そこが崩れて文字ベース
91.5% → 67.5% まで落ちる（§39）。ならば **N を貰う代わりに自分で作れないか**。
発話の声紋を寄せ集めて人の集合を作れるなら、席の仕組みはそのまま使える。

答えの順に三つ測る。

  1. **分離しているか**  同じ人どうしの類似度と別人どうしの類似度が、
     そもそも分かれているか。分かれていなければ何をしても無理なので、
     ここが最初の関門。発話長で切って見る（短い発話は原理的に不利）。
  2. **人数を当てられるか**  固有ギャップとシルエットで話者数を推定し、
     正解（3人）に当たるかを見る。
  3. **成績はどこまで戻るか**  クラスタをそのまま答えにして採点する。
     人数を知っている場合（k=3固定）と、推定した場合を分けて出す。

**これは上限であって実装可能な数字ではない**。会話全体を見てからまとめる
（非因果）ので、ライブでは同じことはできない。上限が低ければ実装を考える
必要も無い、という順序で使う。

使い方:
    uv run python eval/roster_free.py --prefix 2026-07-20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _gtlib  # noqa: E402
import _utt_embeddings as ue  # noqa: E402
import decompose_attribution as dec  # noqa: E402

MAX_K = 8
BANDS = ((0, 500), (500, 1000), (1000, 2000), (2000, 10**9))


# ------------------------------------------------------------ 1. 分離度


def separation(emb: np.ndarray, code: list[str], dur: np.ndarray) -> list[dict]:
    """発話長ごとに、同じ人／別人の類似度分布と、両者を分ける線の性能を返す.

    「分ける線の性能」は等誤り率（同じ人を別人と言う率と、別人を同じ人と言う率
    が釣り合う点の誤り率）。0.5 なら区別できていない、0 なら完全に分かれている。
    """
    sims = emb @ emb.T
    same = np.array(code)[:, None] == np.array(code)[None, :]
    out = []
    for lo, hi in BANDS:
        sel = (dur >= lo) & (dur < hi)
        if sel.sum() < 5:
            continue
        # 片側の三角だけ、かつ両方が同じ帯にある組
        idx = np.where(sel)[0]
        iu = np.triu_indices(len(idx), k=1)
        s = sims[np.ix_(idx, idx)][iu]
        y = same[np.ix_(idx, idx)][iu]
        if y.sum() == 0 or (~y).sum() == 0:
            continue
        out.append({"band": f"{lo / 1000:g}〜{hi / 1000:g}秒" if hi < 10**8
                    else f"{lo / 1000:g}秒以上",
                    "n": int(sel.sum()),
                    "same": float(s[y].mean()), "diff": float(s[~y].mean()),
                    "eer": _eer(s[y], s[~y])})
    return out


def _eer(same: np.ndarray, diff: np.ndarray) -> float:
    """等誤り率（しきい値を動かして両側の誤りが釣り合う点）."""
    best = 1.0
    for t in np.quantile(np.concatenate([same, diff]), np.linspace(0, 1, 201)):
        fr = float((same < t).mean())      # 同じ人を別人と言う
        fa = float((diff >= t).mean())     # 別人を同じ人と言う
        best = min(best, max(fr, fa))
    return best


# -------------------------------------------------------- 2. 人数の推定


def _linkage(emb: np.ndarray):
    d = 1.0 - emb @ emb.T
    np.fill_diagonal(d, 0.0)
    d = np.clip((d + d.T) / 2, 0, None)
    return linkage(squareform(d, checks=False), method="average")


def estimate_k(emb: np.ndarray) -> dict[str, int]:
    """話者数を推定する（固有ギャップ／シルエット）."""
    return {"固有ギャップ": _eigengap(emb), "シルエット": _silhouette_k(emb)}


def _eigengap(emb: np.ndarray) -> int:
    """正規化した親和行列の固有値の並びで、いちばん大きな段差を探す（定番）."""
    a = np.clip(emb @ emb.T, 0, None)
    np.fill_diagonal(a, 0.0)
    d = a.sum(1)
    d[d == 0] = 1e-9
    inv = 1.0 / np.sqrt(d)
    lap = np.eye(len(a)) - (a * inv[:, None]) * inv[None, :]
    w = np.sort(np.linalg.eigvalsh(lap))[:MAX_K + 1]
    gaps = np.diff(w)
    return int(np.argmax(gaps) + 1)


def _silhouette_k(emb: np.ndarray) -> int:
    """k=2..8 で切って、いちばんまとまりの良い k を選ぶ."""
    z = _linkage(emb)
    d = 1.0 - emb @ emb.T
    np.fill_diagonal(d, 0.0)
    best, best_k = -2.0, 1
    for k in range(2, MAX_K + 1):
        lab = fcluster(z, k, criterion="maxclust")
        if len(set(lab)) < 2:
            continue
        s = _silhouette(d, lab)
        if s > best:
            best, best_k = s, k
    return best_k


def _silhouette(d: np.ndarray, lab: np.ndarray) -> float:
    """シルエット係数（自分の群への近さ vs 最も近い他群への近さ）."""
    labs = np.unique(lab)
    vals = []
    for i in range(len(lab)):
        own = lab[i] == lab
        own[i] = False
        if own.sum() == 0:
            continue
        a = d[i][own].mean()
        b = min(d[i][lab == c].mean() for c in labs if c != lab[i])
        vals.append((b - a) / max(a, b))
    return float(np.mean(vals)) if vals else -1.0


# ------------------------------------------------------------ 3. 成績


def cluster_score(emb, code, chars, k: int) -> tuple[float, float, float, float]:
    """クラスタをそのまま答えにして採点する（件数の正解率と文字の正解率）."""
    lab = fcluster(_linkage(emb), k, criterion="maxclust")
    pairs = [(f"c{x}", c) for x, c in zip(lab, code, strict=True)]
    _a, m = _gtlib.best_mapping(pairs, dec.GT_CODES, unsure="?")
    ok = [m.get(p) == c for p, c in pairs]
    n = len(pairs)
    w = int(chars.sum()) or 1
    wok = int(sum(x for x, g in zip(chars, ok, strict=True) if g))
    return sum(ok) / n, 1 - sum(ok) / n, wok / w, 1 - wok / w


# ------------------------------------- 4. 長い発話でまとめ、短い発話を寄せる


def anchored_score(emb, code, chars, dur, k, *, min_ms: int
                   ) -> tuple[float, float, float, float, int]:
    """長い発話だけでまとめて参照を作り、全発話をその参照に寄せる.

    席の仕組みと同じ形である——短い発話どうしを比べるのではなく、**長い音声
    から作った参照**と比べる。違いは参照の作り方だけで、席は人数の設定から
    作り、こちらは長い発話のまとまりから作る。人数を教えずに同じ強さが出るか。
    """
    anchor = dur >= min_ms
    if anchor.sum() < k or len(set(np.array(code)[anchor])) < 2:
        return (0.0, 1.0, 0.0, 1.0, 0)
    lab = fcluster(_linkage(emb[anchor]), k, criterion="maxclust")
    refs = []
    for c in np.unique(lab):
        v = emb[anchor][lab == c].mean(0)
        n = np.linalg.norm(v)
        refs.append(v / n if n else v)
    refs = np.array(refs)
    pred = (emb @ refs.T).argmax(1)
    pairs = [(f"c{x}", c) for x, c in zip(pred, code, strict=True)]
    _a, m = _gtlib.best_mapping(pairs, dec.GT_CODES, unsure="?")
    ok = [m.get(p) == c for p, c in pairs]
    n = len(pairs)
    w = int(chars.sum()) or 1
    wok = int(sum(x for x, g in zip(chars, ok, strict=True) if g))
    return sum(ok) / n, 1 - sum(ok) / n, wok / w, 1 - wok / w, int(anchor.sum())


def estimate_k_long(emb, dur, min_ms: int) -> int:
    """長い発話だけで話者数を推定する（短い発話は分離しないので雑音になる）."""
    sel = dur >= min_ms
    if sel.sum() < 4:
        return 1
    return _silhouette_k(emb[sel])


# ------------------------------------------------ 5. 因果的にやった場合


def causal_score(emb, code, chars, dur, ms, *, min_ms: int, k=None,
                 every_sec: float = 30.0, retro: bool = True):
    """その時点までの長い発話だけでまとめ直す（ライブで実際にできる形）.

    非因果の上限（4節）は会話全体を見ている。ライブでできるのは「いままでに
    聞いた長い発話でまとめ直す」ところまでで、序盤は材料が無い。どれだけ
    目減りするかがそのまま実装の価値になる。

    `retro=True` は本番と同じ遡及訂正——まとめ直した時点で、過去の発話も
    新しい参照で貼り直す。声紋は控えてあるので計算は増えない（§28）。
    """
    order = np.argsort(ms)
    refs = None
    next_at = 0.0
    pred: dict[int, int] = {}
    decided: list[int] = []
    t0 = float(ms[order[0]])
    for i in order:
        elapsed = (float(ms[i]) - t0) / 1000.0
        if elapsed >= next_at:
            next_at = elapsed + every_sec
            seen = [j for j in order if ms[j] <= ms[i] and dur[j] >= min_ms]
            if len(seen) >= 4:
                e = emb[seen]
                kk = k if k is not None else _silhouette_k(e)
                kk = min(kk, len(seen))
                lab = fcluster(_linkage(e), kk, criterion="maxclust")
                rs = []
                for c in np.unique(lab):
                    v = e[lab == c].mean(0)
                    nv = np.linalg.norm(v)
                    rs.append(v / nv if nv else v)
                refs = np.array(rs)
                if retro:
                    for j in decided:
                        pred[j] = int((emb[j] @ refs.T).argmax())
        if refs is not None and len(refs) >= 2:
            pred[int(i)] = int((emb[i] @ refs.T).argmax())
            decided.append(int(i))
        else:
            pred[int(i)] = -1
    pairs = [(f"c{pred[i]}" if pred[i] >= 0 else "?", code[i])
             for i in range(len(code))]
    _a, m = _gtlib.best_mapping(pairs, dec.GT_CODES, unsure="?")
    ok = [m.get(p) == c for p, c in pairs]
    un = [p == "?" for p, _c in pairs]
    n = len(pairs)
    w = int(chars.sum()) or 1
    wok = int(sum(x for x, g in zip(chars, ok, strict=True) if g))
    wun = int(sum(x for x, g in zip(chars, un, strict=True) if g))
    ident = len({p for p, _c in pairs if p != "?"})
    return (sum(ok) / n, (n - sum(ok) - sum(un)) / n, sum(un) / n,
            wok / w, (w - wok - wun) / w, wun / w, ident,
            len(refs) if refs is not None else 0)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prefix", default="2026-07-20")
    args = p.parse_args(argv)
    runs = [r for r in sorted(dec.discover()) if r.startswith(args.prefix)]
    data = [(r, ue.load(r)) for r in runs]
    data = [(r, g) for r, g in data if g]
    if not data:
        raise SystemExit("# 声紋の保存が無い（_utt_embeddings.load で作る）")

    print(f"\n## 1. そもそも分かれているか（{len(data)}本）")
    print(f"{'発話長':<12}{'件数':>6}{'同じ人':>8}{'別人':>8}{'差':>8}{'等誤り率':>10}")
    agg: dict[str, list] = {}
    for _r, g in data:
        for row in separation(g["emb"], g["code"], g["dur_ms"]):
            agg.setdefault(row["band"], []).append(row)
    for band in [f"{lo / 1000:g}〜{hi / 1000:g}秒" if hi < 10**8
                 else f"{lo / 1000:g}秒以上" for lo, hi in BANDS]:
        rows = agg.get(band)
        if not rows:
            continue
        n = sum(r["n"] for r in rows)
        sa = sum(r["same"] for r in rows) / len(rows)
        di = sum(r["diff"] for r in rows) / len(rows)
        ee = sum(r["eer"] for r in rows) / len(rows)
        print(f"{band:<12}{n:>6}{sa:>8.3f}{di:>8.3f}{sa - di:>8.3f}{ee:>10.1%}")

    print(f"\n## 2. 人数を当てられるか（正解は3人・{len(data)}本）")
    print(f"{'run':<20}{'固有ギャップ':>12}{'シルエット':>12}")
    est = {"固有ギャップ": [], "シルエット": []}
    for r, g in data:
        k = estimate_k(g["emb"])
        for name, v in k.items():
            est[name].append(v)
        print(f"{r:<20}{k['固有ギャップ']:>12}{k['シルエット']:>12}")
    for name, vs in est.items():
        hit = sum(1 for v in vs if v == 3)
        print(f"{name:<20} 3人と答えた {hit}/{len(vs)}  "
              f"（平均 {sum(vs) / len(vs):.1f}）")

    print("\n## 3. クラスタだけで決めたときの成績（非因果＝上限）")
    print(f"{'条件':<24}{'件数:正解':>10}{'誤帰属':>8}"
          f"{'  ':>2}{'文字:正解':>10}{'誤帰属':>8}")
    for label, pick in (("人数を知っている（k=3）", lambda _g: 3),
                        ("シルエットの推定", lambda g: est_k_of(g)[1]),
                        ("固有ギャップの推定", lambda g: est_k_of(g)[0])):
        acc = []
        for _r, g in data:
            acc.append(cluster_score(g["emb"], g["code"], g["chars"], pick(g)))
        n = len(acc)
        a = [sum(x[i] for x in acc) / n for i in range(4)]
        print(f"{label:<24}{a[0]:>10.1%}{a[1]:>8.1%}{'  ':>2}"
              f"{a[2]:>10.1%}{a[3]:>8.1%}")

    print("\n## 4. 長い発話でまとめ、短い発話をそれに寄せる（非因果＝上限）")
    print(f"{'条件':<28}{'参照本数':>9}{'件数:正解':>10}{'誤帰属':>8}"
          f"{'  ':>2}{'文字:正解':>10}{'誤帰属':>8}")
    for min_ms in (1000, 2000, 3000):
        for label, kf in ((f"{min_ms / 1000:g}秒以上でまとめ・k=3", lambda _g: 3),
                          (f"{min_ms / 1000:g}秒以上でまとめ・k推定",
                           lambda g, m=min_ms: estimate_k_long(
                               g["emb"], g["dur_ms"], m))):
            acc = []
            for _r, g in data:
                acc.append(anchored_score(g["emb"], g["code"], g["chars"],
                                          g["dur_ms"], kf(g), min_ms=min_ms))
            n = len(acc)
            a = [sum(x[i] for x in acc) / n for i in range(4)]
            ref = sum(x[4] for x in acc) / n
            print(f"{label:<28}{ref:>9.0f}{a[0]:>10.1%}{a[1]:>8.1%}{'  ':>2}"
                  f"{a[2]:>10.1%}{a[3]:>8.1%}")

    print("\n## 5. 因果的にやった場合（ライブで実際にできる形）")
    print(f"{'条件':<30}{'件数:正解':>10}{'誤帰属':>8}{'未確定':>8}"
          f"{'  ':>2}{'文字:正解':>10}{'誤帰属':>8}{'未確定':>8}"
          f"{'  ':>2}{'話者数':>7}{'最終k':>7}")
    for label, kk, rt in (("2秒以上・k推定・貼り直しあり", None, True),
                          ("2秒以上・k推定・貼り直しなし", None, False),
                          ("2秒以上・k=3・貼り直しあり", 3, True)):
        acc = [causal_score(g["emb"], g["code"], g["chars"], g["dur_ms"],
                            g["ms"], min_ms=2000, k=kk, retro=rt)
               for _r, g in data]
        n = len(acc)
        a = [sum(x[i] for x in acc) / n for i in range(8)]
        print(f"{label:<30}{a[0]:>10.1%}{a[1]:>8.1%}{a[2]:>8.1%}{'  ':>2}"
              f"{a[3]:>10.1%}{a[4]:>8.1%}{a[5]:>8.1%}{'  ':>2}"
              f"{a[6]:>7.1f}{a[7]:>7.1f}")

    print("\n比較（§39）: 決めている 91.5% / 上限だけ外す 85.8% / 人数を使わない 67.5%"
          "（いずれも文字:正解）")
    print("注: ここの数字は会話全体を見てからまとめた非因果の上限で、ライブでは")
    print("  同じことはできない。低ければ実装を考える必要も無い、という使い方。")


_K_CACHE: dict[int, tuple[int, int]] = {}


def est_k_of(g) -> tuple[int, int]:
    """(固有ギャップ, シルエット) の推定を1回だけ計算して使い回す."""
    key = id(g)
    if key not in _K_CACHE:
        k = estimate_k(g["emb"])
        _K_CACHE[key] = (k["固有ギャップ"], k["シルエット"])
    return _K_CACHE[key]


if __name__ == "__main__":
    main()
