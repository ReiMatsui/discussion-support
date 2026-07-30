"""人数を教えられていないとき、長い発話から参加者の集合を自分で作る.

**なぜ要るのか**（handoff §39-§41）: 席の割当ては「席を持つN人のうち誰か」
という閉集合の問題を解いているだけで、その N は `--diarization-max-speakers`
から来ている。人数が入力されないと席ルールが素通しになり、文字ベースの正解が
91.5% → 67.5% まで落ちる。

そこで N を貰う代わりに**自分で作る**。作り方は席と同じ形——短い発話どうしを
比べるのではなく、長い音声から作った参照と比べる。違いは参照の出どころだけで、
席は人数の設定から、こちらは**長い発話のまとまり**から作る。実測で
文字ベース 67.5% → 86.4%（校正セット）/ 72.2% → 82.0%（持ち越し）。

**なぜ長い発話だけを材料にするのか**: 声紋の分離は長さに強く依存する。
実測（等誤り率）は 0.5秒未満で46.6%＝ほぼ偶然、2秒以上で6.1%。短い発話を
材料に混ぜると、まとまりそのものが壊れる。

**なぜ scipy を使わないのか**: 材料は会話1本あたり60件程度で、平均連結法も
シルエットもこの規模なら numpy で足りる。本番の依存を増やしたくない
（eval 側は scipy を使っており、そちらとの一致はテストで固定してある）。
"""
from __future__ import annotations

import numpy as np


def average_linkage(sim: np.ndarray, k: int) -> np.ndarray:
    """平均連結法で k 個にまとめ、点ごとの群番号（0始まり）を返す.

    `sim` は対称な類似度行列（L2正規化済み声紋の内積）。距離ではなく類似度で
    受けるのは、呼び出し側が既に内積を持っているため。

    毎回いちばん似ている2群を繋ぐ。群どうしの類似度は**構成点の平均**で、
    融合したら関わる行と列を重みつき平均で畳む（Lance-Williams の更新式を
    平均連結に当てた形）。点数が60程度なら素朴な O(n^3) で足りる。
    """
    n = len(sim)
    if k >= n:
        return np.arange(n)
    # groups[i] は「その行がいま代表している点の集合」。cur は生きている行。
    groups: list[list[int]] = [[i] for i in range(n)]
    s = sim.astype(np.float64).copy()
    np.fill_diagonal(s, -np.inf)
    alive = list(range(n))
    while len(alive) > k:
        sub = s[np.ix_(alive, alive)]
        flat = int(np.argmax(sub))
        ai, bi = divmod(flat, len(alive))
        a, b = alive[ai], alive[bi]
        wa, wb = len(groups[a]), len(groups[b])
        # a に b を畳む（平均連結: 重みつき平均）
        s[a, :] = (s[a, :] * wa + s[b, :] * wb) / (wa + wb)
        s[:, a] = s[a, :]
        s[a, a] = -np.inf
        groups[a] = groups[a] + groups[b]
        alive.remove(b)
    lab = np.empty(n, dtype=int)
    for new, old in enumerate(alive):
        for i in groups[old]:
            lab[i] = new
    return lab


def silhouette(sim: np.ndarray, lab: np.ndarray) -> float:
    """まとまりの良さ（-1〜1）。自分の群への近さと、最も近い他群への近さの差.

    距離は 1 - 類似度。群が1つしか無いときは -1（＝最悪）を返す——「全部
    同じ人」を選ばせないため。
    """
    d = 1.0 - sim
    np.fill_diagonal(d, 0.0)
    labs = np.unique(lab)
    if len(labs) < 2:
        return -1.0
    vals = []
    for i in range(len(lab)):
        own = lab == lab[i]
        own[i] = False
        if not own.any():
            continue          # 1点だけの群は評価に入れない
        a = float(d[i][own].mean())
        b = min(float(d[i][lab == c].mean()) for c in labs if c != lab[i])
        if max(a, b) > 0:
            vals.append((b - a) / max(a, b))
    return float(np.mean(vals)) if vals else -1.0


def cluster(embeddings: np.ndarray, *, max_k: int = 8) -> np.ndarray:
    """話者数を選んでまとめ、点ごとの群番号を返す.

    人数は**推定させる**。上限として与えても成績は落ちる（実測: 上限3で
    文字 86.4% → 76.7%）。理由は、割りすぎは安く混ぜるのは高いこと——一人の
    声が2つに割れても片方が本人に対応づくが、材料の少ない序盤に別人どうしを
    まとめると抜け出せない（handoff §41.1）。
    """
    n = len(embeddings)
    if n < 2:
        return np.zeros(n, dtype=int)
    sim = embeddings @ embeddings.T
    best, best_lab = -2.0, np.zeros(n, dtype=int)
    for k in range(2, min(max_k, n) + 1):
        lab = average_linkage(sim, k)
        score = silhouette(sim, lab)
        if score > best:
            best, best_lab = score, lab
    return best_lab


def centroids(embeddings: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """群ごとの重心（L2正規化済み）を群番号の順に返す."""
    out = []
    for c in np.unique(lab):
        v = embeddings[lab == c].mean(0)
        norm = float(np.linalg.norm(v))
        out.append(v / norm if norm else v)
    return np.array(out) if out else np.zeros((0, embeddings.shape[1]))
