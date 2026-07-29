#!/usr/bin/env python3
"""埋め込みモデルを替えると席の割当てがどれだけ当たるようになるかを測る.

いまの声紋は ReDimNet **b2**（`_voice_profiles._load_embedder` に直書き）。
同じ配布元に b3 / b5 / b6 があり、`model_name` を替えるだけで載る。

**なぜいま測り直すのか**: 一度測って「効果なし」と却下したが、それは §34 以前
の測り方——時間の重なりで正解を当て、**短い発話を分母から落とす**測り方——で
の判定だった。残る誤りは 0.5〜1秒の音声から誰かを当てる問題に集約されている
（§35: 誤帰属の81%が1秒未満）ので、埋め込みの質が効くとすればまさにそこ。
落としていた層で判定していたことになるので、判定自体が無効である。

**なぜ席の割当てだけを測るのか**: この段は**類似度のしきい値を使わない**
（席を持つ人の中で1位を選ぶだけ）ので、モデルを替えても再校正が要らない。
上流の声紋層（`classify` の閾値・自動登録・合流）は b2 で校正されているため、
記録のまま固定して比べる——モデル差だけを見る。したがってここで出るのは
「替えたときに**すぐ得られる**改善」であり、全部を替えた場合の上限ではない。

遅延も測る。b5/b6 は b2 より重く、ライブでは1発話ごとに埋め込みを計算する
ので、精度が上がっても遅延が許容外なら採れない。

使い方:
    uv run python eval/embedding_model_compare.py --models b2,b3 --prefix 2026-07-20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _gtlib  # noqa: E402
import _pipeline as pipe  # noqa: E402
import decompose_attribution as dec  # noqa: E402

from das.asr.live._constants import UNSURE_SPEAKER  # noqa: E402

SHORT_MS = 1000     # 効くとすればここ、という帯（§35）


class Encoder:
    """ReDimNet を任意のサイズで読み、`embed_audio` だけを提供する薄い殻.

    `SeatAudio` が使うのは `embed_audio` のみなので、本番の `VoiceProfiles`
    を丸ごと差し替えずにモデルだけ入れ替えられる。正規化は本番の `_embed` と
    同じ（L2正規化、非有限は None）。
    """

    def __init__(self, name: str) -> None:
        import torch
        self.name = name
        self._torch = torch
        self._enc = torch.hub.load("IDRnD/ReDimNet", "ReDimNet",
                                   model_name=name, train_type="ft_lm",
                                   dataset="vox2", trust_repo=True)
        self._enc.eval()
        self.calls = 0
        self.seconds = 0.0

    def embed_audio(self, wav):
        if wav is None or wav.size == 0:
            return None
        t0 = time.perf_counter()
        with self._torch.no_grad():
            v = self._enc(
                self._torch.from_numpy(np.ascontiguousarray(wav)).float()
                .unsqueeze(0)).squeeze().numpy()
        self.calls += 1
        self.seconds += time.perf_counter() - t0
        v = np.asarray(v, dtype=np.float64)
        n = float(np.linalg.norm(v))
        if n == 0.0 or not np.isfinite(n):
            return None
        return v / n


def outcomes(run: str, enc) -> list[dict] | None:
    """1ランを今日の規則で流し、発話ごとの結末を返す（再現は `_pipeline`）."""
    data = pipe.replay_seats(run, enc, align="text")
    if data is None:
        return None
    steps = data["steps"]
    final = pipe.apply_schedule(steps)
    pairs = [(f, st["code"]) for f, st in zip(final, steps, strict=True)]
    _a, m = _gtlib.best_mapping(pairs, dec.GT_CODES, unsure=UNSURE_SPEAKER)
    out = []
    for f, st in zip(final, steps, strict=True):
        u = st["utt"]
        out.append({
            "dur_ms": max(0, int(u.get("end") or u["ms"]) - int(u["ms"])),
            "chars": len(str(u.get("_text") or "")),
            "outcome": ("未確定" if f == UNSURE_SPEAKER
                        else "正解" if m.get(f) == st["code"] else "誤帰属")})
    return out


def _rates(rows, weigh):
    tot = sum(weigh(r) for r in rows) or 1
    return tuple(sum(weigh(r) for r in rows if r["outcome"] == k) / tot
                 for k in ("正解", "誤帰属", "未確定"))


def _line(label: str, rows: list[dict], ms: float | None = None) -> None:
    c = _rates(rows, lambda r: 1)
    w = _rates(rows, lambda r: r["chars"])
    tail = f"{ms:>9.0f}" if ms is not None else f"{'':>9}"
    print(f"{label:<10}{len(rows):>6}{c[0]:>8.1%}{c[1]:>8.1%}{c[2]:>8.1%}"
          f"{'  ':>2}{w[0]:>8.1%}{w[1]:>8.1%}{w[2]:>8.1%}{tail}")


def _header(title: str) -> None:
    print(f"\n## {title}")
    print(f"{'model':<10}{'件数':>6}{'正解':>8}{'誤帰属':>8}{'未確定':>8}"
          f"{'  ':>2}{'文字:正解':>8}{'誤帰属':>8}{'未確定':>8}{'埋込ms':>9}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--models", default="b2,b3")
    p.add_argument("--prefix", default="2026-07-20")
    p.add_argument("--split", type=int, default=5,
                   help="開発/検証に分ける本数（0で分けない）")
    args = p.parse_args(argv)
    runs = [r for r in sorted(dec.discover()) if r.startswith(args.prefix)]
    names = [x.strip() for x in args.models.split(",") if x.strip()]

    got: dict[str, list[list[dict]]] = {}
    ms: dict[str, float] = {}
    for name in names:
        try:
            enc = Encoder(name)
        except Exception as e:
            print(f"# {name} 読み込み失敗: {type(e).__name__}: {e}", flush=True)
            continue
        per_run = [r for r in (outcomes(x, enc) for x in runs) if r]
        if not per_run:
            continue
        got[name] = per_run
        ms[name] = enc.seconds / enc.calls * 1000 if enc.calls else float("nan")
        print(f"# {name} 済み（{sum(len(r) for r in per_run)}発話・"
              f"埋込 {ms[name]:.0f}ms）", flush=True)
    if not got:
        raise SystemExit("# 測れるモデルが無い")

    _header(f"全体（{len(runs)}本）")
    for name, per_run in got.items():
        _line(name, [x for r in per_run for x in r], ms[name])

    _header("1秒未満だけ（効くとすればここ）")
    for name, per_run in got.items():
        _line(name, [x for r in per_run for x in r if x["dur_ms"] < SHORT_MS])

    _header("1秒以上だけ（壊していないかの確認）")
    for name, per_run in got.items():
        _line(name, [x for r in per_run for x in r if x["dur_ms"] >= SHORT_MS])

    if 0 < args.split < len(runs):
        _header(f"開発（{args.split}本）")
        for name, per_run in got.items():
            _line(name, [x for r in per_run[:args.split] for x in r])
        _header(f"検証（{len(runs) - args.split}本）")
        for name, per_run in got.items():
            _line(name, [x for r in per_run[args.split:] for x in r])

    print("\n読み方:")
    print("  席の割当ては類似度のしきい値を使わない（1位を選ぶだけ）ので、")
    print("  モデルを替えても再校正なしでこの差がそのまま得られる。上流の")
    print("  声紋層は b2 で校正済みのため記録のまま固定してある。")
    print("  埋込ms は1回あたりの平均。ライブは発話ごとに呼ぶので遅延に直結する。")


if __name__ == "__main__":
    main()
