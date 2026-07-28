#!/usr/bin/env python3
"""埋め込みモデルを替えると席の割当てがどれだけ当たるようになるかを測る.

いまの声紋は ReDimNet **b2**（`_voice_profiles.py` にモデル名が直書き）。
同じ配布元に b3 / b5 / b6 があり、`model_name` を替えるだけで載る。

**なぜ席の割当てだけを測るのか**: §27.12 以降、残る誤帰属の過半は
「席の音声と比べて寄せ先を選ぶ」段の誤りである。そしてこの段は**類似度の
しきい値を使わない**（席を持つ人の中で1位を選ぶだけ）ので、モデルを替えても
再校正が要らない。上流の声紋層（`classify` の閾値・自動登録・合流）は b2 で
校正されているため、そちらは記録のまま固定して比べる——モデル差だけを見る。

したがってここで出るのは「モデルを替えたときに**すぐ得られる**改善」であり、
上流も含めて全部を替えた場合の上限ではない（そちらは再校正が要る）。

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

import _pipeline as pipe  # noqa: E402
import cluster_merge_feasibility as feas  # noqa: E402
import decompose_attribution as dec  # noqa: E402

from das.asr.live._attribution import _VOICEPRINT_RELIABLE_KINDS  # noqa: E402
from das.asr.live._constants import SR, UNSURE_SPEAKER  # noqa: E402
from das.asr.live._recv_loop import _LABEL_ONLY_KINDS  # noqa: E402
from das.asr.live._seat_audio import SeatAudio  # noqa: E402


class _Encoder:
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


def run_one(run: str, enc) -> tuple[float, float, float] | None:
    """本番と同じ順序・同じ条件で1ランを流し、(正解, 誤帰属, 未確定) を返す."""
    loaded = dec.load_run(run)
    wav_path = ROOT / "transcripts" / f"{run}.wav"
    if loaded is None or not wav_path.exists():
        return None
    utts, code_by_ms = loaded
    rows = [(u, code_by_ms.get(u["ms"])) for u in utts]
    rows = [(u, c) for u, c in rows if c in dec.GT_CODES
            and not dec._BACKCHANNEL_RE.match(str(u["_text"]).strip())]
    if not rows or not any(u.get("final_key") is not None for u, _ in rows):
        return None
    rows.sort(key=lambda r: int(r[0]["ms"]))
    pcm = feas.read_wav(wav_path)
    seat = SeatAudio(enc)
    pick: dict[int, str] = {}
    for u, _c in rows:
        a, b = int(u["ms"]), int(u.get("end") or u["ms"])
        wav = pcm[int(a / 1000 * SR):int(b / 1000 * SR)]
        final = str(u["final_key"])
        kind = u.get("kind")
        # _recv_loop.flush と同じ順序（門番→上書き/参照育成→席落ちの拾い直し）
        if (final != UNSURE_SPEAKER and kind == "蓄積中"
                and not dec.endorsed(u)):
            final = UNSURE_SPEAKER
        if kind in _LABEL_ONLY_KINDS or (final == UNSURE_SPEAKER
                                         and str(u.get("key")) != UNSURE_SPEAKER):
            got = seat.nearest(wav)
            if got is not None:
                pick[int(u["ms"])] = got[0]
        elif final != UNSURE_SPEAKER and kind in _VOICEPRINT_RELIABLE_KINDS:
            seat.observe(final, wav)

    def _final(u):
        # 規則は eval/_pipeline.resolved_key に一本化（書き写すとずれる）
        return pipe.resolved_key(u, pick.get(int(u["ms"])))

    pairs = [(_final(u), c) for u, c in rows]
    return pipe.score(pairs)[:3]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--models", default="b2,b3")
    p.add_argument("--prefix", default=None)
    p.add_argument("--split", type=int, default=5)
    args = p.parse_args(argv)
    runs = [r for r in sorted(dec.discover())
            if not args.prefix or r.startswith(args.prefix)]
    names = [x.strip() for x in args.models.split(",") if x.strip()]
    print("# 埋め込みモデル別: 席の割当てだけを差し替えた成績（上流は記録のまま）")
    print(f"{'model':<8}{'全体 正解':>10}{'誤帰属':>8}{'未確定':>8}"
          f"{'開発':>8}{'検証':>8}{'埋込ms':>8}")
    for name in names:
        try:
            enc = _Encoder(name)
        except Exception as e:
            print(f"{name:<8}読み込み失敗: {type(e).__name__}: {e}")
            continue
        vals = [v for v in (run_one(x, enc) for x in runs) if v]
        if not vals:
            continue
        n = len(vals)
        acc, wrong, uns = (sum(v[i] for v in vals) / n for i in (0, 1, 2))
        dev = (sum(v[0] for v in vals[:args.split]) / args.split
               if 0 < args.split < n else float("nan"))
        val = (sum(v[0] for v in vals[args.split:]) / (n - args.split)
               if 0 < args.split < n else float("nan"))
        ms = enc.seconds / enc.calls * 1000 if enc.calls else float("nan")
        print(f"{name:<8}{acc:>10.1%}{wrong:>8.1%}{uns:>8.1%}"
              f"{dev:>8.1%}{val:>8.1%}{ms:>8.0f}")
    print("\n読み方:")
    print("  席の割当ては類似度のしきい値を使わない（1位を選ぶだけ）ので、")
    print("  モデルを替えても再校正なしでこの差がそのまま得られる。上流の")
    print("  声紋層は b2 で校正済みのため記録のまま固定してある。")
    print("  埋込ms は1回あたりの平均。ライブは発話ごとに呼ぶので遅延に直結する。")


if __name__ == "__main__":
    main()
