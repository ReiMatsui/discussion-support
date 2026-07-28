#!/usr/bin/env python3
"""現行の帰属と Streaming Sortformer を、同じ音源・同じ採点規則で比べ直す.

**なぜ測り直すのか**: 2026-07-22 の検証（`sortformer_feasibility_2026-07-22.md`）
は「土台交換は否」と結論したが、その表にはこういう行もあった:

    千葉1723（クリーン音源）  現行 39.6%  /  Sortformer v2.1 88.1%

クリーンな素材では現行が大敗していた。ところが §27-§28 の作業で現行は
同じ Chiba で 89.9% になった。**Sortformer が勝っていた領域で並んだ可能性が
高いのに、誰も測り直していない**。長く開いている「土台交換」の問いに、
現在の数字で決着をつける。

採点規則は 2026-07-22 と揃える（比較可能にするため）:

  - 採点単位は GT 付き発話（`gt_<run>.json` の S1/S2/S3）
  - Sortformer 側の答えは、その発話区間と最も重なった話者（重なりゼロは未確定）
  - `best_mapping` で最適1:1対応を取り、**未確定は常に不正解**
  - 相槌は現行側の測定と同じく除外する（分母を揃える）

現行側は diag の `final_key` に遡及訂正を適用した値を使う（＝いま出荷して
いる挙動）。API コストは0（Sortformer はローカル推論）。

使い方:
    # 1. 推論（NeMo venv が要る）
    /root/nemo-venv/bin/python eval/sortformer_infer.py --prefix 2026-07-20
    # 2. 採点
    uv run python eval/sortformer_compare.py --prefix 2026-07-20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _pipeline as pipe  # noqa: E402
import decompose_attribution as dec  # noqa: E402

from das.asr.live._constants import UNSURE_SPEAKER  # noqa: E402

SEG_DIR = ROOT / "eval" / "_sortformer"


def load_segments(run: str) -> list[tuple[int, int, str]] | None:
    """`sortformer_infer.py` が書いた (開始ms, 終了ms, 話者) を読む."""
    p = SEG_DIR / f"{run}.json"
    if not p.exists():
        return None
    return [(int(a), int(b), str(s))
            for a, b, s in json.loads(p.read_text(encoding="utf-8"))]


def dominant(segs, a: int, b: int) -> str:
    """区間 [a,b) と最も重なった話者（重なりゼロなら未確定）."""
    best, best_ov = UNSURE_SPEAKER, 0
    for x, y, sp in segs:
        ov = min(b, y) - max(a, x)
        if ov > best_ov:
            best, best_ov = sp, ov
    return best


def score(pairs) -> tuple[float, float, float]:
    return pipe.score(pairs)[:3]


def current_keys(run: str, vp) -> dict[int, str] | None:
    """**今日の実装**で決まる最終キー（再現は `_pipeline` に一本化）.

    diag の `final_key` は記録時（2026-07-20）の判定で、§27-§28 の改善が
    入っていない。それをそのまま「現行」として比べると 2026-07-22 と同じ
    古い比較を繰り返すことになる——実際にそれで 54.4% 対 87.7% という
    誤った結論を出した。
    """
    return pipe.current_keys(run, vp)


def compare(run: str, cur_by_ms: dict | None = None) -> dict | None:
    loaded = dec.load_run(run)
    segs = load_segments(run)
    if loaded is None or segs is None:
        return None
    utts, code_by_ms = loaded
    rows = [(u, code_by_ms.get(u["ms"])) for u in utts]
    rows = [(u, c) for u, c in rows if c in dec.GT_CODES
            and not dec._BACKCHANNEL_RE.match(str(u["_text"]).strip())]
    if not rows:
        return None
    cur = [((cur_by_ms or {}).get(int(u["ms"]), str(u["final_key"])), c)
           for u, c in rows]
    sf = [(dominant(segs, int(u["ms"]), int(u.get("end") or u["ms"])), c)
          for u, c in rows]
    n_spk = len({s for _, _, s in segs})
    return {"run": run, "n": len(rows), "cur": score(cur), "sf": score(sf),
            "sf_speakers": n_spk}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prefix", default="2026-07-20")
    p.add_argument("--stale", action="store_true",
                   help="記録時の final_key をそのまま現行として使う"
                        "（2026-07-22 と同じ古い比較。検証用）")
    args = p.parse_args(argv)
    runs = [r for r in sorted(dec.discover()) if r.startswith(args.prefix)]
    vp = None
    if not args.stale:
        from das.asr.live._voice_profiles import VoiceProfiles
        vp = VoiceProfiles(model="redimnet")
    out = []
    for x in runs:
        cur = None if args.stale else current_keys(x, vp)
        r = compare(x, cur)
        if r:
            out.append(r)
    if not out:
        raise SystemExit(f"# 推論結果が無い。先に sortformer_infer.py を実行"
                         f"（{SEG_DIR} に .json ができる）")
    print("# 同じ音源・同じ採点規則での比較（未確定は不正解として扱わず別掲）")
    print(f"{'run':<20}{'n':>5}{'現行 正解':>10}{'誤':>7}{'未':>7}"
          f"{'SF 正解':>10}{'誤':>7}{'未':>7}{'SF話者数':>9}")
    for r in out:
        c, s = r["cur"], r["sf"]
        print(f"{r['run']:<20}{r['n']:>5}{c[0]:>10.1%}{c[1]:>7.1%}{c[2]:>7.1%}"
              f"{s[0]:>10.1%}{s[1]:>7.1%}{s[2]:>7.1%}{r['sf_speakers']:>9}")
    n = len(out)
    cm = [sum(r["cur"][i] for r in out) / n for i in (0, 1, 2)]
    sm = [sum(r["sf"][i] for r in out) / n for i in (0, 1, 2)]
    print(f"{'平均':<20}{'':>5}{cm[0]:>10.1%}{cm[1]:>7.1%}{cm[2]:>7.1%}"
          f"{sm[0]:>10.1%}{sm[1]:>7.1%}{sm[2]:>7.1%}")
    print(f"\n  差: 正解 {(sm[0] - cm[0]) * 100:+.1f}pt（正なら Sortformer 有利）")
    print("  2026-07-22 の時点ではクリーン音源で 現行 39.6% / SF 88.1% と")
    print("  大差がついていた。§27-§28 の作業でその差が埋まったかを見る。")
    print("  注: マイク経由の会議音声では SF は 42%（現行79%）と大敗しており、")
    print("  ここはクリーン音源での比較である（本番ドメインとは別）。")


if __name__ == "__main__":
    main()
