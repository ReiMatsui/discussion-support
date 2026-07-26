#!/usr/bin/env python3
"""遡及訂正の「予定表」を比べる（何分ごとに貼り直すのが良いか）.

初回実装（2分・5分、以後5分ごと）は 85.2% で、上限に近い retro_5m（89.5%）に
届かなかった。原因は間隔で、10分の会話だと 300秒→600秒の間に一度も貼り直しが
入らない区間ができる。

**貼り直しは実質ただである**: `RetroAttributor.revise` は保存済みの声紋
（192次元）と席の参照の内積を取るだけで、埋め込みの計算は一切しない。
1200発話×3席でも内積3600回。したがって間隔を詰めない理由は計算量ではなく、
**表示が頻繁に書き換わること**（UX）だけになる。

ここでは席の参照の推移を1回だけ計算して保存し、予定表だけを差し替えて比べる
（埋め込みの計算をやり直さないので、条件を増やしても時間が増えない）。

使い方:
    uv run python eval/retro_schedule.py --prefix 2026-07-20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _gtlib  # noqa: E402
import cluster_merge_feasibility as feas  # noqa: E402
import decompose_attribution as dec  # noqa: E402

from das.asr.live._attribution import _VOICEPRINT_RELIABLE_KINDS  # noqa: E402
from das.asr.live._constants import (  # noqa: E402
    SEAT_AUDIO_MIN_REF_SEC,
    SR,
    UNSURE_SPEAKER,
)
from das.asr.live._recv_loop import _LABEL_ONLY_KINDS  # noqa: E402
from das.asr.live._seat_audio import SeatAudio  # noqa: E402

# (名前, 予定表, 以後の間隔秒)。間隔 0 は「発話ごと」
CONFIGS = (
    ("2分5分/5分ごと（初回実装）", (120.0, 300.0), 300.0),
    ("2分5分/2分ごと", (120.0, 300.0), 120.0),
    ("2分5分/1分ごと", (120.0, 300.0), 60.0),
    ("1分ごと", (60.0,), 60.0),
    ("発話ごと", (0.0,), 0.0),
)


def collect(run: str, vp) -> dict | None:
    """席の参照の推移と、貼り直せる発話の声紋を1回だけ計算する.

    席の参照は「高信頼で確定した発話」だけから作られ、その集合は予定表に
    依存しない。したがって推移を保存しておけば、どの予定表でも再利用できる。
    """
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
    t0 = int(rows[0][0]["ms"])
    seat = SeatAudio(vp)
    steps = []      # 発話ごとの (経過秒, GT, 可変か, 声紋, その時点の参照)
    for u, code in rows:
        a, b = int(u["ms"]), int(u.get("end") or u["ms"])
        wav = pcm[int(a / 1000 * SR):int(b / 1000 * SR)]
        cur = str(u["final_key"])
        kind = u.get("kind")
        if cur != UNSURE_SPEAKER and kind == "蓄積中" and not dec.endorsed(u):
            cur = UNSURE_SPEAKER
        revisable = (kind in _LABEL_ONLY_KINDS
                     or (cur == UNSURE_SPEAKER
                         and str(u.get("key")) != UNSURE_SPEAKER))
        emb = seat.embed(wav) if revisable else None
        if not revisable and cur != UNSURE_SPEAKER \
                and kind in _VOICEPRINT_RELIABLE_KINDS:
            seat.observe(cur, wav)
        refs = {k: v for k, v in seat._embeddings.items()
                if seat._seconds.get(k, 0.0) >= SEAT_AUDIO_MIN_REF_SEC}
        steps.append({"elapsed": (a - t0) / 1000.0, "code": code,
                      "base": cur, "revisable": revisable, "emb": emb,
                      "refs": dict(refs)})
    return {"run": run, "steps": steps}


def _pick(emb, refs):
    if emb is None or len(refs) < 2:
        return None
    return max(((float(np.dot(emb, v)), k) for k, v in refs.items()))[1]


def evaluate(data, schedule, interval) -> tuple[float, float, float]:
    steps = data["steps"]
    final = []
    remembered = []          # (index, emb)
    idx = 0
    next_at = schedule[0] if schedule else interval
    for i, st in enumerate(steps):
        cur = st["base"]
        if st["revisable"]:
            got = _pick(st["emb"], st["refs"])
            cur = got if got is not None else cur
            remembered.append(i)
        final.append(cur)
        if st["elapsed"] >= next_at:
            idx += 1
            next_at = (schedule[idx] if idx < len(schedule)
                       else st["elapsed"] + interval)
            for j in remembered:
                got = _pick(steps[j]["emb"], st["refs"])
                if got is not None:
                    final[j] = got
    pairs = [(final[i], steps[i]["code"]) for i in range(len(steps))]
    _a, m = _gtlib.best_mapping(pairs, dec.GT_CODES, unsure=UNSURE_SPEAKER)
    n = len(pairs)
    good = sum(1 for f, c in pairs if m.get(f) == c)
    uns = sum(1 for f, _ in pairs if f == UNSURE_SPEAKER)
    return good / n, (n - good - uns) / n, uns / n


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prefix", default="2026-07-20")
    p.add_argument("--model", default="redimnet")
    p.add_argument("--split", type=int, default=5)
    args = p.parse_args(argv)
    from das.asr.live._voice_profiles import VoiceProfiles
    vp = VoiceProfiles(model=args.model)
    runs = [r for r in sorted(dec.discover()) if r.startswith(args.prefix)]
    data = [d for d in (collect(x, vp) for x in runs) if d]
    if not data:
        raise SystemExit("# 測れるランが無い")

    def _report(subset, label):
        print(f"\n## {label}（{len(subset)}本）")
        print(f"{'予定表':<26}{'正解':>8}{'誤帰属':>9}{'未確定':>9}{'貼直し回数':>10}")
        for name, sched, interval in CONFIGS:
            vals = [evaluate(d, sched, interval) for d in subset]
            n = len(vals)
            acc, wrong, uns = (sum(v[i] for v in vals) / n for i in (0, 1, 2))
            # 貼り直しの回数（10分の会話での目安）
            times = 1 if interval <= 0 else int(600 / max(interval, 1))
            shown = "毎発話" if interval <= 0 else str(len(sched) + times)
            print(f"{name:<26}{acc:>8.1%}{wrong:>9.1%}{uns:>9.1%}{shown:>10}")

    _report(data, "全体")
    if 0 < args.split < len(data):
        _report(data[:args.split], "開発")
        _report(data[args.split:], "検証")
    print("\n読み方:")
    print("  貼り直しは保存済みの声紋との内積だけで、埋め込みの計算は要らない。")
    print("  したがって間隔を詰めない理由は計算量ではなく、表示が頻繁に")
    print("  書き換わること（UX）だけになる。")


if __name__ == "__main__":
    main()
