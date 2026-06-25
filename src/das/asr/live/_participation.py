"""話者別の参加度メトリクス（人間同士ファシリテーション S2）.

records から直近ウィンドウの話者別「発話時間シェア」「発話回数シェア」
「最終発話時刻」を算出する純粋関数。SessionState には依存しない。
"""
from __future__ import annotations

from collections.abc import Iterable

# 1人あたりの公平シェア比 = 1/人数。これを下回る度合いで「静か」を測る。
_DEFAULT_WINDOW_MS = 300_000  # 直近5分


def participation_stats(records: list[dict], *,
                        window_ms: int = _DEFAULT_WINDOW_MS,
                        exclude_speakers: Iterable[str] = ()) -> dict[str, dict]:
    """直近ウィンドウの話者別参加度を返す.

    Args:
        records: 発話記録（speaker/text/ms/end_ms を持つ dict のリスト）。
        window_ms: 直近何ミリ秒を見るか（全期間でなく窓で見る）。
        exclude_speakers: 除外する話者キー（ファシリテーター等）。

    Returns:
        {speaker: {"talk_ms": float, "turns": int, "chars": int,
                   "time_share": float, "turn_share": float, "char_share": float,
                   "last_end_ms": int | None}}
        発話が無ければ空 dict。ms/end_ms が無い発話は時間計算から除外（回数は加算）。
    """
    exclude = set(exclude_speakers)
    rows = [r for r in records
            if "speaker" in r and r.get("text")
            and r.get("speaker") not in exclude]
    if not rows:
        return {}

    # 現在時刻の基準 = 最新のタイムスタンプ
    times = [r["end_ms"] if r.get("end_ms") is not None else r.get("ms")
             for r in rows]
    times = [t for t in times if t is not None]
    now_ms = max(times) if times else None

    if now_ms is not None:
        win_rows = [r for r in rows
                    if r.get("ms") is None or r["ms"] >= now_ms - window_ms]
    else:
        win_rows = rows

    stats: dict[str, dict] = {}
    for r in win_rows:
        sp = r["speaker"]
        d = stats.setdefault(sp, {"talk_ms": 0.0, "turns": 0, "chars": 0,
                                  "last_end_ms": None})
        ms, end = r.get("ms"), r.get("end_ms")
        if ms is not None and end is not None and end > ms:
            d["talk_ms"] += float(end - ms)
        d["turns"] += 1
        d["chars"] += len(r.get("text", ""))
        t = end if end is not None else ms
        if t is not None and (d["last_end_ms"] is None or t > d["last_end_ms"]):
            d["last_end_ms"] = t

    total_ms = sum(d["talk_ms"] for d in stats.values())
    total_turns = sum(d["turns"] for d in stats.values())
    total_chars = sum(d["chars"] for d in stats.values())
    for d in stats.values():
        d["time_share"] = (d["talk_ms"] / total_ms) if total_ms > 0 else 0.0
        d["turn_share"] = (d["turns"] / total_turns) if total_turns > 0 else 0.0
        d["char_share"] = (d["chars"] / total_chars) if total_chars > 0 else 0.0
    return stats
