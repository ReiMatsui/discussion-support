"""話者別の参加度メトリクス（人間同士ファシリテーション S2）.

records から直近ウィンドウの話者別「発話時間シェア」「発話回数シェア」
「最終発話時刻」を算出する純粋関数。SessionState には依存しない。
"""
from __future__ import annotations

from collections.abc import Iterable

# 1人あたりの公平シェア比 = 1/人数。これを下回る度合いで「静か」を測る。
_DEFAULT_WINDOW_MS = 300_000  # 直近5分
_PARTICIPATION_SHARE_LABELS = {
    "time_share": "発話時間",
    "char_share": "発話文字数",
    "turn_share": "発話回数",
}


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


def participation_share_key(stats: dict[str, dict]) -> str:
    """声かけ判断で使う参加シェア指標を返す.

    STTやリプレイ入力によっては ms/end_ms が欠けるため、時間情報が全くない
    場合は文字数、文字数もない場合は発話回数を自然な代替指標として使う。
    """
    if any(d.get("talk_ms", 0.0) > 0 for d in stats.values()):
        return "time_share"
    if any(d.get("chars", 0) > 0 for d in stats.values()):
        return "char_share"
    return "turn_share"


def quietest_participation_share(stats: dict[str, dict]) -> float:
    """声かけ事前ゲートで使う最小参加シェアを返す."""
    if not stats:
        return 1.0
    key = participation_share_key(stats)
    return min(float(d.get(key, 0.0)) for d in stats.values())


def participation_share_label(key: str) -> str:
    """参加シェア指標のプロンプト表示名を返す."""
    return _PARTICIPATION_SHARE_LABELS.get(key, "発話時間")


def diarization_time_stats(events, resolve_key, *,
                           window_ms: int = _DEFAULT_WINDOW_MS,
                           now_ms: int | None = None) -> dict[str, float]:
    """分離(diarization)の閉区間から、話者キー別の発話時間を集計する（§49.17 案D）.

    帰属（名前）が未確定でも「誰がどれだけ喋ったか」は分離層が知っている。
    名前ベースの集計は未確定ぶんが消えるため、実際にはよく喋っている人を
    「静か」と誤認して的外れの声かけをしうる（§49.15のレビューで特定）。

    Args:
        events: DiarizationEvent の列（end_ms が無い開区間は除外）。
        resolve_key: event -> 表示キー。未対応クラスタは None を返せば除外。
        window_ms/now_ms: participation_stats と同じ直近窓。

    Returns: {表示キー: 発話ミリ秒}（窓内に何も無ければ空）。
    """
    rows = []
    for e in events:
        end = getattr(e, "end_ms", None)
        if end is None:
            continue
        key = resolve_key(e)
        if not key:
            continue
        rows.append((str(key), int(e.start_ms), int(end)))
    if not rows:
        return {}
    latest = now_ms if now_ms is not None else max(r[2] for r in rows)
    lo = latest - window_ms
    out: dict[str, float] = {}
    for key, s0, s1 in rows:
        s0 = max(s0, lo)
        if s1 > s0:
            out[key] = out.get(key, 0.0) + float(s1 - s0)
    return out


def apply_diarization_time(stats: dict[str, dict],
                           diar_ms: dict[str, float]) -> dict[str, dict]:
    """participation_stats の時間軸を分離計測へ置き換える（純関数・非破壊）.

    2つの効果:
      - 既存話者の talk_ms/time_share を分離計測に置き換える（未確定に
        落ちた発話の時間も本人のクラスタに乗っているため過小評価が消える）
      - records に1件も現れない（全部未確定になった）話者も、分離が検出して
        いれば参加者として現れる（turns=0, chars=0。§49.15の「3人目が不可視」
        の対策の本体）

    測定基盤の混在を避けるため、**records 側の全話者が diar_ms に載っている
    ときだけ**適用する（載っていない話者がいると、その人だけ別の物差しで
    比較することになる）。適用しない場合は stats をそのまま返す＝挙動不変。
    """
    if not diar_ms:
        return stats
    if any(sp not in diar_ms for sp in stats):
        return stats
    out = {sp: dict(d) for sp, d in stats.items()}
    for key, ms in diar_ms.items():
        d = out.setdefault(key, {"talk_ms": 0.0, "turns": 0, "chars": 0,
                                 "last_end_ms": None, "time_share": 0.0,
                                 "turn_share": 0.0, "char_share": 0.0})
        d["talk_ms"] = float(ms)
    total = sum(d["talk_ms"] for d in out.values())
    for d in out.values():
        d["time_share"] = (d["talk_ms"] / total) if total > 0 else 0.0
    return out
