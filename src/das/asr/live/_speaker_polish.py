"""録音後に発話全体を見直して話者ラベルを安定化する."""
from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from ._constants import AGENT_SPEAKER, SR, UNSURE_SPEAKER


def relabel_records_by_embeddings(
    records: list[dict[str, Any]],
    embeddings: dict[int, np.ndarray],
    *,
    max_speakers: int | None = None,
    distance_threshold: float = 0.50,
) -> list[dict[str, Any]]:
    """発話声紋をクラスタリングし、recordsのspeakerを後処理で安定化する."""
    if len(embeddings) < 2:
        return copy.deepcopy(records)

    idxs = sorted(embeddings)
    x = np.stack([embeddings[i] for i in idxs])
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    if max_speakers is not None and 1 < max_speakers <= len(idxs):
        labels = _cluster_cosine_average(x, n_clusters=max_speakers)
    else:
        labels = _cluster_cosine_average(x, distance_threshold=distance_threshold)
    cluster_of = dict(zip(idxs, labels, strict=True))
    speaker_for_cluster = _speaker_names(records, cluster_of)

    out = copy.deepcopy(records)
    for i, rec in enumerate(out):
        cluster = cluster_of.get(i)
        if cluster is None:
            continue
        rec["speaker_before_polish"] = rec.get("speaker")
        rec["speaker"] = speaker_for_cluster[cluster]
        rec["speaker_source"] = "speaker_polish"
    _smooth_short_between_same_speaker(out)
    return out


def polish_speakers_from_wav(
    records: list[dict[str, Any]],
    wav_path: str,
    tracker,
    *,
    max_speakers: int | None = None,
) -> list[dict[str, Any]]:
    """保存済みwavから各発話を再埋め込みし、クラスタリングで話者を後処理する."""
    pcm = _read_pcm16_wav(wav_path)
    embeddings: dict[int, np.ndarray] = {}
    for i, rec in enumerate(records):
        if not _usable_record(rec):
            continue
        start = max(int(rec["ms"] or 0), 0)
        end = max(int(rec["end_ms"] or 0), start)
        seg = pcm[start * 16:end * 16].astype(np.float32) / 32768.0
        if seg.size < int(SR * 0.8):
            continue
        emb = tracker._embed(seg)  # 既存の声紋モデルを再利用する。
        if emb is not None:
            embeddings[i] = emb
    return relabel_records_by_embeddings(
        records,
        embeddings,
        max_speakers=max_speakers,
    )


def _read_pcm16_wav(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()
    return np.frombuffer(data[44:], dtype="<i2")


def _cluster_cosine_average(
    x: np.ndarray,
    *,
    n_clusters: int | None = None,
    distance_threshold: float | None = None,
) -> list[int]:
    clusters: list[list[int]] = [[i] for i in range(len(x))]
    while len(clusters) > 1:
        best: tuple[float, int, int] | None = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = _avg_cosine_distance(x, clusters[i], clusters[j])
                if best is None or distance < best[0]:
                    best = (distance, i, j)
        if best is None:
            break
        distance, i, j = best
        if n_clusters is not None and len(clusters) <= n_clusters:
            break
        if n_clusters is None and distance_threshold is not None and distance > distance_threshold:
            break
        clusters[i].extend(clusters[j])
        del clusters[j]
    labels = [0] * len(x)
    for label, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = label
    return labels


def _avg_cosine_distance(x: np.ndarray, left: list[int], right: list[int]) -> float:
    sims = [float(np.dot(x[i], x[j])) for i in left for j in right]
    return 1.0 - (sum(sims) / len(sims))


def _usable_record(rec: dict[str, Any]) -> bool:
    speaker = rec.get("speaker")
    if speaker in {None, AGENT_SPEAKER, "パートナー", UNSURE_SPEAKER}:
        return False
    if rec.get("bc"):
        return False
    if not rec.get("text") or rec.get("ms") is None or rec.get("end_ms") is None:
        return False
    return int(rec["end_ms"]) - int(rec["ms"]) >= 800


def _speaker_names(records: list[dict[str, Any]], cluster_of: dict[int, int]) -> dict[int, str]:
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    first_ms: dict[int, int] = {}
    for i, cluster in cluster_of.items():
        rec = records[i]
        speaker = str(rec.get("speaker", ""))
        if speaker and not speaker.startswith(("#", "@")) and speaker != UNSURE_SPEAKER:
            counts[cluster][speaker] += max(
                int(rec.get("end_ms") or 0) - int(rec.get("ms") or 0),
                1,
            )
        first_ms.setdefault(cluster, int(rec.get("ms") or 0))

    ordered = sorted(first_ms, key=lambda c: first_ms[c])
    fallback = {cluster: f"話者{n + 1}" for n, cluster in enumerate(ordered)}
    names: dict[int, str] = {}
    used: set[str] = set()
    for cluster in ordered:
        name = counts[cluster].most_common(1)[0][0] if counts[cluster] else fallback[cluster]
        if name in used:
            name = fallback[cluster]
        used.add(name)
        names[cluster] = name
    return names


def _smooth_short_between_same_speaker(records: list[dict[str, Any]]) -> None:
    for i in range(1, len(records) - 1):
        rec = records[i]
        if not rec.get("text") or rec.get("ms") is None or rec.get("end_ms") is None:
            continue
        if int(rec["end_ms"]) - int(rec["ms"]) > 1200:
            continue
        prev = records[i - 1]
        nxt = records[i + 1]
        if (
            prev.get("speaker") == nxt.get("speaker")
            and prev.get("speaker") != rec.get("speaker")
            and int(rec["ms"]) - int(prev.get("end_ms") or 0) <= 1800
        ):
            rec["speaker_before_smooth"] = rec.get("speaker")
            rec["speaker"] = prev["speaker"]
