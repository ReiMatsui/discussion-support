#!/usr/bin/env python3
"""録音済み会議で、声紋埋め込みモデルの「話者分離度」を実測するハーネス.

目的: 分岐を足す前に「埋め込みモデルを替えたら、あなたの実際の声が分離されるか」を
数値で判断する。同一話者ペアと別話者ペアのコサイン類似度の分布が分かれているほど良い。
EER（同一/別の取り違え率）が低いほど良い。

使い方（uv で実行 = torch/torchaudio が必要）:
    uv run python scripts/measure_embeddings.py                # 最新の長いセッションを自動選択
    uv run python scripts/measure_embeddings.py 2026-06-25_1614
    uv run python scripts/measure_embeddings.py 2026-06-25_1614 --speakers 黒田 としや わっち
    uv run python scripts/measure_embeddings.py 2026-06-25_1614 --no-redimnet2   # 現行モデルだけ

出力: モデルごとに、全体と「短い発話(0.4〜1.5秒)」のEER・平均類似度を表示し、
候補モデルが現行より分離を改善するかの所見を出す。

注意: 正解ラベルは議事録の最終話者名（＝今まさに改善したい不完全な出力）なので、
完璧な基準ではない。だが同じ音声・同じラベルでモデル間を比べる相対比較は、ラベル
ノイズの影響を受けにくく、「より良い埋め込みが効くか」の判断には十分使える。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import wave

import numpy as np

SR = 16000
TR = "transcripts"


# ---------------------------------------------------------------------------
# データ読み込み
# ---------------------------------------------------------------------------
def _latest_session() -> str | None:
    """turns が最も大きい（＝発話が多い）セッションの basename を返す."""
    turns = glob.glob(os.path.join(TR, "*.turns.jsonl"))
    if not turns:
        return None
    turns.sort(key=os.path.getsize, reverse=True)
    return os.path.basename(turns[0])[: -len(".turns.jsonl")]


def _read_turns(base: str) -> list[dict]:
    with open(os.path.join(TR, base + ".turns.jsonl")) as f:
        return [json.loads(ln) for ln in f]


def _read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def _load_segments(base: str, min_dur: float, speakers: list[str] | None,
                   per_speaker: int):
    """[(speaker, wav_segment, dur_sec), ...] を返す."""
    wav = _read_wav(os.path.join(TR, base + ".wav"))
    rows = _read_turns(base)
    segs, counts = [], {}
    for r in rows:
        sp = r.get("speaker")
        ms, end = r.get("ms"), r.get("end_ms")
        if sp is None or ms is None or end is None or end <= ms:
            continue
        if speakers and sp not in speakers:
            continue
        dur = (end - ms) / 1000.0
        if dur < min_dur:
            continue
        if counts.get(sp, 0) >= per_speaker:
            continue
        a, b = int(ms * SR / 1000), int(end * SR / 1000)
        seg = wav[a:b]
        if seg.size < int(SR * min_dur):
            continue
        segs.append((sp, seg, dur))
        counts[sp] = counts.get(sp, 0) + 1
    return segs, counts


# ---------------------------------------------------------------------------
# 埋め込みモデル
# ---------------------------------------------------------------------------
def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _embedder_current(model_name: str):
    """アプリ現行の声紋抽出（VoiceProfiles._embed を再利用）."""
    sys.path.insert(0, "src")
    from das.asr.live._voice_profiles import VoiceProfiles
    vp = VoiceProfiles(path="/tmp/_measure_no_voices.json", model=model_name, auto=False)

    def embed(seg: np.ndarray):
        e = vp._embed(seg)
        return None if e is None else _unit(e)
    return embed


def _embedder_redimnet2(model_name: str, train_type: str, dataset: str):
    """ReDimNet2（候補モデル）を torch.hub で読み込む."""
    import torch
    model = torch.hub.load("PalabraAI/redimnet2", "redimnet2", model_name=model_name,
                           train_type=train_type, dataset=dataset, pretrained=True)
    model.eval()

    def embed(seg: np.ndarray):
        with torch.no_grad():
            wav = torch.from_numpy(seg.astype(np.float32)).unsqueeze(0)
            emb = model(wav).squeeze(0).cpu().numpy()
        return _unit(emb)
    return embed


# ---------------------------------------------------------------------------
# 分離度の計算
# ---------------------------------------------------------------------------
def _eer(same: list[float], diff: list[float]) -> float:
    """同一ペアsim と 別ペアsim から EER(%) を求める."""
    if not same or not diff:
        return float("nan")
    same_a, diff_a = np.array(same), np.array(diff)
    ts = np.linspace(-0.2, 1.0, 1201)
    fars = np.array([np.mean(diff_a >= t) for t in ts])   # 別人を同一と誤る率
    frrs = np.array([np.mean(same_a < t) for t in ts])    # 同一を別人と誤る率
    i = int(np.argmin(np.abs(fars - frrs)))               # FAR=FRR となる点
    return float((fars[i] + frrs[i]) / 2 * 100)


def _pairs(embs: list[tuple[str, np.ndarray, float]], dur_lo=None, dur_hi=None,
           max_pairs=20000):
    same, diff = [], []
    sel = [(sp, e, d) for sp, e, d in embs
           if (dur_lo is None or d >= dur_lo) and (dur_hi is None or d < dur_hi)]
    rng = np.random.default_rng(0)
    n = len(sel)
    idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(idx) > max_pairs:
        idx = [idx[k] for k in rng.choice(len(idx), max_pairs, replace=False)]
    for i, j in idx:
        sim = float(np.dot(sel[i][1], sel[j][1]))
        (same if sel[i][0] == sel[j][0] else diff).append(sim)
    return same, diff


def _report(name: str, embs: list[tuple[str, np.ndarray, float]]):
    print(f"\n=== {name} ===")
    same, diff = _pairs(embs)
    if not same or not diff:
        print("  ペアが不足（話者数/セグメント数が少なすぎ）")
        return None
    eer = _eer(same, diff)
    print(f"  全体:      同一sim {np.mean(same):.3f} / 別sim {np.mean(diff):.3f}"
          f" / 差 {np.mean(same) - np.mean(diff):.3f} / EER {eer:.1f}%"
          f"  (同{len(same)}/別{len(diff)}ペア)")
    s_s, s_d = _pairs(embs, 0.4, 1.5)
    if s_s and s_d:
        eer_s = _eer(s_s, s_d)
        print(f"  短い発話(0.4-1.5s): 同一sim {np.mean(s_s):.3f} / 別sim {np.mean(s_d):.3f}"
              f" / 差 {np.mean(s_s) - np.mean(s_d):.3f} / EER {eer_s:.1f}%")
    return eer


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session", nargs="?", default=None,
                    help="セッションの basename（省略時は最大のものを自動選択）")
    ap.add_argument("--speakers", nargs="*", default=None,
                    help="対象話者名（省略時は命名済みの話者を自動抽出）")
    ap.add_argument("--current-model", default="redimnet",
                    help="現行モデル名（redimnet/ecapa/resemblyzer）")
    ap.add_argument("--min-dur", type=float, default=0.4, help="最小セグメント長(秒)")
    ap.add_argument("--per-speaker", type=int, default=50, help="話者ごとの最大セグメント数")
    ap.add_argument("--no-redimnet2", action="store_true", help="ReDimNet2比較を省略")
    ap.add_argument("--rd2", default="b6:lm:vb2+vox2_v0",
                    help="ReDimNet2 構成 model:train_type:dataset（例 b6:lm:vb2+vox2_v0, b3:lm:vox2）")
    args = ap.parse_args()

    base = args.session or _latest_session()
    if not base or not os.path.exists(os.path.join(TR, base + ".wav")):
        sys.exit(f"音声が見つかりません: {base}（{TR}/ に *.wav と *.turns.jsonl が必要）")
    print(f"# セッション: {base}")

    speakers = args.speakers
    if not speakers:
        # 命名済み（#/話者/人物/?/AI 以外）の話者を自動抽出
        rows = _read_turns(base)
        from collections import Counter
        c = Counter(r.get("speaker") for r in rows if r.get("speaker"))
        bad = ("#", "話者", "人物", "?", "未確定", "ファシリテーター", "パートナー")
        speakers = [s for s, _ in c.most_common()
                    if s and not any(s.startswith(b) or s == b for b in bad)]
        print(f"# 対象話者（命名済み・自動抽出）: {speakers}")
        if len(speakers) < 2:
            print("# 命名済み話者が2人未満のため、全話者(人物N含む)で計測します"
                  "（ラベルノイズに注意）")
            speakers = [s for s, _ in c.most_common()
                        if s and not any(s.startswith(b) for b in ("?", "ファ", "パ"))]

    segs, counts = _load_segments(base, args.min_dur, speakers, args.per_speaker)
    print(f"# セグメント数: {len(segs)}  話者別: {counts}")
    if len({sp for sp, _, _ in segs}) < 2:
        sys.exit("話者が2人未満。--speakers で指定するか別セッションで試してください。")

    results = {}
    # 現行モデル
    try:
        emb = _embedder_current(args.current_model)
        embs = [(sp, e, d) for sp, s, d in segs if (e := emb(s)) is not None]
        results[f"現行({args.current_model})"] = _report(f"現行モデル: {args.current_model}", embs)
    except Exception as e:
        print(f"# 現行モデルの読み込み失敗: {type(e).__name__}: {e}")

    # 候補 ReDimNet2
    if not args.no_redimnet2:
        try:
            mn, tt, ds = [*args.rd2.split(":"), "lm", "vox2"][:3]
            emb2 = _embedder_redimnet2(mn, tt, ds)
            embs2 = [(sp, e, d) for sp, s, d in segs if (e := emb2(s)) is not None]
            results[f"ReDimNet2-{mn}"] = _report(f"候補: ReDimNet2-{mn} ({tt},{ds})", embs2)
        except Exception as e:
            print(f"# ReDimNet2 の読み込み失敗（--no-redimnet2 で省略可）: {type(e).__name__}: {e}")

    # 所見
    print("\n=== 所見 ===")
    vals = {k: v for k, v in results.items() if v is not None and v == v}
    if len(vals) >= 2:
        cur = next((v for k, v in vals.items() if k.startswith("現行")), None)
        cand = next((v for k, v in vals.items() if k.startswith("ReDimNet2")), None)
        if cur is not None and cand is not None:
            d = cur - cand
            if d > 3:
                print(f"  候補が現行よりEERを{d:.1f}pt改善 → 埋め込み差し替えに効果が見込める。")
            elif d < -1:
                print(f"  候補は現行より悪い（{-d:.1f}pt）→ 差し替えの価値は薄い。別路線を検討。")
            else:
                print(f"  ほぼ同等（差{d:.1f}pt）→ 埋め込み更新では大きく変わらない。"
                      "登録前提や専用diarizerなど別路線を検討すべき。")
    print("  目安: EERが概ね10%未満なら実用的な分離。20%超なら、その音声では"
          "話者がそもそも分離しづらい（モデル更新だけでは厳しい）。")


if __name__ == "__main__":
    main()
