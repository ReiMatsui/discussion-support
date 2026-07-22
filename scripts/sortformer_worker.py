#!/usr/bin/env python3
"""Streaming Sortformer 常駐ワーカー（NeMo専用venvで動かすサブプロセス）.

本体(das)のプロセスからは `SortformerLocalDiarizationProvider` が起動し、
  stdin  : 16kHz mono PCM16 の生バイト列（チャンク粒度は任意）
  stdout : JSON Lines の話者イベント
           {"e": "ready"}                                  … モデル読込完了
           {"e": "start", "ms": <int>, "spk": "SPEAKER_00"} … 発話開始
           {"e": "end",   "ms": <int>, "spk": "SPEAKER_00"} … 発話終了
  stderr : ログ（本体は読み流すだけ）
で通信する。依存は NeMo venv 側のみ（このファイルは das パッケージを
import しない）。設計: docs/design/sortformer_live_setup_2026-07-22.md。

実装は NeMo の SortformerEncLabelModel.forward_streaming() のチャンク分割
（streaming_feat_loader）を「stdin から届く分だけ逐次」に置き換えたもの。
モデルの streaming プリセットは HF モデルカード（v2/v2.1）の公表値を使う。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# イベント出力(JSON Lines)の専用チャネルを確保し、fd1 は stderr へ付け替える。
# NeMo/依存ライブラリが stdout に吐くログが混ざるとイベントのパースが壊れる
# ため、fd レベルで防御する（Python の print も C ライブラリも巻き取れる）。
_EVENT_OUT = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)

# HF モデルカード公表のストリーミング設定（単位: 80ms フレーム）。
# low は「1.04秒レイテンシ」構成。high は精度重視のバッチ寄り構成で、
# ライブには不向きだが検証用に残す。
_PRESETS = {
    "low":  {"chunk_len": 6,   "chunk_right_context": 7,
             "fifo_len": 188, "spkcache_update_period": 144,
             "spkcache_len": 188},
    "high": {"chunk_len": 340, "chunk_right_context": 40,
             "fifo_len": 40,  "spkcache_update_period": 300,
             "spkcache_len": 188},
}

SR = 16000
HOP = 160          # 特徴フレームのホップ（10ms @16kHz）
WIN = 512          # STFT 窓長（32ms）
SUB = 8            # エンコーダの subsampling（10ms×8 = 80ms 出力フレーム）
FEAT_WINDOW_SEC = 30.0   # 特徴正規化(per_feature)を安定させる後方窓


def _emit(obj: dict) -> None:
    _EVENT_OUT.write(json.dumps(obj) + "\n")
    _EVENT_OUT.flush()


def _log(msg: str) -> None:
    print(f"# [sortformer-worker] {msg}", file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/diar_streaming_sortformer_4spk-v2.1")
    ap.add_argument("--latency", choices=sorted(_PRESETS), default="low")
    ap.add_argument("--device", default="cpu",
                    help="cpu / mps (Apple Silicon GPU) / cuda")
    ap.add_argument("--onset", type=float, default=0.5)
    ap.add_argument("--offset", type=float, default=0.5)
    ap.add_argument("--min-on-frames", type=int, default=2,
                    help="開始とみなす連続活性フレーム数（80ms単位）")
    ap.add_argument("--min-off-frames", type=int, default=3,
                    help="終了とみなす連続非活性フレーム数（80ms単位）")
    args = ap.parse_args()

    import numpy as np
    import torch

    t0 = time.time()
    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.from_pretrained(args.model, map_location="cpu")
    model.eval()
    if args.device != "cpu":
        model = model.to(args.device)
    for k, v in _PRESETS[args.latency].items():
        setattr(model.sortformer_modules, k, v)
    n_spk = model.sortformer_modules.n_spk
    chunk_len = model.sortformer_modules.chunk_len            # 80msフレーム数
    lc = model.sortformer_modules.chunk_left_context * SUB    # 10msフレーム数
    rc = model.sortformer_modules.chunk_right_context * SUB
    _log(f"model={args.model} latency={args.latency} loaded in {time.time()-t0:.0f}s "
         f"(chunk={chunk_len*80}ms rc={rc*10}ms)")

    state = model.sortformer_modules.init_streaming_state(
        batch_size=1, async_streaming=True, device=model.device)
    total_preds = torch.zeros((1, 0, n_spk), device=model.device)

    audio = np.zeros(0, dtype=np.float32)   # セッション先頭からの全音声
    consumed = 0                            # 生バイトの端数持ち越し
    stt_feat = 0                            # 次チャンク先頭（10msフレーム）
    frame_cursor = 0                        # 出力済み 80ms フレーム数
    active: dict[int, bool] = {i: False for i in range(n_spk)}
    run_on: dict[int, int] = {i: 0 for i in range(n_spk)}
    run_off: dict[int, int] = {i: 0 for i in range(n_spk)}
    pending_start: dict[int, int] = {}

    _emit({"e": "ready"})

    def _features(a_from: int, a_to: int):
        """音声サンプル区間の特徴量を、後方窓つき正規化で計算して返す.

        per_feature 正規化がチャンク長に依存して暴れないよう、常に
        FEAT_WINDOW_SEC ぶんの後方文脈を含めて計算し、必要フレームを切り出す。
        """
        ctx_from = max(0, a_from - int(FEAT_WINDOW_SEC * SR))
        seg = audio[ctx_from:a_to]
        sig = torch.from_numpy(seg).unsqueeze(0).to(model.device)
        length = torch.tensor([seg.shape[0]], device=model.device)
        feats, _ = model.process_signal(audio_signal=sig, audio_signal_length=length)
        skip = (a_from - ctx_from) // HOP
        return feats[:, :, skip:]

    def _step(feat_lo_frames: int, feats, right_offset: int):
        nonlocal state, total_preds, frame_cursor
        with torch.no_grad():
            state, total_preds = model.forward_streaming_step(
                processed_signal=torch.transpose(feats, 1, 2).to(model.device),
                processed_signal_length=torch.tensor([feats.shape[2]],
                                                     device=model.device),
                streaming_state=state,
                total_preds=total_preds,
                left_offset=feat_lo_frames,
                right_offset=right_offset,
            )
        preds = total_preds[0, frame_cursor:, :].cpu().numpy()
        for i in range(preds.shape[0]):
            t_ms = (frame_cursor + i) * 80
            for s in range(n_spk):
                on = preds[i, s] >= (args.onset if not active[s] else args.offset)
                if on:
                    run_on[s] += 1
                    run_off[s] = 0
                    if not active[s]:
                        if s not in pending_start:
                            pending_start[s] = t_ms
                        if run_on[s] >= args.min_on_frames:
                            active[s] = True
                            _emit({"e": "start", "ms": pending_start.pop(s),
                                   "spk": f"SPEAKER_{s:02d}"})
                else:
                    run_off[s] += 1
                    run_on[s] = 0
                    pending_start.pop(s, None)
                    if active[s] and run_off[s] >= args.min_off_frames:
                        active[s] = False
                        _emit({"e": "end", "ms": t_ms, "spk": f"SPEAKER_{s:02d}"})
        frame_cursor = total_preds.shape[1]

    buf = b""
    stream = sys.stdin.buffer
    while True:
        data = stream.read(3200)   # 100ms ぶん。EOF で b""
        eof = not data
        if data:
            buf += data
            n = len(buf) // 2 * 2
            if n:
                pcm = np.frombuffer(buf[:n], dtype=np.int16)
                audio = np.concatenate([audio, pcm.astype(np.float32) / 32768.0])
                buf = buf[n:]
                consumed += n
        # チャンク（+右文脈）が揃っている限り処理する
        while True:
            end_feat = stt_feat + chunk_len * SUB
            need_samples = (end_feat + rc) * HOP + WIN
            if len(audio) < need_samples:
                if not eof:
                    break
                # EOF: 右文脈なしで、残りをチャンク刻みのまま出し切る
                avail_feat = max(0, (len(audio) - WIN) // HOP)
                if avail_feat <= stt_feat:
                    break
                end_feat = min(stt_feat + chunk_len * SUB, avail_feat)
                lo = min(lc, stt_feat)
                feats = _features((stt_feat - lo) * HOP,
                                  min(len(audio), end_feat * HOP + WIN))
                _step(lo, feats, 0)
                stt_feat = end_feat
                continue
            lo = min(lc, stt_feat)
            feats = _features((stt_feat - lo) * HOP, (end_feat + rc) * HOP + WIN)
            _step(lo, feats, rc)
            stt_feat = end_feat
        if eof:
            for s in range(n_spk):
                if active[s]:
                    _emit({"e": "end", "ms": frame_cursor * 80,
                           "spk": f"SPEAKER_{s:02d}"})
            _log("EOF, done")
            return


if __name__ == "__main__":
    main()
