"""清書（会議後の非同期再処理）."""
from __future__ import annotations

import json
import time

import numpy as np

from ._constants import SR

# ---------- 清書（会議後の非同期再処理） ----------
# RTの話者分離は速い応酬で崩れる(実測: 高速応酬区間で1ラベルに併合)。非同期APIは
# 全文脈を見られるため分離精度が大幅に高い(公式)。終了時に録音全体を再処理し、
# async話者を声紋プロファイルで実名に対応づけて「清書版」議事録を作る。

API_BASE = "https://api.soniox.com"


def _wav_bytes(pcm: bytes) -> bytes:
    import struct
    n = len(pcm)
    return (b"RIFF" + struct.pack("<I", 36 + n) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16) +
            b"data" + struct.pack("<I", n) + pcm)


def _api(api_key: str, method: str, path: str, body=None, ctype=None, timeout=120):
    import urllib.request
    req = urllib.request.Request(API_BASE + path, data=body, method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    if ctype:
        req.add_header("Content-Type", ctype)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


def _group_tokens(tokens: list[dict]) -> list[tuple]:
    """async結果のトークン列を (start_ms, end_ms, 話者, テキスト) の発話列へ."""
    utts = []
    cur = None   # [start, end, spk, text]
    for tk in tokens:
        text = tk.get("text") or ""
        if not text or text == "<end>":
            continue
        spk = tk.get("speaker")
        if cur is None or spk != cur[2]:
            if cur and cur[3].strip():
                utts.append(tuple(cur))
            cur = [tk.get("start_ms"), tk.get("end_ms"), spk, ""]
        if tk.get("end_ms") is not None:
            cur[1] = tk["end_ms"]
        cur[3] += text
    if cur and cur[3].strip():
        utts.append(tuple(cur))
    return utts


def _map_speakers(utts: list[tuple], pcm: bytes, tracker) -> dict:
    """async話者ID → 表示キー（人物との1対1割当）.

    各async話者の長い発話の声紋平均をプロファイルと照合し、類似の高いペアから
    貪欲に1対1で割り当てる。1対1にしないと、同一再生チェーン等で複数のasync話者が
    同じ人物に畳まれ、清書の話者数がライブより減る事故が起きる（2026-06-12実測）。
    """
    mapping = {}
    if tracker is None:
        return mapping
    by_spk: dict = {}
    for s, e, spk, _ in utts:
        if s is None or e is None or spk is None:
            continue
        by_spk.setdefault(str(spk), []).append((e - s, s, e))
    # アクティブなプロファイルのみ対象（セッション中に使ったもの＋自動登録）
    active = {k: v for k, v in tracker.profiles.items() if k in tracker._active_keys}
    pairs = []   # (sim, async話者, 人物)
    for spk, segs in by_spk.items():
        segs = [x for x in sorted(segs, reverse=True) if x[0] >= 1200][:6]
        embs = []
        for _, s, e in segs:
            wav = np.frombuffer(pcm[s * 32: e * 32], dtype="<i2").astype(np.float32) / 32768.0
            emb = tracker._embed(wav)
            if emb is not None:
                embs.append(emb)
        if embs:
            prof = np.mean(embs, axis=0)
            prof = prof / np.linalg.norm(prof)
            for n, v in active.items():
                sim = float(np.dot(v, prof))
                if sim >= tracker.dedupe:
                    pairs.append((sim, spk, n))
    used_spk, used_person = set(), set()
    for _sim, spk, n in sorted(pairs, reverse=True):
        if spk in used_spk or n in used_person:
            continue
        mapping[spk] = n
        used_spk.add(spk)
        used_person.add(n)
    return mapping


def polish(api_key: str, pcm: bytes, lang: str, tracker, log=print) -> list[dict]:
    """録音全体を非同期APIで再処理し、清書版のrecordsを返す."""
    log("# 清書: 音声をアップロード中…")
    import uuid
    b = "----spkattr" + uuid.uuid4().hex
    body = ((f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; "
             f"filename=\"meeting.wav\"\r\nContent-Type: audio/wav\r\n\r\n").encode()
            + _wav_bytes(pcm) + f"\r\n--{b}--\r\n".encode())
    file_id = _api(api_key, "POST", "/v1/files", body,
                   f"multipart/form-data; boundary={b}", timeout=600)["id"]
    tid = None
    try:
        cfg = {"model": "stt-async-v4", "language_hints": [lang],
               "enable_speaker_diarization": True, "file_id": file_id}
        tid = _api(api_key, "POST", "/v1/transcriptions",
                   json.dumps(cfg).encode(), "application/json")["id"]
        log("# 清書: 再処理を待っています…")
        t0 = time.time()
        while True:
            st = _api(api_key, "GET", f"/v1/transcriptions/{tid}")
            if st["status"] == "completed":
                break
            if st["status"] == "error":
                raise RuntimeError(st.get("error_message", "unknown"))
            if time.time() - t0 > 600:
                raise TimeoutError("非同期処理が10分以内に完了しませんでした")
            time.sleep(2)
        tokens = _api(api_key, "GET", f"/v1/transcriptions/{tid}/transcript")["tokens"]
    finally:   # 後始末（失敗しても続行）
        try:
            if tid:
                _api(api_key, "DELETE", f"/v1/transcriptions/{tid}")
            _api(api_key, "DELETE", f"/v1/files/{file_id}")
        except Exception:
            pass
    utts = _group_tokens(tokens)
    log(f"# 清書: {len(utts)}発話を取得、話者を声紋で照合中…")
    mapping = _map_speakers(utts, pcm, tracker)
    return [{"ms": s, "speaker": mapping.get(str(spk), "#" + str(spk)), "text": tx.strip()}
            for s, e, spk, tx in utts]
