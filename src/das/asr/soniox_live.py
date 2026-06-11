"""1マイク + Soniox リアルタイム議事録ツール（日本語・話者分離内蔵）.

【das統合版】speaker-attribution リポジトリ v1.0 からの移植（上流は凍結、以後はこちらが正）。
das連携フック: モジュール変数 ON_UTTERANCE に callable(speaker:str, text:str) を設定すると、
確定発話ごとに呼ばれる（das listen-soniox がオーケストレータへ流すのに使用）。

1本のマイクで「誰が・何を」をライブ取得する本線ツール。
Sonioxのストリーミング(WebSocket)に音声を流し、speaker付きトークンが返るので、
多マイク・ゲート・同期なしで who-said-what が出る。

機能:
  - 話者ごとに色分けしたライブ表示（確定前のテキストは薄く表示）
  - Markdown議事録 + HTML を transcripts/ に自動保存（発話確定ごと＝クラッシュ安全）
  - HTMLはブラウザ自動オープン、ライブ中2秒ごと自動更新（--no-openで無効）
  - 声紋プロファイル方式の話者特定（登録不要で自動補正）。判定は2経路のみ:
      ① 即時判定: 声紋が強一致した発話はその場で人物確定（入れ替わりも補正）
      ② それ以外は3発話バッファ: 一貫した3発話を束ねて「既存人物に合流 or 新規人物N」
      しきい値は2層: モデル別既定値 → 人物別しきい値(本人の一致sim中央値-0.12、
      新声の巻き取り防止。厳しくする方向にのみ働く)。
      不変条件: 一度確定した人物キーは書き換えない（遡及置換は 話者N→人物N の昇格のみ）。
      「1=松井」で実名化、実名のみ voices.json に永続化 → 次回から自動で実名表示。
  - 終了時に清書: 録音全体を非同期APIで再処理し、全文脈の話者分離＋声紋実名対応の
    最終版(日時.final.md/.html)を自動生成（高速応酬でのRT分離崩れへの対策。--no-polishで無効）
  - 「fix 2=1」「fix 人物2=人物1」で誤った話者の統合（過去の発言も修正）
  - 診断ログ(日時.diag.jsonl): 発話ごとの判定根拠を常時記録（問題解析用）

準備(Mac):
  uv add websockets sounddevice
  export SONIOX_API_KEY=...   # https://console.soniox.com で取得

使い方:
  uv run python offshelf/live_soniox.py            # 実マイクでライブ
  uv run python offshelf/live_soniox.py --wav offshelf/ami_raw/mic0.wav  # ファイル擬似ライブ
  実行中: 「1=松井」Enter で話者登録 / Ctrl+C で終了（保存先を表示）
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import queue
import re
import sys
import threading
import time

import numpy as np

ON_UTTERANCE = None   # das連携: 確定発話ごとに (話者表示名, テキスト) で呼ばれる
_SYS_HOOK = None      # main()実行中のみ登録される(add_sys+saveへの橋)


def post_system(text: str) -> None:
    """das連携: ライブ議事録のタイムラインにシステム行(💡介入など)を外部から追加する."""
    if _SYS_HOOK is not None:
        _SYS_HOOK(text)

SR = 16000
WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
SM_WS_URL = "wss://eu.rt.speechmatics.com/v2/"


def sm_to_res(msg: dict, lang: str = "ja") -> dict:
    """SpeechmaticsのRTメッセージをSoniox互換のトークン列に翻訳する.

    供給源を差し替えるだけで、声紋層・表示・保存・清書は無変更で動く。
    話者ラベル: S1→"1"(表示は話者1)、不明UUはそのまま。
    """
    m = msg.get("message")
    if m == "Error":
        return {"error_code": msg.get("type"), "error_message": msg.get("reason")}
    if m == "EndOfTranscript":
        return {"finished": True, "tokens": []}
    if m == "EndOfUtterance":
        return {"tokens": [{"text": "<end>", "is_final": True}]}
    if m in ("AddTranscript", "AddPartialTranscript"):
        final = m == "AddTranscript"
        toks = []
        for r in msg.get("results", []):
            alts = r.get("alternatives") or []
            content = alts[0].get("content", "") if alts else ""
            if not content:
                continue
            spk = (alts[0].get("speaker") or "UU")
            if spk.startswith("S") and spk[1:].isdigit():
                spk = spk[1:]
            if (lang not in ("ja", "zh", "cmn", "yue") and toks
                    and r.get("type") == "word"):
                content = " " + content   # 分かち書き言語は語間スペースを補う
            toks.append({"text": content, "speaker": spk,
                         "start_ms": int(r["start_time"] * 1000),
                         "end_ms": int(r["end_time"] * 1000),
                         "is_final": final})
        return {"tokens": toks}
    return {"tokens": []}   # RecognitionStarted / AudioAdded / Info / Warning 等は無視

RESET = "\x1b[0m"
DIM = "\x1b[2m"
CLEAR_LINE = "\r\x1b[K"
PALETTE = ["\x1b[36m", "\x1b[33m", "\x1b[35m", "\x1b[32m", "\x1b[34m", "\x1b[91m"]
HTML_PALETTE = ["#0e7490", "#a16207", "#7e22ce", "#15803d", "#1d4ed8", "#dc2626"]

HTML_TMPL = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
{refresh}<title>議事録 {title}</title>
<style>
body {{ font-family: -apple-system, "Hiragino Sans", sans-serif; max-width: 760px;
       margin: 2rem auto; padding: 0 1rem; background: #fafafa; color: #1f2937; }}
h1 {{ font-size: 1.2rem; }} .meta {{ color: #6b7280; font-size: .85rem; }}
.u {{ margin: .5rem 0; padding: .55rem .8rem; background: #fff; border-radius: 10px;
     border: 1px solid #e5e7eb; }}
.u .who {{ font-weight: 700; margin-right: .5em; }}
.u .ts {{ color: #9ca3af; font-size: .8rem; margin-right: .6em; font-variant-numeric: tabular-nums; }}
.live {{ color: #16a34a; font-size: .8rem; }}
.badge {{ background: #fef3c7; color: #92400e; font-size: .7rem; border-radius: 6px;
         padding: .05em .45em; margin-left: .55em; vertical-align: middle; }}
.sys {{ text-align: center; color: #6b7280; font-size: .78rem; margin: .45rem 0; }}
</style></head><body>
<h1>議事録 {title}</h1>
<p class="meta">{status} / 話者: {speakers}</p>
{body}
<script>window.scrollTo(0, document.body.scrollHeight);</script>
</body></html>
"""


def load_env(path: str = ".env") -> None:
    """プロジェクト直下の .env からAPIキー等を読み込む（既に設定済みの環境変数を優先）.

    形式: KEY=VALUE の行（#始まりはコメント）。依存なしの最小実装。
    """
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


def fmt_ts(ms: int | None) -> str:
    if ms is None:
        return "--:--"
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


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
            for n, v in tracker.profiles.items():
                sim = float(np.dot(v, prof))
                if sim >= tracker.dedupe:
                    pairs.append((sim, spk, n))
    used_spk, used_person = set(), set()
    for sim, spk, n in sorted(pairs, reverse=True):
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


class VoiceProfiles:
    """凍結プロファイル照合による話者特定（台帳固定・誤り非伝播）.

    判定は2経路だけ:
      ① 即時判定 — 単発声紋が強一致(thresh＋2位とmargin差)した時だけ、その場で人物確定
      ② それ以外は3発話バッファ — 一貫した3発話を束ね「既存人物に合流(dedupe) or 新規人物N」
    しきい値は2層構造（厳しくする方向にのみ働き、最悪でも既定値の挙動に戻る）:
      1. モデル別既定値(DEFAULTS)
      2. 人物別しきい値(その人物の一致sim中央値-0.12 = 新規性検出。中途半端な類似の
         新しい声を既存人物に巻き取らない)。即時判定のみに適用
    不変条件: 確定済みの人物キーは書き換えない（遡及置換は #ラベル→人物 の昇格のみ）。
    実名(enroll)のみ voices.json に永続化、匿名「人物N」はセッション限り。
    """

    ANON = re.compile(r"^人物\d+$")

    # モデル別の既定しきい値（実音声プールで校正済み。スコアのスケールが違う）
    # resemblyzer: 軽量・依存少。同一/別人の分布に重なりあり（分離マージン-0.06）
    # ecapa: ほぼ完全分離(+0.01)＋10倍速。混合音声を成分話者と強くマッチさせる癖
    # redimnet: Interspeech 2024。本プールで最良の分離(+0.10)・27ms級・5M params
    # (即時判定th, 合流dedupe, 一貫性consist)。dedupeは三発話プロファイル同士の比較なので
    # 単発より高め（2026-06-11夜: 0.30→巻き取り復活/個人別→本人分裂のため固定の中庸値に）
    DEFAULTS = {"resemblyzer": (0.75, 0.72, 0.62), "ecapa": (0.35, 0.40, 0.30),
                "redimnet": (0.42, 0.50, 0.34)}

    def __init__(self, path: str = "voices.json", thresh: float | None = None,
                 min_sec: float = 1.0, margin: float = 0.05, auto: bool = True,
                 consist: float | None = None, dedupe: float | None = None,
                 model: str = "resemblyzer"):
        self.model = model
        if model == "ecapa":
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
            enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

            def _embed_raw(wav):
                with torch.no_grad():
                    return enc.encode_batch(torch.from_numpy(wav).float().unsqueeze(0)).squeeze().numpy()
            self._embed_raw = _embed_raw
        elif model == "redimnet":
            import torch   # 初回はGitHubからコード＋重み(20MB)をダウンロード
            enc = torch.hub.load("IDRnD/ReDimNet", "ReDimNet", model_name="b2",
                                 train_type="ft_lm", dataset="vox2", trust_repo=True)
            enc.eval()

            def _embed_raw(wav):
                with torch.no_grad():
                    return enc(torch.from_numpy(wav).float().unsqueeze(0)).squeeze().numpy()
            self._embed_raw = _embed_raw
        else:
            from resemblyzer import VoiceEncoder, preprocess_wav  # 初回ロード数秒
            enc = VoiceEncoder("cpu", verbose=False)
            self._embed_raw = lambda wav: enc.embed_utterance(preprocess_wav(wav, source_sr=SR))
        d_th, d_dd, d_cs = self.DEFAULTS[model]
        self.path = path
        self.thresh = thresh if thresh is not None else d_th   # 即時判定のしきい値
        self.margin = margin   # 即時判定の追加条件: 2位との差（似た声の誤マッチ防止）
        self.auto = auto       # 未知の声の自動登録（匿名「人物N」プロファイル）
        self.consist = consist if consist is not None else d_cs  # 3発話の全ペア類似の下限
        self.dedupe = dedupe if dedupe is not None else d_dd     # 既存人物への合流しきい値
        self.min_sec = min_sec
        self.profiles: dict[str, np.ndarray] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if data.pop("_model", "resemblyzer") == model:   # 別モデルの声紋は互換性なし
                self.profiles = {k: np.asarray(v, dtype=np.float64) for k, v in data.items()}
            else:
                print(f"# 注意: {path} は別の声紋モデルで作成されたため読み込みません", flush=True)
        self.sp_map: dict[str, str] = {}                    # Sonioxラベル -> 表示キー
        self.label_embs: dict[str, list[np.ndarray]] = {}   # ラベル -> 直近声紋（手動登録・校正用）
        # 未確定の声のプール（ラベルで仕切らない）。Sonioxは新しい声を既存ラベルに混ぜて
        # 出すことがあり、ラベル別バッファだと他話者の混入で3発話一貫が永遠に成立しない
        # （実セッション診断: 蓄積33回vs登録3回）。声は声同士で束ねる。
        self.pool: list[np.ndarray] = []
        self.n_anon = 0
        # 部屋の分布計測（表示・診断専用、判定には使わない）。かつてしきい値の自動校正に
        # 使っていたが、実セッションで未発動＋「ラベル=人物」前提が崩れている(Sonioxは
        # 新しい声を既存ラベルに混ぜる)＋人物別しきい値が同じ役割をより清潔なデータで
        # 果たすため、判定への結線は撤去した(2026-06-11)。
        self.same_sims: list[float] = []
        self.diff_sims: list[float] = []
        # 人物別しきい値: 「その人物が普段一致するスコアの典型範囲」を下回る一致は弾く
        # （新規性検出）。同一再生チェーン等で別人が0.5前後の中途半端な類似を出しても、
        # 本人の典型(例:0.7台)に届かなければ巻き取らない。診断ログ解析(2026-06-11)で
        # 吸収帯0.45-0.59と本人帯0.67-0.82の分離を確認、検証で吸収率91%→0%。
        self.own_sims: dict[str, list[float]] = {}   # 人物 -> 受理された一致simの履歴
        self.embed_ms: list[float] = []                     # レイテンシ統計
        self.counts: dict[str, int] = {}                    # 判定種別の集計
        self.last: dict | None = None                       # 直近の判定内容（可視化用）
        self._lock = threading.RLock()   # classify(受信スレッド)とenroll/remap(入力スレッド)の排他

    def _note(self, kind: str, **info) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + 1
        self.last = {"kind": kind, **info}

    def _update_room_stats(self, sp: str, emb: np.ndarray) -> None:
        for l2, es in self.label_embs.items():
            tgt = self.same_sims if l2 == sp else self.diff_sims
            tgt.extend(float(np.dot(emb, e2)) for e2 in es[-3:])
        del self.same_sims[:-60]
        del self.diff_sims[:-120]

    def _person_th(self, name: str, base: float) -> float:
        """人物別しきい値 = max(基準値, その人物の一致sim中央値 - 0.12)."""
        h = self.own_sims.get(name, [])
        if len(h) >= 3:
            return max(base, float(np.median(h)) - 0.12)
        return base

    def _embed(self, wav: np.ndarray) -> np.ndarray | None:
        t0 = time.perf_counter()
        try:
            emb = self._embed_raw(wav)
            if emb is None or np.asarray(emb).ndim != 1:
                return None
        except Exception:
            return None
        self.embed_ms.append((time.perf_counter() - t0) * 1000)
        emb = np.asarray(emb, dtype=np.float64)
        return emb / np.linalg.norm(emb)

    def classify(self, wav: np.ndarray, sp, overlapped: bool = False) -> str:
        """発話を人物キーに割り当てる（経路はクラスdocstring参照）.

        overlapped=True の発話は声が混ざっていて声紋がデタラメになるため、
        声での判定をスキップして直前の対応を維持する。
        """
        with self._lock:
            return self._classify(wav, sp, overlapped)

    def _classify(self, wav: np.ndarray, sp, overlapped: bool) -> str:
        sp = str(sp)
        prev = self.sp_map.get(sp)
        kind, info = "相槌追従", {}
        if overlapped and wav.size >= SR * self.min_sec:
            kind = "重なりスキップ"
        elif wav.size >= SR * self.min_sec:
            emb = self._embed(wav)
            if emb is None:
                kind = "声紋計算不可"
            else:
                self._update_room_stats(sp, emb)   # 部屋の同一/別人分布を実測(表示・診断用)
                self.label_embs.setdefault(sp, []).append(emb)
                del self.label_embs[sp][:-10]    # 手動登録用に直近10発話だけ保持
                th, dd, cs = self.thresh, self.dedupe, self.consist
                info = {"n_prof": len(self.profiles)}   # 診断ログ用
                if self.profiles:
                    ranked = sorted(((float(np.dot(p, emb)), n)
                                     for n, p in self.profiles.items()), reverse=True)
                    sim, cand = ranked[0]
                    second = ranked[1][0] if len(ranked) > 1 else -1.0
                    info.update(sim=round(sim, 3), second=round(second, 3), name=cand, prev=prev)
                    if sim >= self._person_th(cand, th) and sim - second >= self.margin:
                        # 注: ここでbufは消さない。たまたま強一致した発話でバッファを
                        # リセットすると、新しい話者の3発話が永遠に貯まらない(検証で確認)。
                        self.sp_map[sp] = cand
                        h = self.own_sims.setdefault(cand, [])
                        h.append(sim)
                        del h[:-20]
                        self._note("補正" if (prev is not None and not prev.startswith("#")
                                              and prev != cand) else "声紋一致", label=sp, **info)
                        return cand
                kind = "蓄積中" if self.auto else "未確定"
                if self.auto:
                    # 声プール: ラベル不問で、互いに一貫する3発話が揃ったら人物化
                    sims = sorted(((float(np.dot(p, emb)), i) for i, p in enumerate(self.pool)),
                                  reverse=True)
                    cand = [i for s, i in sims[:2] if s >= cs]
                    if len(cand) == 2 and float(np.dot(self.pool[cand[0]],
                                                       self.pool[cand[1]])) >= cs:
                        triple = [self.pool[cand[0]], self.pool[cand[1]], emb]
                        for i in sorted(cand, reverse=True):
                            self.pool.pop(i)
                        prof = np.mean(triple, axis=0)
                        prof = prof / np.linalg.norm(prof)
                        hit_sim, hit = max(((float(np.dot(p, prof)), n)
                                            for n, p in self.profiles.items()), default=(-1.0, None))
                        if hit is not None and hit_sim >= dd:
                            target = hit          # 既存人物の声だった → 合流
                        else:
                            self.n_anon += 1
                            target = f"人物{self.n_anon}"
                            self.profiles[target] = prof   # 新規人物（以後凍結）
                        # 遡及置換は未確定キー(#ラベル)の昇格のみ。人物キーは絶対に書き換えない。
                        rename = ("#" + sp, target) if (prev is None or prev.startswith("#")) else None
                        self.sp_map[sp] = target
                        self._note("自動登録", label=sp, name=target, rename=rename)
                        return target
                    self.pool.append(emb)
                    del self.pool[:-12]
        # 声紋で決められない（重なり/短い相槌/蓄積中）→ ラベルの直近判定に追従
        key = prev if prev is not None else "#" + sp
        self.sp_map[sp] = key
        self._note(kind, label=sp, **info)
        return key

    def enroll(self, label: str, name: str) -> str | None:
        """「1=松井」「人物2=田中」: 話者に名前を付ける（声の登録 or 既存人物のリネーム）.

        実名を付けたプロファイルのみ voices.json に永続化される（匿名「人物N」は
        そのセッション限り）。戻り値: 旧表示キー（過去のrecords付け替え用）。
        十分な音声がまだ無ければ None。
        """
        with self._lock:
            return self._enroll(str(label), name)

    def _enroll(self, label: str, name: str) -> str | None:
        if label in self.profiles:
            # 「人物1=松井」: 既存プロファイルのリネーム
            self.profiles[name] = self.profiles.pop(label)
            old = label
        else:
            cur = self.sp_map.get(label)
            if cur is not None and cur in self.profiles:
                # ラベルが（自動登録済みの）人物に対応済み → その人物に命名
                self.profiles[name] = self.profiles.pop(cur)
                old = cur
            else:
                # ラベルの直近声紋から新規登録
                embs = self.label_embs.get(label)
                if not embs:
                    return None
                prof = np.mean(embs, axis=0)
                self.profiles[name] = prof / np.linalg.norm(prof)
                old = cur if cur is not None else "#" + label
        for k, v in list(self.sp_map.items()):
            if v == old:
                self.sp_map[k] = name
        if old in self.own_sims:
            self.own_sims[name] = self.own_sims.pop(old)
        if old != label:   # 「人物N=名前」のリネーム以外は、ラベル自体も対応づける
            self.sp_map[label] = name
        self._persist()
        return old

    def remap(self, src: str, dst: str) -> bool:
        """「fix 人物2=人物1」: srcをdstに統合（srcのプロファイルも削除し、復活を防ぐ）."""
        with self._lock:
            if src == dst:
                return False
            self.profiles.pop(src, None)   # 残すと同じ声が再びsrcと判定されて復活してしまう
            self.own_sims.pop(src, None)
            for k, v in list(self.sp_map.items()):
                if v == src:
                    self.sp_map[k] = dst
            self._persist()
            return True

    def _persist(self):
        named = {k: v.tolist() for k, v in self.profiles.items() if not self.ANON.match(k)}
        named["_model"] = self.model   # 声紋はモデル間で互換性がないため記録
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(named, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    def stats(self) -> str:
        parts = []
        if self.embed_ms:
            a = np.array(self.embed_ms)
            parts.append(f"声紋計算 {len(a)}回 平均{a.mean():.0f}ms 最大{a.max():.0f}ms")
        if len(self.same_sims) >= 8 and len(self.diff_sims) >= 12:
            parts.append(f"部屋の声紋分布(参考): ラベル内{np.median(self.same_sims):.2f}"
                         f"/ラベル間{np.median(self.diff_sims):.2f}")
        if self.counts:
            order = ["声紋一致", "補正", "自動登録", "蓄積中", "未確定", "相槌追従",
                     "重なりスキップ", "声紋計算不可"]
            parts.append("判定内訳: " + " / ".join(
                f"{k}{self.counts[k]}" for k in order if self.counts.get(k)))
        return "、".join(parts) or "判定なし"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--model", default="stt-rt-v4")
    ap.add_argument("--wav", default=None, help="指定で実マイクの代わりにファイル擬似ライブ")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="保存先mdファイル（省略時 transcripts/日時.md）")
    ap.add_argument("--no-open", action="store_true", help="ブラウザを自動で開かない")
    ap.add_argument("--no-vp", action="store_true", help="声紋照合を無効化（Sonioxのラベルをそのまま使う）")
    ap.add_argument("--voices", default="voices.json", help="声紋プロファイルの保存先(既定 voices.json)")
    ap.add_argument("--vp-model", default="redimnet", choices=["redimnet", "ecapa", "resemblyzer"],
                    help="声紋モデル(既定redimnet=2024年世代、実測の分離・通し精度とも最良。"
                         "読み込み失敗時は ecapa → resemblyzer へ自動フォールバック)")
    ap.add_argument("--vp-match", type=float, default=None,
                    help="即時判定のしきい値。省略時はモデル別の既定値"
                         "(redimnet 0.42 / ecapa 0.35 / resemblyzer 0.75)")
    ap.add_argument("--vp-no-auto", action="store_true",
                    help="未知の声の自動登録（匿名「人物N」）を無効化")
    ap.add_argument("--vp-debug", action="store_true", help="発話ごとの声紋判定の内訳を表示")
    ap.add_argument("--no-polish", action="store_true",
                    help="終了時の清書（非同期APIでの全体再処理）を行わない")
    ap.add_argument("--stt", default="soniox", choices=["soniox", "speechmatics"],
                    help="リアルタイムSTTの供給源。speechmaticsは要 SPEECHMATICS_API_KEY"
                         "（話者分離の評判が良い代替。声紋層など他の機能は不変）")
    args = ap.parse_args(argv)

    load_env()   # .env からAPIキーを読み込み（export済みの値が優先）
    api_key = os.environ.get("SONIOX_API_KEY")
    sm_key = os.environ.get("SPEECHMATICS_API_KEY")
    if args.stt == "speechmatics":
        if not sm_key:
            raise SystemExit("環境変数 SPEECHMATICS_API_KEY を設定してください"
                             "（https://portal.speechmatics.com/settings/api-keys）")
    elif not api_key:
        raise SystemExit("環境変数 SONIOX_API_KEY を設定してください（https://console.soniox.com）")

    try:
        from websockets.sync.client import connect
    except ImportError:
        raise SystemExit("uv add websockets を実行してください")

    if args.stt == "speechmatics":
        ws_url = SM_WS_URL
        ws_headers = {"Authorization": f"Bearer {sm_key}"}
        start_msg = {
            "message": "StartRecognition",
            "audio_format": {"type": "raw", "encoding": "pcm_s16le", "sample_rate": SR},
            "transcription_config": {
                "language": args.lang,
                "operating_point": "enhanced",
                "diarization": "speaker",
                "enable_partials": True,
                "max_delay": 1.2,
                "conversation_config": {"end_of_utterance_silence_trigger": 0.8},
            },
        }
    else:
        ws_url = WS_URL
        ws_headers = None
        start_msg = {
            "api_key": api_key,
            "model": args.model,
            "language_hints": [args.lang],
            "enable_speaker_diarization": True,
            "enable_endpoint_detection": True,
            "audio_format": "pcm_s16le",
            "sample_rate": SR,
            "num_channels": 1,
        }

    started = datetime.datetime.now()
    if args.out:
        out_path = args.out
    else:
        os.makedirs("transcripts", exist_ok=True)
        out_path = os.path.join("transcripts", started.strftime("%Y-%m-%d_%H%M") + ".md")
    html_path = os.path.splitext(out_path)[0] + ".html"
    diag_path = os.path.splitext(out_path)[0] + ".diag.jsonl"   # 発話ごとの判定根拠(劣化解析用)
    turns_path = os.path.splitext(out_path)[0] + ".turns.jsonl"  # das(議論支援)連携用

    # --- 状態 ---
    names: dict[str, str] = {}          # 表示キー -> 別名（声紋OFF時の命名用）
    colors: dict[str, str] = {}         # 表示キー -> ANSI色（出現順に割当）
    records: list[dict] = []            # 確定発話 {"ms", "speaker", "text"}
    state_lock = threading.Lock()

    tracker: VoiceProfiles | None = None
    if not args.no_vp:
        print("# 声紋モデルを読み込み中…", flush=True)
        for model in dict.fromkeys([args.vp_model, "ecapa", "resemblyzer"]):
            try:
                tracker = VoiceProfiles(path=args.voices, thresh=args.vp_match,
                                        auto=not args.vp_no_auto, model=model)
                if model != args.vp_model:
                    print(f"# 注意: {args.vp_model} を読み込めなかったため {model} で動作します"
                          f"（依存: uv add speechbrain torchaudio / redimnetは初回ネット接続必要）",
                          flush=True)
                print(f"# 声紋モデル: {model}", flush=True)
                break
            except Exception as e:   # 依存欠如(ImportError)もDL失敗等も次の候補へ
                print(f"#   {model}: 読み込み失敗 ({type(e).__name__})", flush=True)
                continue
        if tracker is None:
            print("# 警告: 声紋照合がOFFです！ 依存が未導入のため人物の確定・補正は行われません。", flush=True)
            print("#   有効化するには: uv add speechbrain torchaudio  →  再起動", flush=True)
        elif tracker.profiles:
            print(f"# 声紋プロファイル: {', '.join(tracker.profiles)}（{args.voices}）", flush=True)
        else:
            print(f"# 声紋プロファイル: なし。未知の声は「人物N」として自動追跡、"
                  f"「1=松井」で実名化すると次回から自動表示（{args.voices}）", flush=True)

    pcm_buf = bytearray()               # 送信済み音声の全バッファ（声紋切り出し用, 16bit）
    buf_lock = threading.Lock()

    def disp_name(key) -> str:
        key = str(key)
        if key in names:
            return names[key]
        return f"話者{key[1:]}" if key.startswith("#") else key

    def key_for_label(sp) -> str:
        sp = str(sp)
        if tracker is not None and sp in tracker.sp_map:
            return tracker.sp_map[sp]
        return "#" + sp

    def color_of(key) -> str:
        key = str(key)
        if key not in colors:
            colors[key] = PALETTE[len(colors) % len(PALETTE)]
        return colors[key]

    def rekey(old: str, new: str):
        """表示キーの付け替え: recordsと色を一括移行（話者一覧に旧キーの幽霊を残さない）."""
        with state_lock:
            for r in records:
                if r.get("speaker") == old:
                    r["speaker"] = new
            if old in colors:
                colors.setdefault(new, colors.pop(old))

    def add_sys(ms, text: str):
        """システムイベント（補正・自動登録・命名・統合）を議事録のタイムラインに残す."""
        with state_lock:
            records.append({"ms": ms, "sys": text})

    global _SYS_HOOK

    def _sys_hook(text: str) -> None:   # das介入をライブHTML/MDに反映
        add_sys(None, text)
        save()
    _SYS_HOOK = _sys_hook

    def write_md(recs=None, path=None):
        with state_lock:
            rs = records if recs is None else recs
            speakers = list(dict.fromkeys(r["speaker"] for r in rs if "speaker" in r))
            lines = [
                f"# 議事録 {started.strftime('%Y-%m-%d %H:%M')}",
                "",
                "話者: " + (", ".join(disp_name(s) for s in speakers) or "（未検出）"),
                "",
            ]
            for r in rs:
                if "sys" in r:
                    lines.append(f"> [{fmt_ts(r['ms'])}] {r['sys']}")
                    continue
                mark = " ⚡" if r.get("vp") == "補正" else ""
                lines.append(f"- **[{fmt_ts(r['ms'])}] {disp_name(r['speaker'])}{mark}**: {r['text']}")
            dst = path or out_path
            tmp = dst + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            os.replace(tmp, dst)

    def write_html(live: bool = True, recs=None, path=None, status=None):
        import html as _html
        with state_lock:
            rs = records if recs is None else recs
            parts = []
            for r in rs:
                if "sys" in r:
                    parts.append(f'<div class="sys">⚙ {_html.escape(r["sys"])}</div>')
                    continue
                sp = str(r["speaker"])
                color_of(sp)
                idx = list(colors).index(sp)
                c = HTML_PALETTE[idx % len(HTML_PALETTE)]
                badge = ""
                if r.get("vp") == "補正":
                    note = _html.escape(r.get("note", ""))
                    badge = f'<span class="badge" title="{note}">⚡声紋補正</span>'
                parts.append(
                    f'<div class="u"><span class="ts">{fmt_ts(r["ms"])}</span>'
                    f'<span class="who" style="color:{c}">{_html.escape(disp_name(sp))}</span>'
                    f'{_html.escape(r["text"])}{badge}</div>'
                )
            speakers = list(dict.fromkeys(r["speaker"] for r in rs if "speaker" in r))
            doc = HTML_TMPL.format(
                refresh='<meta http-equiv="refresh" content="2">' if live else "",
                title=started.strftime("%Y-%m-%d %H:%M"),
                status=status or ('<span class="live">● ライブ（2秒ごと自動更新）</span>'
                                  if live else "終了"),
                speakers=_html.escape(", ".join(disp_name(s) for s in speakers) or "（未検出）"),
                body="\n".join(parts) or '<p class="meta">（まだ発話なし）</p>',
            )
            dst = path or html_path
            tmp = dst + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(doc)
            os.replace(tmp, dst)

    def write_turns(recs=None, path=None):
        """discussion-support(das)のUtteranceスキーマでJSONL出力.

        `das run-session 日時.turns.jsonl` がそのまま読める形式。saveごとに全体を
        書き直すので、後からの実名化(1=松井)や統合(fix)が過去の行にも反映される。
        """
        with state_lock:
            rs = records if recs is None else recs
            lines = []
            tid = 0
            for r in rs:
                if "speaker" not in r or not r.get("text"):
                    continue
                tid += 1
                lines.append(json.dumps({"turn_id": tid, "speaker": disp_name(r["speaker"]),
                                         "text": r["text"]}, ensure_ascii=False))
            dst = path or turns_path
            tmp = dst + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            os.replace(tmp, dst)

    def save(live: bool = True):
        write_md()
        write_html(live)
        write_turns()

    def print_line(text: str):
        sys.stdout.write(CLEAR_LINE + text + "\n")
        sys.stdout.flush()

    def show_partial(sp, text: str):
        if not text.strip():
            sys.stdout.write(CLEAR_LINE)
        else:
            cols = os.get_terminal_size().columns if sys.stdout.isatty() else 120
            line = f"{disp_name(key_for_label(sp))}: {text.strip()}"
            sys.stdout.write(CLEAR_LINE + DIM + line[-(cols - 2):] + RESET)
        sys.stdout.flush()

    # --- 音声入力（マイク or ファイル）→ audio_q ---
    audio_q: "queue.Queue[bytes | None]" = queue.Queue()
    stop = threading.Event()

    def from_mic():
        import sounddevice as sd

        def cb(indata, frames, t, status):
            pcm = (np.clip(indata[:, 0], -1, 1) * 32767).astype("<i2").tobytes()
            audio_q.put(pcm)
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            device=args.device, callback=cb, blocksize=int(SR * 0.1)):
            while not stop.is_set():
                time.sleep(0.1)
        audio_q.put(None)

    def from_wav():
        import librosa
        y, _ = librosa.load(args.wav, sr=SR)
        step = int(SR * 0.12)
        for i in range(0, len(y), step):
            if stop.is_set():
                break
            pcm = (np.clip(y[i:i + step], -1, 1) * 32767).astype("<i2").tobytes()
            audio_q.put(pcm)
            time.sleep(0.12)
        audio_q.put(None)

    # --- 実行中コマンド ---
    def key_of(tok: str) -> str:
        """コマンド引数を表示キーへ: 人物名はそのまま、数字はそのラベルの現在の表示先."""
        if tracker is not None:
            if tok in tracker.profiles:
                return tok
            if tok in tracker.sp_map:
                return tracker.sp_map[tok]
        return "#" + tok

    def stdin_commands():
        while not stop.is_set():
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                break
            mfix = re.match(r"^\s*fix\s+(\S+)\s*=\s*(\S+)\s*$", line)
            m = re.match(r"^\s*(\S+?)\s*=\s*(.+?)\s*$", line)
            if mfix:
                src, dst = key_of(mfix.group(1)), key_of(mfix.group(2))
                if tracker is not None:
                    tracker.remap(src, dst)
                rekey(src, dst)
                add_sys(None, f"{disp_name(src)} を {disp_name(dst)} に統合（手動fix）")
                save()
                print_line(f"# {disp_name(src)} を {disp_name(dst)} に統合しました（過去の発言も修正済み）")
            elif m:
                label, name = m.group(1), m.group(2)
                if tracker is not None:
                    old = tracker.enroll(label, name)
                    if old is None:
                        print_line(f"# 話者{label}の音声がまだ足りません（1秒以上話してから再実行）")
                        continue
                    rekey(old, name)
                    add_sys(None, f"「{name}」の声を登録（次回の会議から自動表示）")
                    save()
                    print_line(f"# {name} の声を登録しました（過去の発言も置換、次回の会議から自動表示）")
                else:
                    with state_lock:
                        names["#" + label] = name
                    save()
                    print_line(f"# 話者{label} → {name}（過去の発言も置換済み）")
            elif line.strip():
                print_line("# コマンド: 「1=松井」(声を登録) / 「fix 2=1」「fix 人物2=人物1」(統合) / Ctrl+Cで終了")

    print(f"# {args.stt} に接続中…", flush=True)
    with connect(ws_url, additional_headers=ws_headers) as ws:
        ws.send(json.dumps(start_msg))
        threading.Thread(target=from_wav if args.wav else from_mic, daemon=True).start()
        threading.Thread(target=stdin_commands, daemon=True).start()

        def sender():
            seq = 0
            while True:
                pcm = audio_q.get()
                if pcm is None:   # 終端
                    if args.stt == "speechmatics":
                        ws.send(json.dumps({"message": "EndOfStream", "last_seq_no": seq}))
                    else:
                        ws.send("")
                    break
                with buf_lock:
                    pcm_buf.extend(pcm)   # STTの時刻軸と完全一致する位置で蓄積
                ws.send(pcm)
                seq += 1
        threading.Thread(target=sender, daemon=True).start()

        save()
        print("# 開始。話してください（「1=松井」で声を登録 / Ctrl+Cで終了）", flush=True)
        print(f"# 保存先: {out_path}", flush=True)
        print(f"# ブラウザ表示: open {html_path}（ライブ中は2秒ごと自動更新）\n", flush=True)
        if not args.no_open:
            import webbrowser
            webbrowser.open("file://" + os.path.abspath(html_path))

        cur_speaker = None
        cur_text = ""
        cur_ms: int | None = None
        cur_end: int | None = None
        recent_segs: list[tuple] = []   # (start, end, ラベル) 直近の確定発話（重なり検出用）

        def overlaps_other(start, end, label) -> bool:
            if start is None or end is None:
                return False
            return any(l != label and min(e, end) - max(s, start) > 0
                       for s, e, l in recent_segs)

        def flush():
            nonlocal cur_text, cur_ms, cur_end
            if cur_text.strip():
                label = str(cur_speaker)
                if tracker is not None:
                    if cur_ms is not None and cur_end is not None and cur_end > cur_ms:
                        with buf_lock:
                            seg = bytes(pcm_buf[cur_ms * 32: cur_end * 32])  # 16サンプル/ms×2byte
                        wav = np.frombuffer(seg, dtype="<i2").astype(np.float32) / 32768.0
                    else:
                        wav = np.zeros(0, dtype=np.float32)
                    sp_id = tracker.classify(wav, cur_speaker,
                                             overlapped=overlaps_other(cur_ms, cur_end, label))
                    d = tracker.last
                    rec_extra = {}
                    if d and d["kind"] == "補正":
                        note = (f"声紋でラベル{d['label']}の取り違えを修正"
                                f"（類似{d['sim']:.2f}、放置なら{disp_name(d['prev'])}の発言になっていた）")
                        rec_extra = {"vp": "補正", "note": note}
                        print_line(f"# ⚡補正: {note}")
                    elif d and d["kind"] == "自動登録":
                        if d["rename"]:   # 「#ラベル→人物」の昇格のみ遡及置換（人物キーは不変）
                            rekey(*d["rename"])
                        add_sys(cur_ms, f"この声を「{d['name']}」として追跡開始"
                                        f"（実名にするには {d['label']}=名前）")
                        print_line(f"# この声を「{d['name']}」として追跡します"
                                   f"（実名にするには {d['label']}=名前 と入力）")
                    elif args.vp_debug and d:
                        extra = f" 類似{d['sim']:.2f}({d['name']})" if "sim" in d else ""
                        print_line(f"# vp判定[{d['kind']}]{extra}")
                else:
                    sp_id = "#" + str(cur_speaker)
                    rec_extra = {}
                if cur_ms is not None and cur_end is not None:
                    recent_segs.append((cur_ms, cur_end, label))
                    del recent_segs[:-12]
                if tracker is not None and tracker.last is not None:
                    # 診断ログ: 後から「いつ・なぜ判定が崩れたか」を解析するための1行JSON
                    try:
                        with open(diag_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"ms": cur_ms, "end": cur_end, "label": label,
                                                "key": sp_id, **tracker.last},
                                               ensure_ascii=False, default=str) + "\n")
                    except OSError:
                        pass
                with state_lock:   # colorsの変更も保存処理との競合を避けるためロック内で
                    records.append({"ms": cur_ms, "speaker": sp_id, "text": cur_text.strip(),
                                    **rec_extra})
                    c = color_of(sp_id)
                if ON_UTTERANCE is not None:
                    try:
                        ON_UTTERANCE(disp_name(sp_id), cur_text.strip())
                    except Exception:
                        pass   # das側の例外で文字起こしを止めない
                print_line(f"{c}[{fmt_ts(cur_ms)}] {disp_name(sp_id)}{RESET}: {cur_text.strip()}")
                save()
            cur_text = ""
            cur_ms = None
            cur_end = None

        try:
            while True:
                res = json.loads(ws.recv())
                if args.stt == "speechmatics":
                    res = sm_to_res(res, args.lang)
                if res.get("error_code") is not None:
                    print_line(f"# エラー: {res['error_code']} - {res.get('error_message')}")
                    break
                partial = ""
                partial_sp = cur_speaker
                for token in res.get("tokens", []):
                    text = token.get("text") or ""
                    if text == "<end>":
                        flush()
                        continue
                    if not text:
                        continue
                    if token.get("is_final"):
                        sp = token.get("speaker")
                        if sp != cur_speaker:
                            flush()
                            cur_speaker = sp
                        if cur_ms is None:
                            cur_ms = token.get("start_ms")
                        if token.get("end_ms") is not None:
                            cur_end = token["end_ms"]
                        cur_text += text
                    else:
                        partial += text
                        partial_sp = token.get("speaker") or partial_sp
                show_partial(partial_sp if partial else cur_speaker, cur_text + partial)
                if res.get("finished"):
                    flush()
                    print_line("# 終了")
                    break
        except KeyboardInterrupt:
            pass
        finally:
            globals()["_SYS_HOOK"] = None
            stop.set()
            flush()
            save(live=False)
            if tracker is not None:
                print_line(f"# レイテンシ統計: {tracker.stats()}")
            print_line(f"# 議事録を保存しました: {out_path} / {html_path}")
            if len(pcm_buf) > SR * 2 * 10:
                # 録音を保存（清書の再実験・診断用。*.wavはgitignore済み）
                wav_path = os.path.splitext(out_path)[0] + ".wav"
                try:
                    with open(wav_path, "wb") as f:
                        f.write(_wav_bytes(bytes(pcm_buf)))
                    print_line(f"# 録音を保存しました: {wav_path}")
                except OSError as e:
                    print_line(f"# 録音保存に失敗: {e}")
            # 清書: RT分離は高速応酬で崩れる(実測)ため、全文脈の非同期再処理で最終版を作る
            if not args.no_polish and not api_key and len(pcm_buf) > SR * 2 * 10:
                print_line("# 清書はスキップ（SONIOX_API_KEY未設定。清書はSoniox非同期APIを使用）")
            if not args.no_polish and api_key and len(pcm_buf) > SR * 2 * 10:
                try:
                    recs = polish(api_key, bytes(pcm_buf), args.lang, tracker, log=print_line)
                    fmd = os.path.splitext(out_path)[0] + ".final.md"
                    fht = os.path.splitext(out_path)[0] + ".final.html"
                    write_md(recs, fmd)
                    write_html(live=False, recs=recs, path=fht, status="清書（非同期再処理済み）")
                    write_turns(recs, os.path.splitext(out_path)[0] + ".final.turns.jsonl")
                    print_line(f"# 清書版を保存しました: {fmd} / {fht}")
                    if not args.no_open:
                        import webbrowser
                        webbrowser.open("file://" + os.path.abspath(fht))
                except KeyboardInterrupt:
                    print_line("# 清書をスキップしました")
                except Exception as e:
                    print_line(f"# 清書に失敗しました: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
