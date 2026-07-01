"""声紋プロファイル照合による話者特定."""
from __future__ import annotations

import json
import os
import re
import threading
import time
import unicodedata
from difflib import SequenceMatcher
from typing import ClassVar

import numpy as np

from ._constants import SR, UNSURE_SPEAKER

# ---------- リサンプル（AI声紋登録用） ----------


def _resample_24_to_16(pcm_24k: np.ndarray) -> np.ndarray:
    """24kHz float32 → 16kHz float32 リサンプル（線形補間）。AI声紋登録用。"""
    n_in = len(pcm_24k)
    n_out = int(n_in * 2 / 3)
    if n_out < 2:
        return np.empty(0, dtype=np.float32)
    idx = np.linspace(0, n_in - 1, n_out)
    return np.interp(idx, np.arange(n_in), pcm_24k).astype(np.float32)


# ---------- テキスト類似度（エコー判定用） ----------


def _normalize_text(text: str) -> str:
    """テキスト比較用の正規化: 句読点・空白・記号を除去。"""
    t = unicodedata.normalize("NFKC", text)
    return re.sub(r'[\s　、。,.!?！？「」『』（）()・…\-―ー～~]+', '', t)


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    """文字n-gramの集合を返す。"""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def _best_text_similarity(text: str, recent_texts: list[str],
                          streaming_buf: str = "") -> float:
    """正規化テキストとAI生成テキスト群の最大類似度を返す（0.0〜1.0）。

    完了済みの応答(recent_texts)に加え、現在ストリーミング中の応答
    (streaming_buf)も比較対象に含める。部分包含 + SequenceMatcher
    + trigram jaccard + coverage の最大値。
    """
    if not text:
        return 0.0
    targets = list(recent_texts)
    if streaming_buf:
        targets.append(streaming_buf)
    if not targets:
        return 0.0
    norm = _normalize_text(text)
    if len(norm) < 2:
        return 0.0
    best = 0.0
    for ai_text in targets:
        ai_norm = _normalize_text(ai_text)
        if len(norm) >= 4 and norm in ai_norm:
            return 1.0
        if len(ai_norm) >= 4 and ai_norm in norm:
            return 1.0
        sm = SequenceMatcher(None, norm, ai_norm).ratio()
        ng_a = _char_ngrams(norm)
        ng_b = _char_ngrams(ai_norm)
        jaccard = len(ng_a & ng_b) / max(len(ng_a | ng_b), 1)
        coverage = len(ng_a & ng_b) / max(len(ng_a), 1)
        best = max(best, sm, jaccard, coverage)
    return best


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
    DEFAULTS: ClassVar[dict] = {"resemblyzer": (0.75, 0.72, 0.62), "ecapa": (0.35, 0.40, 0.30),
                                "redimnet": (0.42, 0.50, 0.34)}
    # AI声紋判定用の閾値（人間より高め — TTS音声はスピーカー経由でも特徴が明瞭）
    AI_THRESH: ClassVar[dict] = {"resemblyzer": 0.80, "ecapa": 0.42, "redimnet": 0.50}

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
            import torch  # 初回はGitHubからコード＋重み(20MB)をダウンロード
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
        # 短い発話の取り違え安定化（2人会話の短いラリー対策）。min_sec 未満でも、
        # 既知の2人以上を「厳格に」区別できるときだけ声紋で割り当てを正す。
        self.short_floor = 0.45        # 厳格照合する下限秒数（これ未満は声が短すぎるので追従）
        self.short_bonus = 0.05        # 採用閾値を本人閾値からどれだけ引き上げるか
        self.short_margin_mult = 2.0   # 要求する2位とのmargin倍率
        # 新規人物の自動登録は「発話数」ではなく「声ごとのクリーンな発声の累積文字数」で
        # 判定する。声紋の質は本質的に発声の総量で決まり、文字数はその良い代理（長いだけで
        # 無音・雑音の区間を弾く）。連続して長く話す人も、短く何度も話す人も同じ原理で確定。
        self.enroll_min_total_chars = 45  # 一貫クラスタの累積文字数がこれを超えたら登録
        self.enroll_win_sec = 1.5         # 長い発話を分割する窓の長さ（内部一貫性の確認用）
        self.enroll_consist_bonus = 0.08  # 一貫性しきい値(cs)への上乗せ（混入抑制）
        self._POOL_CAP = 24               # 保留サンプルの上限（古いものから捨てる）
        self.max_human_speakers: int | None = None
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
        # 未確定の声の保留プール（ラベルで仕切らない）。Sonioxは新しい声を既存ラベルに混ぜて
        # 出すことがあるので、声は声同士で束ねる。各サンプルは (声紋, 文字数) を持ち、
        # 一貫したクラスタの累積文字数が閾値に達した時点で人物として確定する。
        self.pool: list[tuple[np.ndarray, float]] = []
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
        # プロファイル選択: セッション中に照合対象とするプロファイルのキー集合。
        # voices.jsonから読んだ名前付きプロファイルは全て非アクティブで開始し、
        # ユーザーが明示的にONにしたもの＋セッション中に自動登録された人物Nのみが照合対象。
        self._active_keys: set[str] = set()

    def reset_session(self) -> None:
        """会議リセット時に、人間話者の割り当て・蓄積をクリアする（課題③）.

        新しい会議を素の状態から始められるように、Sonioxラベルの割り当て・
        未確定プール・人物別履歴をクリアし、照合対象（アクティブ）も外す。
        ただし AI声紋(__AI__/__PARTNER__)はエコー除去に使うため維持する。
        voices.json（永続化ファイル）は変更しない（永続化は別機能）。
        """
        with self._lock:
            self.sp_map.clear()
            self.label_embs.clear()
            self.pool.clear()
            self.n_anon = 0
            self.own_sims.clear()
            self.last = None
            # 人間話者は照合対象から外す。AI声紋(__..__)だけ残す。
            self._active_keys = {k for k in self._active_keys
                                 if k.startswith("__") and k.endswith("__")}

    def _active_human(self) -> dict:
        """照合対象の人間プロファイル（AI声紋 __..__ は除く）."""
        ai = {k for k in self._active_keys if k.startswith("__") and k.endswith("__")}
        return {k: v for k, v in self.profiles.items()
                if k in self._active_keys and k not in ai}

    def _rank_active(self, emb: np.ndarray, active: dict):
        """active内で最も似た人物の (cand, sim, second) を返す（空ならNone）."""
        if not active:
            return None
        ranked = sorted(((float(np.dot(p, emb)), n) for n, p in active.items()),
                        reverse=True)
        sim, cand = ranked[0]
        second = ranked[1][0] if len(ranked) > 1 else -1.0
        return cand, sim, second

    def _ai_echo(self, emb: np.ndarray, active: dict):
        """AI声紋(エコー)に一致すれば (キー, sim) を返す（無ければNone, 人間より高い閾値）."""
        ai_profs = {k: self.profiles[k] for k in self._active_keys
                    if k.startswith("__") and k.endswith("__") and k in self.profiles}
        if not ai_profs:
            return None
        best_human = max((float(np.dot(p, emb)) for p in active.values()), default=-1.0)
        for ai_key, ai_prof in ai_profs.items():
            ai_th = self.AI_THRESH.get(self.model, self.thresh + 0.10)
            ai_sim = float(np.dot(ai_prof, emb))
            if ai_sim >= ai_th and ai_sim > best_human:
                return ai_key, ai_sim
        return None

    # ------------------------------------------------------------------
    # 自動登録（累積文字数ベース）
    # ------------------------------------------------------------------
    def _weighted_mean(self, embs: list[np.ndarray], weights: list[float]) -> np.ndarray:
        """文字数で重み付けした平均声紋（長い＝信頼できるサンプルを重視）."""
        w = np.asarray(weights, dtype=np.float64)
        if w.sum() <= 0:
            w = np.ones(len(embs))
        prof = np.average(np.stack(embs), axis=0, weights=w)
        return prof / np.linalg.norm(prof)

    def _segment_samples(self, wav: np.ndarray, emb: np.ndarray,
                         chars: float) -> list[tuple[np.ndarray, float]]:
        """登録用サンプル列 [(声紋, 文字数), ...] を返す.

        長い発話は窓に分割して複数サンプルにし、文字数を比例配分する。これにより
        連続して長く話す人でも声紋サンプルが増え（＝登録が進む）、同時に窓どうしの
        一貫性で「実は2人が混ざった1ターン」を弾ける。短い発話はそのまま1サンプル。
        """
        n = int((wav.size / SR) // self.enroll_win_sec)
        if n <= 1:
            return [(emb, float(chars))]
        n = min(n, 6)   # 過剰な埋め込み計算を抑制
        win = wav.size // n
        samples: list[tuple[np.ndarray, float]] = []
        for i in range(n):
            e = self._embed(wav[i * win:(i + 1) * win])
            if e is not None:
                samples.append((e, chars / n))
        return samples or [(emb, float(chars))]

    def _enroll_accumulate(self, samples: list[tuple[np.ndarray, float]],
                           sp: str, prev, ecs: float) -> str | None:
        """サンプルを声ごとに貯め、一貫クラスタの累積文字数が閾値を超えたら登録.

        現在の声(anchor)と一貫する保留サンプルを集め、その累積文字数が閾値に達したら
        その人物を確定する。戻り値: 確定した人物キー、まだ足りなければ None。
        """
        self.pool.extend(samples)
        if len(self.pool) > self._POOL_CAP:
            del self.pool[:-self._POOL_CAP]
        anchor = samples[-1][0]
        idx = [i for i, (e, _c) in enumerate(self.pool)
               if float(np.dot(e, anchor)) >= ecs]
        total = sum(self.pool[i][1] for i in idx)
        if total < self.enroll_min_total_chars:
            return None
        embs = [self.pool[i][0] for i in idx]
        wts = [self.pool[i][1] for i in idx]
        prof = self._weighted_mean(embs, wts)
        drop = set(idx)
        self.pool = [s for i, s in enumerate(self.pool) if i not in drop]
        return self._commit_profile(prof, sp, prev, total)

    def _commit_profile(self, prof: np.ndarray, sp: str, prev,
                        total_chars: float) -> str:
        """クラスタの代表声紋を、既存人物に合流 or 新規人物として確定する."""
        active = self._active_human()
        hit_sim, hit = max(((float(np.dot(p, prof)), n) for n, p in active.items()),
                           default=(-1.0, None))
        if hit is not None and hit_sim >= self.dedupe:
            target, is_new = hit, False   # 既存人物の声だった → 合流（重複登録を防ぐ）
        else:
            if self.max_human_speakers is not None and len(active) >= self.max_human_speakers:
                self.sp_map[sp] = UNSURE_SPEAKER
                self._note("話者数上限", label=sp, max_speakers=self.max_human_speakers,
                           chars=round(total_chars))
                return UNSURE_SPEAKER
            self.n_anon += 1
            target = f"人物{self.n_anon}"
            self.profiles[target] = prof   # 新規人物（以後凍結）
            self._active_keys.add(target)
            is_new = True
        # 遡及置換は未確定キー(#ラベル)の昇格のみ。人物キーは絶対に書き換えない。
        rename = ("#" + sp, target) if (prev is None or prev.startswith("#")) else None
        self.sp_map[sp] = target
        self._note("自動登録" if is_new else "合流", label=sp, name=target,
                   rename=rename, chars=round(total_chars))
        return target

    def set_max_human_speakers(self, value: int | None) -> None:
        self.max_human_speakers = value

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

    def classify(self, wav: np.ndarray, sp, overlapped: bool = False,
                 count: bool = True, chars: int = 0) -> str:
        """発話を人物キーに割り当てる（経路はクラスドキュメント参照）.

        overlapped=True の発話は声が混ざっていて声紋がデタラメになるため、
        声での判定をスキップして直前の対応を維持する。
        count=False（相槌など）の発話は声紋の蓄積・人物登録に使わず、
        既存の割り当てに追従するだけにする（課題④）。
        chars はその発話の文字数。新規人物の自動登録は、声ごとにこの文字数を
        累積し、一貫したクラスタが閾値を超えた時点で確定する（発話数では数えない）。
        """
        with self._lock:
            return self._classify(wav, sp, overlapped, count, chars)

    def _classify(self, wav: np.ndarray, sp, overlapped: bool,
                  count: bool = True, chars: int = 0) -> str:
        sp = str(sp)
        prev = self.sp_map.get(sp)
        kind, info = "相槌追従", {}
        if overlapped and wav.size >= SR * self.min_sec:
            kind = "重なりスキップ"
        elif count and wav.size >= SR * self.min_sec:
            emb = self._embed(wav)
            if emb is None:
                kind = "声紋計算不可"
            else:
                self._update_room_stats(sp, emb)   # 部屋の同一/別人分布を実測(表示・診断用)
                self.label_embs.setdefault(sp, []).append(emb)
                del self.label_embs[sp][:-10]    # 手動登録用に直近10発話だけ保持
                th, cs = self.thresh, self.consist
                active = self._active_human()
                info = {"n_prof": len(active), "n_all": len(self.profiles)}   # 診断ログ用
                # ① AI声紋の先行チェック（エコー除去用 — 人間より高い閾値）
                ai = self._ai_echo(emb, active)
                if ai is not None:
                    self.sp_map[sp] = ai[0]
                    self._note("AI声紋一致", label=sp, sim=round(ai[1], 3), key=ai[0])
                    return ai[0]
                # ② 通常の話者照合（人間のプロファイルのみ）
                ranked = self._rank_active(emb, active)
                if ranked is not None:
                    cand, sim, second = ranked
                    info.update(sim=round(sim, 3), second=round(second, 3), name=cand, prev=prev)
                    if sim >= self._person_th(cand, th) and sim - second >= self.margin:
                        self.sp_map[sp] = cand
                        h = self.own_sims.setdefault(cand, [])
                        h.append(sim)
                        del h[:-20]
                        self._note("補正" if (prev is not None and not prev.startswith("#")
                                              and prev != cand) else "声紋一致", label=sp, **info)
                        return cand
                # 既知の誰にも確信を持って一致しなかった。直前が「確定済みの人」でも、
                # 声紋がその人と一致しないなら追従せず「未確定」に落とす。これにより
                # 新規話者の登録直前の発話が登録済みの人として表示されるのを防ぎ、
                # 登録時の遡及リネーム(#ラベル→人物N)で後からまとめて確定できる。
                if prev is not None and not prev.startswith("#"):
                    pv = active.get(prev)
                    prev_sim = float(np.dot(pv, emb)) if pv is not None else -1.0
                    continuity_th = max(0.25, self._person_th(prev, th) - 0.12)
                    if self.ANON.match(prev) and prev_sim >= continuity_th:
                        self._note("低信頼追従", label=sp, sim=round(prev_sim, 3),
                                   name=prev, prev=prev)
                        return prev
                    if pv is None or prev_sim < self._person_th(prev, th):
                        prev = None
                # 登録: 発話数ではなく「声ごとのクリーンな発声の累積文字数」で確定する。
                # 長い発話は窓分割して複数サンプル化（連続発話でも登録が進み、内部一貫性も確認）。
                kind = "蓄積中" if (self.auto and chars > 0) else "未確定"
                if self.auto and chars > 0:
                    ecs = cs + self.enroll_consist_bonus
                    samples = self._segment_samples(wav, emb, chars)
                    target = self._enroll_accumulate(samples, sp, prev, ecs)
                    if target is not None:
                        return target
        elif count and wav.size >= SR * self.short_floor:
            # 短い発話の取り違え安定化: 既知の2人以上を厳格に区別できるときだけ正す。
            # 登録・蓄積はしない（声が短く不安定なため、登録に混ぜると精度が落ちる）。
            active = self._active_human()
            if len(active) >= 2:
                emb = self._embed(wav)
                ranked = self._rank_active(emb, active) if emb is not None else None
                if ranked is not None:
                    cand, sim, second = ranked
                    strict_th = self._person_th(cand, self.thresh) + self.short_bonus
                    if (sim >= strict_th
                            and sim - second >= self.margin * self.short_margin_mult):
                        self.sp_map[sp] = cand
                        self._note("補正" if (prev is not None and not prev.startswith("#")
                                              and prev != cand) else "声紋一致",
                                   label=sp, sim=round(sim, 3), second=round(second, 3),
                                   name=cand, prev=prev, short=True)
                        return cand
                # 厳格に決められない短い発話は、確定済みの人へ誤って追従せず未確定にする。
                # 開いた名簿でも閉じた名簿でも同じ扱い（証拠の無い決めつけ＝誤った確信付与を
                # 避ける）。sp_map は保持し、次の確信ある発話で連続性を回復する。
                if prev is not None and not prev.startswith("#"):
                    self._note("未確定", label=sp, prev=prev, short=True)
                    return UNSURE_SPEAKER
        # 声紋で決められない（重なり/短い相槌/蓄積中）→ ラベルの直近判定に追従
        key = prev if prev is not None else "#" + sp
        # 閉じた名簿（名簿を確定, auto=False）: 許すのは「登録済みのアクティブな名前付き
        # プロファイルへの継続」だけ。未知/匿名(#ラベル・人物N)は新規参加者を作らず未確定に
        # する。重なり発話は声紋を信用できないので、登録者継続もさせない。
        # 確信ある一致・AI声紋・登録者への証拠つき継続は上流で既に return 済み。
        if not self.auto:
            active = self._active_human()
            if kind == "重なりスキップ" or key not in active or self.ANON.match(str(key)):
                key = UNSURE_SPEAKER
                kind = "未確定"
        self.sp_map[sp] = key
        self._note(kind, label=sp, **info)
        return key

    def enroll(self, label: str, name: str) -> str | None:
        """「1=名前」「人物2=名前」: 話者に名前を付ける（声の登録 or 既存人物のリネーム）.

        実名を付けたプロファイルのみ voices.json に永続化される（匿名「人物N」は
        そのセッション限り）。戻り値: 旧表示キー（過去のrecords付け替え用）。
        十分な音声がまだ無ければ None。
        """
        with self._lock:
            return self._enroll(str(label), name)

    def _enroll(self, label: str, name: str) -> str | None:
        name = str(name).strip()
        if not name:
            self._note("登録失敗", reason="empty_name")
            return None
        if label in self.profiles:
            if name in self.profiles and name != label:
                self._note("登録失敗", label=label, name=name, reason="duplicate_name")
                return None
            # 「人物1=名前」: 既存プロファイルのリネーム
            self.profiles[name] = self.profiles.pop(label)
            old = label
        else:
            cur = self.sp_map.get(label)
            if cur is not None and cur in self.profiles:
                if name in self.profiles and name != cur:
                    self._note("登録失敗", label=label, name=name, reason="duplicate_name")
                    return None
                # ラベルが（自動登録済みの）人物に対応済み → その人物に命名
                self.profiles[name] = self.profiles.pop(cur)
                old = cur
            else:
                if name in self.profiles:
                    self._note("登録失敗", label=label, name=name, reason="duplicate_name")
                    return None
                # ラベルの直近声紋から新規登録
                embs = self.label_embs.get(label)
                if not embs:
                    self._note("登録失敗", label=label, name=name, reason="insufficient_audio")
                    return None
                prof = np.mean(embs, axis=0)
                self.profiles[name] = prof / np.linalg.norm(prof)
                old = cur if cur is not None else "#" + label
        # _active_keysの更新（旧キーが有効だったら新キーに引き継ぐ）
        if old in self._active_keys:
            self._active_keys.discard(old)
            self._active_keys.add(name)
        else:
            self._active_keys.add(name)   # 新規命名は自動的にアクティブ
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
            self._active_keys.discard(src)
            self._persist()
            return True

    def activate(self, name: str) -> str | None:
        """プロファイルをこのセッションで有効化する.

        有効化されたプロファイルは _classify() の照合対象になる。
        既にセッション中に自動登録された人物Nが同一人物だった場合は自動マージし、
        マージされた旧キーを返す（rekey用）。マージなしならNone。
        """
        with self._lock:
            if name not in self.profiles:
                return None
            self._active_keys.add(name)
            prof = self.profiles[name]
            # セッション中の匿名人物Nに同一人物がいたらマージ
            for key in list(self._active_keys):
                if self.ANON.match(key) and key in self.profiles:
                    sim = float(np.dot(prof, self.profiles[key]))
                    if sim >= self.dedupe:
                        self.profiles.pop(key)
                        self._active_keys.discard(key)
                        self.own_sims.pop(key, None)
                        for k, v in list(self.sp_map.items()):
                            if v == key:
                                self.sp_map[k] = name
                        return key   # マージされた旧キー
            return None

    def deactivate(self, name: str) -> None:
        """プロファイルをこのセッションで無効化する（匿名人物Nは無効化不可）."""
        with self._lock:
            if not self.ANON.match(name):
                self._active_keys.discard(name)

    def active_profile_names(self) -> list[str]:
        """現在アクティブな名前付きプロファイルの一覧（UI表示用）."""
        return sorted(k for k in self._active_keys if not self.ANON.match(k) and k in self.profiles)

    def all_profile_names(self) -> list[str]:
        """voices.jsonに保存された全名前付きプロファイル（UI表示用）."""
        return sorted(k for k in self.profiles if not self.ANON.match(k))

    def enroll_from_audio(self, name: str, wav: np.ndarray) -> bool:
        """生の音声からその人の声紋を作って名前付き登録・有効化する（事前登録用）.

        会議前に各人が単独で喋った音声から、2秒窓の平均で頑健な声紋を作る。
        既に同名があれば上書きし、有効化して voices.json に永続化する。
        """
        name = str(name).strip()
        if not name:
            return False
        with self._lock:
            embs = []
            win = int(SR * 2.0)
            for i in range(0, max(wav.size - win + 1, 0), win):
                e = self._embed(wav[i:i + win])
                if e is not None:
                    embs.append(e)
            if not embs:   # 2秒に満たない/窓が取れない → 全体で1つ
                e = self._embed(wav)
                if e is None:
                    return False
                embs = [e]
            prof = np.mean(embs, axis=0)
            self.profiles[name] = prof / np.linalg.norm(prof)
            self._active_keys.add(name)
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
            order = ["声紋一致", "補正", "自動登録", "合流", "蓄積中", "未確定", "相槌追従",
                     "重なりスキップ", "声紋計算不可"]
            parts.append("判定内訳: " + " / ".join(
                f"{k}{self.counts[k]}" for k in order if self.counts.get(k)))
        return "、".join(parts) or "判定なし"
