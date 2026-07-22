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


class _LabelTrustMixin:
    """ラベル信頼度: 「STTラベル→人物」対応をどこまで信じてよいかの判定.

    Sonioxラベルは複数話者を混載し得る（Chiba 0532 実測: 自動登録3人が全て
    同一ラベル発）ため、ラベルに基づく帰属（ラベル継続・#プレースホルダ）と
    学習（蓄積・登録）は、このミックスインの門番を通ったときだけ許す:

      - _label_pure:            直近の照合成功が単一人物に収束しているか（§15.7）
      - _continuation_target:   対応先が実在するアクティブな人間か（AI声紋・
                                deactivate済みの遮断。2026-07-15 レビュー）
      - _trusted_continuation:  上記2つのANDで「ラベル継続」の可否を1回で判定

    状態（label_hist）は VoiceProfiles.__init__ が実体化する（テストの
    __new__ 構築フェイクとの互換のため、このミックスインは状態を持たない）。
    """

    def _continuation_target(self, prev) -> str | None:
        """ラベル継続の対応先として使える人物キーを返す（継続不可なら None）.

        継続を許すのは「現在アクティブな人間プロファイル」への対応だけ
        （#プレースホルダは別経路。docs/design/handoff_2026-07-14_unregistered_speakers.md
        のラベル継続設計を参照）。従来この判定はメインパス（中尺発話）にしか無く、
        短発話・相槌のラベル継続が以下を素通しした（2026-07-15 レビューで確定）:
          - AI声紋キー(__AI__等): sp_map に残った __AI__ を継続で返すと、
            _recv_loop の startswith("__") エコー破棄が発動し、AIエコー直後の
            人間の短発話・相槌が本文ごと消える実バグ。
          - deactivate 済み・remap等で消えた人物、閉じた名簿の非アクティブ登録者:
            実体の無い/照合対象外の対応先へ発話が帰属し続ける。
        3経路（メイン・短発話・相槌）すべてこのヘルパで判定を統一する。
        """
        if prev is None or prev.startswith("#"):
            return None
        return prev if prev in self._active_human() else None

    def _record_label_success(self, sp: str, person: str) -> None:
        """照合成功（一致/補正/登録/合流）のたびにラベル→人物の履歴を残す."""
        if not hasattr(self, "label_hist"):   # __new__ 構築のテスト用フェイク対策
            self.label_hist = {}
        h = self.label_hist.setdefault(str(sp), [])
        h.append(person)
        del h[:-max(int(self.label_purity_window), 8)]

    def _label_pure(self, sp: str) -> bool:
        """ラベルの直近の照合成功が単一人物に収束しているか（継続の門番）.

        不純（直近 window 回に2人以上）なら、そのラベルは複数話者を混載して
        いる可能性が高く、「ラベル継続」は根拠にならない（handoff §15.7）。
        履歴が無い/window=0 の場合は従来どおり健全とみなす。
        """
        w = int(self.label_purity_window)   # replay の sweep は float で渡し得る
        if not w:
            return True
        h = getattr(self, "label_hist", {}).get(str(sp), [])
        return len(set(h[-w:])) <= 1

    def _trusted_continuation(self, sp: str, prev) -> str | None:
        """「ラベル継続」してよい対応先を返す（不可なら None）.

        継続可否は _continuation_target（AI声紋・deactivate済み等の遮断）と
        _label_pure（複数話者混載ラベルの遮断, §15.7）のANDで判定する。
        短発話経路・「照合なし」経路の共通門番。
        """
        cont = self._continuation_target(prev)
        if cont is not None and self._label_pure(sp):
            return cont
        return None


class _ProfileQualityMixin:
    """プロファイル品質: 登録済み声紋の健全性維持と人物別しきい値.

    プロファイルは凍結が原則だが、登録材料の混入（Soniox境界の甘さで1発話に
    両話者が混ざる等）は事前には完全に防げない。ここに集めた3層で品質を守る:

      - _purity_subset:        埋め込み集合の最大自己一貫部分集合（登録時の
                               純度検査と事後検査の共通中核。分布の相対構造のみで
                               判定し、固定しきい値を持たない）
      - _track_own_emb:        受理一致の埋め込みを監視し、二峰性（汚染）を
                               検出したら多数派で再構築する事後回収層（§13.1）
      - _person_th / _record_reference_sim:
                               人物別しきい値（新規性検出）とその学習履歴。
                               記録は person_th と独立な固定基準で行い、
                               自己参照ラチェットを断つ（§13.2）

    状態（own_sims/own_embs/_own_updates）は VoiceProfiles.__init__ が実体化
    する（テストの __new__ 構築フェイクとの互換のため、状態はここに持たない）。
    """

    def _purity_subset(self, embs: list[np.ndarray]) -> list[int]:
        """最大の自己一貫部分集合のインデックスを返す（登録純度検査の中核）.

        medoid（他との類似度合計が最大の埋め込み）を種に、medoidとの類似度分布を
        1次元2クラス分割（クラス間分散最大の割線＝Otsu流）し、下側クラスタの平均が
        上側クラスタの平均の半分未満なら「別の声の混入」とみなして上側だけ返す。
        判定は分布の相対構造のみで行い、固定の類似度しきい値を持たない:
        CallHome 0856 実測（8kHz電話、docs/design/
        handoff_2026-07-14_unregistered_speakers.md §13.1）で話者内類似は
        中央値0.38-0.65、話者間は中央値0.116・最大0.343と、話者間は話者内の
        半分を下回る相対構造が確認されており、絶対値は帯域・マイクで動くが
        この比は保たれる（16kHz会議録の実測でも同様）。単峰（混入なし）の
        集合では下側平均が上側の半分を割らないため全採用のまま素通りする。
        """
        n = len(embs)
        if n < 4:   # 分布を語れるサンプル数がない → 検査せず全採用
            return list(range(n))
        m = np.stack(embs)
        sims = m @ m.T
        med = int(np.argmax(sims.sum(axis=1)))   # medoid（最も「みんなに似ている」声）
        others = [i for i in range(n) if i != med]
        s = np.array([float(sims[med, i]) for i in others])
        order = np.argsort(s)   # medoid類似度の昇順
        sv = s[order]
        best_k, best_score = 0, -1.0
        for k in range(1, len(sv)):   # クラス間分散が最大の割線を探す
            score = k * (len(sv) - k) * (sv[k:].mean() - sv[:k].mean()) ** 2
            if score > best_score:
                best_score, best_k = score, k
        mu_lo = float(sv[:best_k].mean())
        mu_hi = float(sv[best_k:].mean())
        if mu_lo >= 0.5 * mu_hi:   # 下側も「同じ声」の揺らぎの範囲 → 単峰
            return list(range(n))
        return sorted([med] + [others[i] for i in order[best_k:]])

    def _track_own_emb(self, name: str, emb: np.ndarray) -> None:
        """受理された一致の埋め込みを蓄積し、N回ごとに汚染検査→多数派で再構築（P3）.

        登録時の純度検査（_purity_subset）をすり抜けて混合プロファイルができた
        場合、そのプロファイルは両話者の発話を引き寄せるため、受理一致の埋め込み
        列が二峰性を持つ（プール内に相互類似度の低い2クラスタ）。これを検出したら
        多数派クラスタの平均でプロファイルを再構築し、少数派は破棄する。
        検査は _REBUILD_EVERY 回の受理ごと（16×16 の内積行列1回＝サブms級）。
        再構築後は own_sims（人物別しきい値の履歴）もリセットし、新しい
        プロファイルの一致分布を学び直す（汚染期の受理simでしきい値が
        歪んだままになるのを防ぐ）。呼び出し元が classify の受理パス＝
        min_sec 以上の発話のみで、短発話の不安定な埋め込みは混ぜない。
        設計: docs/design/handoff_2026-07-14_unregistered_speakers.md §13.1 の
        当初分析（CallHome 0856、登録材料の53-55%混合で帰属27%に崩壊）への
        事後回収層として導入。なお §13.2 の再現では登録汚染は 0856 の主因では
        なかった（自動登録プロファイルは純粋）と判明済みで、本機構は「将来条件で
        効く汚染への保険」として維持されている（同節の修正記録参照）。
        """
        if not self.ANON.match(name):
            return   # 実名プロファイルは書き換えない（__init__ の own_embs 註釈参照）
        oe = self.own_embs.setdefault(name, [])
        oe.append(emb)
        del oe[:-self._OWN_EMB_CAP]
        cnt = self._own_updates.get(name, 0) + 1
        self._own_updates[name] = cnt
        if cnt % self._REBUILD_EVERY or len(oe) < self._REBUILD_EVERY:
            return
        keep = self._purity_subset(oe)
        # 単峰（健全）なら何もしない。二峰でも「多数派」と呼べる側が過半数に
        # 届かなければ、どちらが本人か判定できないので書き換えない（安全側）。
        if len(keep) >= len(oe) or len(keep) * 2 <= len(oe):
            return
        embs = [oe[i] for i in keep]
        prof = np.mean(np.stack(embs), axis=0)
        self.profiles[name] = prof / np.linalg.norm(prof)
        self.own_embs[name] = embs
        self.own_sims[name] = []
        # 直近判定(last)は呼び出し元の一致noteに使われるため上書きせず、
        # counts のみ記録する（diag の stats 経由で観測可能）。
        self.counts["プロファイル再構築"] = self.counts.get("プロファイル再構築", 0) + 1

    def _person_th(self, name: str, base: float) -> float:
        """人物別しきい値 = max(基準値, その人物の一致sim下位35パーセンタイル - 0.12).

        旧仕様は中央値-0.12。受理simのみの履歴（選択バイアス）と組み合わさると
        「しきい値上昇→低め一致が記録されない→中央値上昇→さらに上昇」の
        ラチェットになった（CallHome 0856 実測で 0.42→0.73 まで肥大し、
        正しい 0.5-0.65 帯の一致を全遮断）。対策は二本立て:
          1. 記録側: person_th と独立な基準（base+margin通過の生sim）で記録し
             自己参照を断つ（own_sims の __init__ 註釈参照）
          2. 統計側: 中央値でなく下位35パーセンタイルを使う。本人のsimは帯域・
             マイクで絶対値が動く（8kHz電話 0.5-0.65 / 16kHz会議 0.67-0.82）ため
             固定の上限は置けないが、下位分位なら「本人の一致の下端」に追従し、
             分布の裾の揺らぎや別人混入の高値外れに引きずられにくい
        いずれも相対設計（観測simの分位）なので 8k/16k どちらでも壊れない。
        巻き取り防止の本来機能は維持: 本人が安定して高sim（例 0.7台）を出す
        環境では p35-0.12 ≈ 0.58-0.61 となり、別人の 0.5 前後の中途半端な
        類似は引き続き弾く。
        分位の選定はオフライン再現4本の実測（2026-07-15）: p25 は両話者のsim帯が
        重なる電話ペアで巻き取りが再発（0743: 62%→60.2%）、中央値はラチェットは
        直っても 0856 の回復が鈍い（29%→30.5%）。p35 は 0856=32.9% / 0743=63.4%
        / 0696=71.6% / YouTube 16kHz=78.9% と全データセットで基準線以上。
        """
        h = self.own_sims.get(name, [])
        if len(h) >= 3:
            return max(base, float(np.percentile(h, 35)) - self.person_th_offset)
        return base

    def _record_reference_sim(self, name: str, sim: float) -> None:
        """人物別しきい値の学習履歴に、固定基準を通過した1位simを記録する.

        記録は person_th 判定の手前（基準しきい値＋margin という固定条件のみ）
        で行う。受理後に記録すると person_th 自身が記録条件に入り、選択バイアス
        でしきい値がラチェットする（_person_th docstring 参照）。
        """
        h = self.own_sims.setdefault(name, [])
        h.append(sim)
        del h[:-20]


class VoiceProfiles(_LabelTrustMixin, _ProfileQualityMixin):
    """凍結プロファイル照合による話者特定（台帳固定・誤り非伝播）.

    判定は2経路だけ:
      ① 即時判定 — 単発声紋が強一致(thresh＋2位とmargin差)した時だけ、その場で人物確定
      ② それ以外は3発話バッファ — 一貫した3発話を束ね「既存人物に合流(dedupe) or 新規人物N」
    しきい値は2層構造（厳しくする方向にのみ働き、最悪でも既定値の挙動に戻る）:
      1. モデル別既定値(DEFAULTS)
      2. 人物別しきい値(その人物の一致sim下位35パーセンタイル-0.12 = 新規性検出。
         中途半端な類似の新しい声を既存人物に巻き取らない)。即時判定のみに適用
    不変条件: 確定済みの人物キーは書き換えない（遡及置換は #ラベル→人物 の昇格のみ）。
    実名(enroll)のみ voices.json に永続化、匿名「人物N」はセッション限り。
    """

    ANON = re.compile(r"^人物\d+$")

    # ハイブリッド構成（pyannoteクラスタ×声紋照合, ClusterVoiceNamer有効）フラグ。
    # クラス属性の既定値 False により、Soniox単独・pyannote単独の既存挙動は不変。
    # True のとき短発話(short_floor〜min_sec)の厳格声紋照合を「既知1人」でも試みる
    # （通常は取り違え防止のため既知2人以上が条件）。実測（transcripts/2026-07-14_1729,
    # GT81発話）で声紋一致92%(n=13)に対し前話者追従は28%(n=32)と、3人会話では
    # 追従が害になるため、当たる機構＝声紋照合の射程を短発話にも広げる。
    # 照合しきい値は既存の短発話用厳格運用（本人しきい値+short_bonus）のまま
    # ＝当たりにくいだけで誤りは増やさない。
    hybrid = False

    # 厳格照合を要求する発話長の上限秒数（クラス既定値。__init__ 経由でインスタンス化）。
    # 3.0→2.0（2026-07-22, handoff §18.10）: 14データセットの開発/検証分割sweepで
    # 2.0 が検証側でも 実質+1.4pt/誤帰属-1.2pt（14中11改善・悪化1、0696は+15.8pt）。
    # 当初の 3.0 は単一録音の観察由来の手置き値で、2〜3秒帯に不要な厳格化を課していた。
    strict_sec = 2.0

    # ラベル継続の健全性窓（クラス既定値。0で無効=旧挙動。テスト用フェイクが
    # __new__ 構築でも動くよう、strict_sec と同じくクラス既定値を持つ）。
    label_purity_window = 4

    # 人物別しきい値のオフセット（_person_th: p35 - この値）。分位点 p35 は
    # sweep 実測で選定済みだが、このオフセット自体は旧仕様（中央値-0.12）からの
    # 流用で未検証（attribution_selfreview_2026-07-21.md）。sweep 可能にするため
    # 属性化（既定 0.12 ＝従来と同一挙動）。
    person_th_offset = 0.12

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
        self.short_bonus = 0.08        # 採用閾値を本人閾値からどれだけ引き上げるか
        # （0.05→0.08, 2026-07-14: 2.0s発話の誤一致 sim=0.49 が本人しきい値0.42+0.05
        #   をすり抜けラベルの人物対応を破壊。正しい一致は全て0.60以上で影響なし）
        # 旧 short_margin_mult（margin を strict帯で2倍要求）は削除（2026-07-16）。
        # ablation（docs/design/attribution_logic_review_2026-07.md §3.2）で
        # 無効化しても出力ビット単位一致＝厳格化の実効成分は short_bonus のみと
        # 実証されたため、「strict_sec 未満は採用閾値に +short_bonus」に一本化。
        # margin（2位差 0.05）は全帯共通の保険として1本だけ残す。
        # 厳格照合を要求する上限秒数。min_sec〜strict_sec の中尺発話も、短発話と同じ
        # 厳格条件（+short_bonus）でしか即時判定しない。
        # 当初の実測(2026-07-14_142016): 1〜2.5秒帯の誤一致（sim0.43-0.49）を排除する
        # ため 3.0 を採用。その後 14データセットsweep（§18.10）で 2.0 に更新
        # （2〜3秒帯は基準しきい値で信じてよいと判明。誤帰属も悪化しない）。
        # 埋め込みの信頼性は発話長に依存するため、基準しきい値は strict_sec 以上のみ。
        self.strict_sec = self.strict_sec   # クラス既定値を実体化（テスト・調整用）
        # ハイブリッドフラグもインスタンス属性として実体化（クラス属性の既定値 False
        # を影にしない明示。set_hybrid がクラス属性を汚染しないことの保証を
        # インスタンス側で完結させる。2026-07-15 レビュー F8）。
        self.hybrid = self.hybrid
        # 新規人物の自動登録は「発話数」ではなく「声ごとのクリーンな発声の累積文字数」で
        # 判定する。声紋の質は本質的に発声の総量で決まり、文字数はその良い代理（長いだけで
        # 無音・雑音の区間を弾く）。連続して長く話す人も、短く何度も話す人も同じ原理で確定。
        self.enroll_min_total_chars = 45  # 一貫クラスタの累積文字数がこれを超えたら登録
        self.enroll_win_sec = 1.5         # 長い発話を分割する窓の長さ（内部一貫性の確認用）
        self.enroll_consist_bonus = 0.08  # 一貫性しきい値(cs)への上乗せ（混入抑制）
        self._POOL_CAP = 24               # 保留サンプルの上限（古いものから捨てる）
        self._REBUILD_EVERY = 8           # 受理一致N回ごとに汚染検査（P3・低頻度＝低コスト）
        self._OWN_EMB_CAP = 16            # 人物ごとに保持する受理一致埋め込みの上限（P3）
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
        # ラベル健全性の履歴: Sonioxラベル -> 直近の照合成功（一致/補正/登録/合流）
        # で確定した人物のリスト。高重なり会話ではSonioxが複数話者を同一ラベルに
        # 混ぜることがあり（Chiba 0532 実測: 自動登録3人が全て同一ラベル発、
        # ラベル継続の正解率22%）、その場合「ラベル継続」は誤帰属を量産する。
        # 直近 label_purity_window 回の成功が単一人物でないラベルは「不純」と
        # みなし、継続を未確定に落とす（handoff §15.7）。0 で無効（従来挙動）。
        self.label_hist: dict[str, list[str]] = {}
        self.label_purity_window = self.label_purity_window  # クラス既定値を実体化
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
        # 履歴は「基準しきい値＋marginを満たした1位候補の生sim」を記録する
        # （person_th 判定の手前で記録）。旧仕様の「受理された一致のみ記録」は、
        # 記録条件に person_th 自身が入る自己参照＝選択バイアスで、しきい値が
        # 単調に肥大するラチェットを生んだ（CallHome 0856 実測: 0.42→0.73、
        # 正しい0.5-0.65帯の一致を全遮断し帰属29%。person_th 無効化で32.5%）。
        # 記録条件を person_th と独立な固定基準にすることで分布が定常になり、
        # ラチェットは構造的に起きない。
        self.own_sims: dict[str, list[float]] = {}   # 人物 -> 基準通過した1位simの履歴
        # 事後回収（P3）: 匿名人物の受理一致の埋め込みを保持し、更新N回ごとに
        # 自己一貫性を検査。二峰性（登録が汚染され、プロファイルが両話者の一致を
        # 引き寄せている状態）を検出したら多数派クラスタで再構築する。
        # 対象は自動登録の匿名「人物N」のみ: 実名プロファイルは単独発話の
        # クリーンな音声から作られ（enroll_from_audio/enroll）、voices.json に
        # 永続化もされるため、セッション中の照合履歴で書き換えない（安全側）。
        # 過去レコードの遡及修正はしない（将来の帰属が直ればよい）。
        self.own_embs: dict[str, list[np.ndarray]] = {}   # 人物 -> 受理一致の埋め込み
        self._own_updates: dict[str, int] = {}            # 人物 -> 受理一致の累計回数
        self.embed_ms: list[float] = []                     # レイテンシ統計
        self.counts: dict[str, int] = {}                    # 判定種別の集計
        self.last: dict | None = None                       # 直近の判定内容（可視化用）
        self._lock = threading.RLock()   # classify(受信スレッド)とenroll/remap(入力スレッド)の排他
        # プロファイル選択: セッション中に照合対象とするプロファイルのキー集合。
        # voices.jsonから読んだ名前付きプロファイルは全て非アクティブで開始し、
        # ユーザーが明示的にONにしたもの＋セッション中に自動登録された人物Nのみが照合対象。
        self._active_keys: set[str] = set()

    def reset_session(self) -> None:
        """会議リセット時に、セッション由来の割り当て・蓄積をクリアする（課題③）.

        新しい会議を素の状態から始められるように、Sonioxラベルの割り当て・
        未確定プール・人物別履歴をクリアする。ただし照合対象（アクティブ）は、
        ユーザーが有効化した実名プロファイルを次の会議へ引き継ぐ（同じ参加者で
        会議を続けるのが通常のため、リセットのたびに全員が未確定へ落ちるのを防ぐ）。
        セッション限りの匿名「人物N」だけを非活性化する。
        AI声紋(__AI__/__PARTNER__)はエコー除去に使うため維持する。
        voices.json（永続化ファイル）は変更しない（永続化は別機能）。
        """
        with self._lock:
            self.sp_map.clear()
            getattr(self, "label_hist", {}).clear()
            self.label_embs.clear()
            self.pool.clear()
            self.n_anon = 0
            self.own_sims.clear()
            self.own_embs.clear()
            self._own_updates.clear()
            self.last = None
            # AI声紋(__..__)と実名プロファイルは残し、匿名「人物N」だけ落とす。
            # ANON は「人物N」のみをカバーする（#ラベルは _active_keys に入らない）。
            self._active_keys = {k for k in self._active_keys
                                 if (k.startswith("__") and k.endswith("__"))
                                 or not self.ANON.match(k)}

    def _active_human(self) -> dict:
        """照合対象の人間プロファイル（AI声紋 __..__ は除く）."""
        ai = {k for k in self._active_keys if k.startswith("__") and k.endswith("__")}
        return {k: v for k, v in self.profiles.items()
                if k in self._active_keys and k not in ai}

    def is_active_human(self, key: str) -> bool:
        """key が現在照合対象のアクティブな人間プロファイルか（人物N含む・AI声紋除く）.

        SessionState.constrain_human_speaker_key の「声紋で実在が裏付けられた
        キーはスロット選別の対象外」判定用の公開API。profiles 全体（inactive・
        voices.json 残留分を含む）で判定すると、無効化済みプロファイルまで
        参加人数上限を素通りする穴になる（2026-07-15 レビューで確定）ため、
        アクティブ集合に限定する。
        """
        with self._lock:
            return key in self._active_human()

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

        コミット直前に純度検査（_purity_subset）を通す（二段構え）:
        anchor一貫性(ecs)は「今の声に似ているか」の粗い前置フィルタだが、
        電話会話等でSonioxのセグメント境界が甘いと1発話に両話者が混ざり、
        窓埋め込みが両者の中間に落ちて ecs を通過し得る（CallHome 0856 の
        当初分析 docs/design/handoff_2026-07-14_unregistered_speakers.md §13.1。
        なお §13.2 の再現で登録汚染は 0856 の主因ではなかったと判明済みで、
        本検査は「将来条件で効く汚染への保険」として維持されている）。
        そこでコミット時に集合全体のペアワイズ一貫性を検査し、
        - 採用部分集合の累積文字数が enroll_min_total_chars に届けば
          その部分集合のみでプロファイル作成（混入分は pool に残す＝
          もう一方の話者の蓄積材料として生かす）
        - 届かなければ登録を保留して蓄積継続（新しい閾値は導入せず、
          既存の「クリーンな発声の累積文字数」条件をそのまま流用）
        純度検査はコミット候補時のみ走るため、混入のない通常セッションの
        登録タイミングは変わらない（二重検査による登録遅延なし）。
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
        keep = self._purity_subset(embs)
        if len(keep) < len(embs):
            kept_total = sum(wts[i] for i in keep)
            if kept_total < self.enroll_min_total_chars:
                # 混入が激しくクリーン分が足りない → 保留して蓄積継続
                self._note("純度保留", label=sp, n=len(embs), n_keep=len(keep),
                           chars=round(kept_total))
                return None
            idx = [idx[i] for i in keep]
            embs = [embs[i] for i in keep]
            wts = [wts[i] for i in keep]
            total = kept_total
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
        self._record_label_success(sp, target)
        self._note("自動登録" if is_new else "合流", label=sp, name=target,
                   rename=rename, chars=round(total_chars))
        return target

    def set_max_human_speakers(self, value: int | None) -> None:
        self.max_human_speakers = value

    def set_hybrid(self, value: bool) -> None:
        """ハイブリッド構成（ClusterVoiceNamer有効）を宣言する.

        インスタンス属性 ``self.hybrid`` を設定する（__init__ で False に実体化
        済み。クラス属性 ``VoiceProfiles.hybrid`` は既定値の定義であり、ここでは
        書き換えない＝他インスタンスへ波及しない。2026-07-15 レビュー F8）。
        """
        self.hybrid = bool(value)

    def _note(self, kind: str, **info) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + 1
        self.last = {"kind": kind, **info}

    def _update_room_stats(self, sp: str, emb: np.ndarray) -> None:
        for l2, es in self.label_embs.items():
            tgt = self.same_sims if l2 == sp else self.diff_sims
            tgt.extend(float(np.dot(emb, e2)) for e2 in es[-3:])
        del self.same_sims[:-60]
        del self.diff_sims[:-120]

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
        norm = float(np.linalg.norm(emb))
        if norm == 0.0 or not np.isfinite(norm):
            # 全ゼロ・非有限の埋め込み（無音区間やモデルの異常出力）は正規化で
            # NaN に化け、以後の内積比較を全て壊す。従来は NaN 伝播で照合不成立に
            # 落ちるだけだったが、クラスタ間名寄せは埋め込みを代表として保存する
            # (docs/design/handoff_2026-07-14_unregistered_speakers.md §3) ため、
            # ここで None に落として保存経路に NaN が入らないようにする。
            return None
        return emb / norm

    def classify(self, wav: np.ndarray, sp, overlapped: bool = False,
                 count: bool = True, chars: int = 0, enroll: bool = True) -> str:
        """発話を人物キーに割り当てる（経路はクラスドキュメント参照）.

        overlapped=True の発話は声が混ざっていて声紋がデタラメになるため、
        声での判定をスキップして直前の対応を維持する。
        count=False（相槌など）の発話は声紋の照合そのものをスキップする。
        声紋で判定できない発話は「ラベル継続」: そのSTTラベルの現在の対応先
        （声紋で裏付けられた人物、まだ無ければ #ラベルのプレースホルダ）を
        そのまま返す（2026-07-14, eval/replay_attribution.py での再設計）。
        一度でも照合に外れたらラベルの人物対応を破棄して #ラベルへ落とす旧仕様は、
        中尺発話の声紋が不安定な実会話で対応が壊れ続け（同一人物が #ラベルと
        人物Nに分裂、1:1帰属精度44%）、継続に変えるだけで54%に改善した。
        ラベル継続の対応先は必ず声紋照合の成功（一致・登録・合流）でしか
        書き換わらないため、「声の証拠に基づく最後の対応を維持する」機構であり、
        根拠なしに直前話者へ寄せる旧「前話者追従」（実測28%で廃止）とは異なる。
        enroll=False（エコー窓中の人間発話など）は照合・補正は行うが、声紋の蓄積・
        人物登録には使わない。エコー窓直後に集中する返答が声紋補正なしのラベル追従に
        落ちるのを防ぎつつ、漏れ込んだAI音声で匿名話者が育つのは防ぐ（P2-2）。
        chars はその発話の文字数。新規人物の自動登録は、声ごとにこの文字数を
        累積し、一貫したクラスタが閾値を超えた時点で確定する（発話数では数えない）。
        """
        with self._lock:
            return self._classify(wav, sp, overlapped, count, chars, enroll)

    def _classify(self, wav: np.ndarray, sp, overlapped: bool,
                  count: bool = True, chars: int = 0, enroll: bool = True) -> str:
        sp = str(sp)
        prev = self.sp_map.get(sp)
        # 初期値「照合なし」は「中尺/長尺の照合経路に入らなかった」ことを示す
        # センチネル（相槌 count=False・short_floor未満・短発話経路の不成立）。
        # 旧称「相槌追従」は前話者追従の廃止（2026-07-14）後は実態と乖離して
        # いたため改名（docs/design/attribution_logic_review_2026-07.md D3。
        # 追従はせず、prev があればラベル継続、無ければ #ラベルに落ちるだけ）。
        kind, info = "照合なし", {}
        if overlapped and wav.size >= SR * self.min_sec:
            kind = "重なりスキップ"
        elif count and wav.size >= SR * self.min_sec:
            emb = self._embed(wav)
            if emb is None:
                kind = "声紋計算不可"
            else:
                self._update_room_stats(sp, emb)   # 部屋の同一/別人分布を実測(表示・診断用)
                if enroll:
                    # 手動登録用の直近サンプル。エコー窓中(enroll=False)は溜めない。
                    self.label_embs.setdefault(sp, []).append(emb)
                    del self.label_embs[sp][:-10]    # 直近10発話だけ保持
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
                    strict = wav.size < SR * self.strict_sec
                    bonus = self.short_bonus if strict else 0.0
                    need_th = self._person_th(cand, th) + bonus
                    # margin は全帯共通の1本（short_margin_mult は ablation で
                    # 出力ビット一致を確認して削除。review P3）。
                    need_mg = self.margin
                    # 人物別しきい値の学習履歴は person_th 判定の手前で記録する
                    # （固定基準のみ。理由は _record_reference_sim docstring）。
                    if sim >= th + bonus and sim - second >= need_mg:
                        self._record_reference_sim(cand, sim)
                    if sim >= need_th and sim - second >= need_mg:
                        self.sp_map[sp] = cand
                        self._record_label_success(sp, cand)
                        self._track_own_emb(cand, emb)   # 事後回収（二峰性監視）の材料
                        self._note("補正" if (prev is not None and not prev.startswith("#")
                                              and prev != cand) else "声紋一致", label=sp, **info)
                        return cand
                # 既知の誰にも確信を持って一致しなかった → ラベル継続。ラベルの
                # 現在の対応（直近の声紋照合成功で決まった人物 or #ラベル）を維持する。
                # 旧仕様は「その人物と再一致しなければ prev を破棄して #ラベルへ」
                # だったが、実測（eval/replay_attribution.py, 2026-07-14_142016）では
                # 中尺発話の声紋が本人でもしきい値に届かず（本人一致 0.17〜0.45）、
                # 一度の不一致で人物対応が壊れて同一人物が #ラベルと人物Nに分裂し、
                # 1:1帰属精度44%の主因になっていた。継続化で54%（他の変更と合わせ79%）。
                # 対応先が継続不可（remap等で消えた・deactivate済み・AI声紋）の
                # 場合だけは継続を断つ（判定は _continuation_target に統一。3経路共通）。
                if (prev is not None and not prev.startswith("#")
                        and self._continuation_target(prev) is None):
                    prev = None
                # 登録: 発話数ではなく「声ごとのクリーンな発声の累積文字数」で確定する。
                # 長い発話は窓分割して複数サンプル化（連続発話でも登録が進み、内部一貫性も確認）。
                # 不純ラベル（直近の照合成功が複数人物に割れている）の音声は登録・
                # 蓄積に使わない: 混載ラベル由来のプロファイルが既存人物へ交互に
                # 合流してプロファイル自体を汚染し、クラスタ照合まで巻き込む
                # （Chiba 0532 実測: 同一ラベルから人物1へ11回/人物2へ4回の合流、
                #  handoff §15.7-15.8）。
                enrollable = (enroll and self.auto and chars > 0
                              and self._label_pure(sp))
                kind = "蓄積中" if enrollable else "未確定"
                if enrollable:
                    ecs = cs + self.enroll_consist_bonus
                    samples = self._segment_samples(wav, emb, chars)
                    target = self._enroll_accumulate(samples, sp, prev, ecs)
                    if target is not None:
                        return target
        elif count and not overlapped and wav.size >= SR * self.short_floor:
            # 短い発話の取り違え安定化: 既知の2人以上を厳格に区別できるときだけ正す。
            # overlapped（重なり発話）は除外: 声が混ざった埋め込みは classify docstring
            # のとおりデタラメで、中尺は「重なりスキップ」なのに短発話だけ照合・補正
            # （sp_map 書き換え）まで走っていた（2026-07-15 レビューで確定）。重なりは
            # 発話長によらず声での判定をせず、下のラベル継続へ落とす。
            # 登録・蓄積はしない（声が短く不安定なため、登録に混ぜると精度が落ちる）。
            # ハイブリッド構成(hybrid=True)では既知1人でも照合を試みる。声紋一致92%
            # に対し前話者追従28%（transcripts/2026-07-14_1729 GT評価）のため、
            # 登録済みプロファイルが1人しか居ない蓄積期でも「声紋で当てられる短発話」
            # を追従に落とさない。しきい値は同じ厳格運用（誤爆は増やさない）。
            active = self._active_human()
            if len(active) >= 2 or (self.hybrid and active):
                emb = self._embed(wav)
                ranked = self._rank_active(emb, active) if emb is not None else None
                if ranked is not None:
                    cand, sim, second = ranked
                    info.update(sim=round(sim, 3), second=round(second, 3),
                                name=cand, short=True)
                    strict_th = self._person_th(cand, self.thresh) + self.short_bonus
                    if (sim >= strict_th
                            and sim - second >= self.margin):
                        self.sp_map[sp] = cand
                        self._record_label_success(sp, cand)
                        self._note("補正" if (prev is not None and not prev.startswith("#")
                                              and prev != cand) else "声紋一致",
                                   label=sp, sim=round(sim, 3), second=round(second, 3),
                                   name=cand, prev=prev, short=True)
                        return cand
                # 厳格に決められない短い発話はラベル継続（そのラベルの現在の人物対応を
                # 維持）。対応先は声紋照合の成功でしか書き換わらないため根拠なしの
                # 決めつけではない（classify docstring の実測根拠を参照）。
                # 継続の門番は _trusted_continuation（ラベル信頼度）に統一。
                cont = self._trusted_continuation(sp, prev)
                if cont is not None:
                    self._note("ラベル継続", label=sp, prev=cont, short=True)
                    return cont
        # 声紋で決められない発話（相槌・短発話・声紋計算不可の短経路）はラベル継続:
        # そのラベルの現在の対応（声紋照合の成功で確定した人物 or #ラベル）を返す。
        # なお相槌テキストの最終的な表示は呼び出し側（RecvLoop.flush）が未確定に
        # 落とす規則を持つ（相槌は聞き手が打つ＝直前話者と別人のことが多い）。
        # 継続の門番は _trusted_continuation（ラベル信頼度）に統一（短発話経路と共通）。
        if kind == "照合なし":
            cont = self._trusted_continuation(sp, prev)
            if cont is not None:
                self._note("ラベル継続", label=sp, prev=cont, **info)
                return cont
        # ラベル不純（直近の照合成功が複数人物に割れている）: このラベルに基づく
        # 帰属（prev 継続も #ラベルのプレースホルダも）は複数話者を混載するため
        # 未確定に落とす。sp_map は汚さない（照合成功が単一人物に収束すれば
        # 継続は自然に復活する）。Chiba 0532 実測でラベル継続22%（54件中12正解）
        # が誤帰属29%の主因だった（handoff §15.7）。
        if not self._label_pure(sp):
            info.setdefault("prev", prev)
            info["hist"] = (self.label_hist.get(str(sp), [])
                            [-int(self.label_purity_window):])
            self._note("ラベル不純", label=sp, **info)
            return UNSURE_SPEAKER
        # 継続不可の人物対応（AI声紋 __AI__、deactivate 済み等）はプレースホルダにも
        # 継続にも使えないため未確定へ落とす（2026-07-15 レビューで確定した実バグ:
        # AI声紋一致で sp_map[sp]="__AI__" になった後の同ラベル短発話・相槌が
        # __AI__ のまま返り、_recv_loop の startswith("__") エコー破棄で本文ごと
        # 消えていた。人間の発話は捨てず「未確定」として表示に残す）。
        if (prev is not None and not prev.startswith("#")
                and self._continuation_target(prev) is None):
            key, kind = UNSURE_SPEAKER, "継続不可"
        else:
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

    def match_profile(self, wav: np.ndarray) -> tuple[str, float] | None:
        """副作用なしのクラスタ単位声紋照合（pyannoteハイブリッド構成用）.

        classify() と異なり sp_map/pool/own_sims 等の状態を一切変更しない。
        pyannote Live-1 のクラスタ音声（複数発話を束ねた長尺）を渡し、現在
        アクティブな登録プロファイルの中で最も近いものを返す。即時判定と同じ
        条件（しきい値＋2位とのmargin）を満たさなければ None（confidence不足
        で未確定のまま蓄積を続ける、の判断は呼び出し側=ClusterVoiceNamer が行う）。
        """
        with self._lock:
            if wav.size < SR * self.min_sec:
                return None
            emb = self._embed(wav)
            if emb is None:
                return None
            active = self._active_human()
            ranked = self._rank_active(emb, active)
            if ranked is None:
                return None
            cand, sim, second = ranked
            if sim >= self._person_th(cand, self.thresh) and sim - second >= self.margin:
                return cand, sim
            return None

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
        # 実名化されたら事後回収(P3)の監視対象から外れる（実名は書き換えない方針）
        self.own_embs.pop(old, None)
        self._own_updates.pop(old, None)
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
            self.own_embs.pop(src, None)
            self._own_updates.pop(src, None)
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
                        self.own_embs.pop(key, None)
                        self._own_updates.pop(key, None)
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
            order = ["声紋一致", "補正", "自動登録", "合流", "蓄積中", "純度保留",
                     "プロファイル再構築", "未確定", "ラベル継続", "ラベル不純",
                     "継続不可", "照合なし", "重なりスキップ", "声紋計算不可"]
            parts.append("判定内訳: " + " / ".join(
                f"{k}{self.counts[k]}" for k in order if self.counts.get(k)))
        return "、".join(parts) or "判定なし"
