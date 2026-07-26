"""席ごとの実音声を持ち、席上限で落ちる発話の寄せ先を選ぶ（ハイブリッド構成）.

**解く問題**（handoff §27）: 3人の会話でも pyannote は同じ人を複数クラスタに
割ることがある。割れた側は席上限（`constrain_human_speaker_key`）で落ち、
未確定になる。実測では実質発話の 11.8% がこれで、しかも落ちたキーは**全て**
`@diar:N`＝既に席を持っている人の分裂だった（新しい参加者ではない）。

**なぜ席の割当てで、クラスタ間名寄せではないのか**: クラスタ埋め込み同士の
比較で同一人物と別人を分ける絶対しきい値は存在しない（§15.12。別人が 0.89
に達する実測を §27.7 で再確認した）。したがって「このクラスタは既存の誰かか、
それとも新しい人か」という**開集合**の判定はできない。

本モジュールが行うのは開集合の判定ではない。対象は
**constrain が席を与えられなかった発話に限られる**——参加人数の設定上そこに
新しい参加者は入れない、と既に決まっている状態である。残る問いは「席を持つ
N人のうち誰か」という**閉集合の割当て**で、これは成立する（実測: その発話
自身の音声で1位を選んで 89%、1秒未満の短い発話でも 82%）。

**確定はしない**: 選択はその発話1件限りで、台帳にも `_cluster_naming` の
確定にも書かない。§15.12 の一般則「不可逆な操作は高確信を要求する」に対し、
こちらは可逆なので高確信を要求しない（下限を課しても成績はほぼ変わらず
＝しきい値は効いていないことが実測で分かっている。§27.7）。

**参照側が人物プロファイルでないのはなぜか**: `VoiceProfiles` のプロファイルは
短窓の登録サンプルから作られており、席が実際に喋った音声と比べると分離が
落ちる。同じ分裂クラスタでも、人物プロファイル相手では確定線に届かないのに、
席の実音声相手では 0.78-0.92 出る（§27.4）。
"""
from __future__ import annotations

import threading

import numpy as np

from ._constants import (
    RETRO_INTERVAL_SEC,
    RETRO_SCHEDULE_SEC,
    SEAT_AUDIO_MIN_REF_SEC,
    SEAT_AUDIO_REF_SEC,
    SR,
    UNSURE_SPEAKER,
)
from ._speaker_keys import is_ai_key


class SeatAudio:
    """席の表示キー -> その席の実音声から作った埋め込み.

    参照は席ごとに ``ref_sec`` 秒まで貯めたら**凍結**する（以後更新しない）。
    理由は2つ:

      - 席には誤帰属も混ざる（実測で16%）。貯め続けると参照が汚れていく
      - 参照を更新するたびに埋め込みを計算し直す必要があり、発話ごとに走ると
        ライブの遅延に効く。凍結すれば席あたり数回で済む

    tracker には書き込まない（``embed_audio`` は副作用なし）。状態は本
    インスタンスに閉じる。
    """

    def __init__(self, tracker, *, ref_sec: float = SEAT_AUDIO_REF_SEC,
                 min_ref_sec: float = SEAT_AUDIO_MIN_REF_SEC) -> None:
        self.tracker = tracker
        self.ref_sec = float(ref_sec)
        self.min_ref_sec = float(min_ref_sec)
        self._buffers: dict[str, list[np.ndarray]] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._seconds: dict[str, float] = {}
        self._frozen: set[str] = set()
        # 診断用（直近の割当ての結果。diag への出力に使う）。
        self.last_pick: dict[str, object] | None = None
        # observe/nearest は recvスレッド、rename/reset は UI・入力スレッドから
        # 呼ばれる（_cluster_naming と同じ事情）。再入可能な RLock を使う。
        self._lock = threading.RLock()

    # -- 参照の構築 ---------------------------------------------------

    def observe(self, key: str, wav: np.ndarray | None) -> None:
        """席が確定した発話の音声を参照として貯める（凍結済みなら何もしない）."""
        if wav is None or wav.size == 0:
            return
        if not key or key == UNSURE_SPEAKER or is_ai_key(key):
            return
        with self._lock:
            if key in self._frozen:
                return
            buf = self._buffers.setdefault(key, [])
            buf.append(np.asarray(wav, dtype=np.float32))
            total = sum(a.size for a in buf)
            concat = np.concatenate(buf) if len(buf) > 1 else buf[0]
        # 埋め込みはロックの外で計算する（tracker 側のロックと入れ子にしない）。
        emb = self.tracker.embed_audio(concat)
        with self._lock:
            if emb is not None:
                self._embeddings[key] = emb
                self._seconds[key] = total / SR
            if total >= int(self.ref_sec * SR):
                self._frozen.add(key)
                self._buffers.pop(key, None)   # 凍結後は音声を保持しない

    # -- 割当て -------------------------------------------------------

    def nearest(self, wav: np.ndarray | None) -> tuple[str, float, float] | None:
        """席を持つ人のうち、この音声に最も似ている1人を返す.

        戻り値: (席の表示キー, 類似度, 2位との差) または None。

        **下限は課さない**。閉集合の割当てなので「誰でもない」という選択肢は
        呼び出し側の適用条件（席上限で落ちた発話に限る）が既に排除している。
        下限を課しても成績はほぼ変わらないことも実測済み（§27.7）。

        席が1つしか無いときは None を返す。比較にならないうえ、その状況で
        寄せるのは「席が空いているのに落ちた」＝別の不具合の可能性がある。

        参照が ``min_ref_sec`` に育っていない席は**候補から外す**（全席が
        育つのを待つのではない。待つ形にすると、たまにしか喋らない人が1人
        いるだけで割当てが止まり、適用が157→17件に落ちる）。
        """
        emb = self.embed(wav)
        return None if emb is None else self.nearest_from(emb)

    def embed(self, wav: np.ndarray | None) -> np.ndarray | None:
        """音声を声紋にする（遡及訂正が同じ埋め込みを取っておくために公開）."""
        if wav is None or wav.size == 0:
            return None
        return self.tracker.embed_audio(wav)

    def nearest_from(self, emb: np.ndarray) -> tuple[str, float, float] | None:
        """既に計算済みの声紋で寄せ先を選ぶ（本体。docstring は `nearest`）.

        遡及訂正（`RetroAttributor`）は序盤の発話の声紋を保持しておき、
        参照が育った後に**この関数だけ**を呼び直す。音声を持ち直す必要も、
        埋め込みを計算し直す必要もない。
        """
        with self._lock:
            cands = {k: v for k, v in self._embeddings.items()
                     if self._seconds.get(k, 0.0) >= self.min_ref_sec}
        if len(cands) < 2:
            return None
        ranked = sorted(((float(np.dot(emb, v)), k) for k, v in cands.items()),
                        reverse=True)
        sim, key = ranked[0]
        second = ranked[1][0]
        self.last_pick = {"key": key, "sim": round(sim, 3),
                          "margin": round(sim - second, 3), "n_seats": len(cands)}
        return key, sim, sim - second

    # -- 台帳の一貫性 -------------------------------------------------

    def rename(self, old: str, new: str) -> None:
        """表示キーの付け替えを反映する（SessionState.rekey からの伝搬）.

        追従しないと、リネーム後に旧キーへ寄せてしまい、消えたはずの人格が
        復活する（`_cluster_naming.rename_confirmed` と同じ事情）。
        """
        if old == new:
            return
        with self._lock:
            for store in (self._buffers, self._embeddings, self._seconds):
                if old in store:
                    store[new] = store.pop(old)
            if old in self._frozen:
                self._frozen.discard(old)
                self._frozen.add(new)

    def reset(self) -> None:
        """会議リセット時に参照をすべて捨てる."""
        with self._lock:
            self._buffers.clear()
            self._embeddings.clear()
            self._seconds.clear()
            self._frozen.clear()
            self.last_pick = None


class RetroAttributor:
    """序盤に決めた帰属を、席の参照が育ってから貼り直す（handoff §28）.

    **なぜ要るのか**: 誤りはセッション序盤に極端に偏る。実測（Chiba 9会話）で
    開始0-1分は正解29%、1-2分は69%、5-10分は90%。システムは収束していて、
    悪いのは立ち上がりだけだった。にもかかわらず、一度決めた帰属を二度と
    見直していなかった。

    貼り直したときの実測:

        いまのまま       正解 79.2% / 誤帰属 13.6% / 未確定 7.1%
        2分時点で貼り直す 84.5% / 13.9% / 1.6%
        5分時点で貼り直す 89.5% /  9.9% / 0.6%
        終了時に一括     93.1% /  6.9% / 0.0%

    **持つものが小さい**: 発話ごとの声紋は192次元。1200発話でも 1.8MB で、
    音声を持ち直す必要はない。判定も `SeatAudio.nearest_from` を新しい参照で
    呼び直すだけ。

    **対象**: 席の音声で決めた発話と、未確定のまま残った発話だけ。声紋層が
    高信頼で判定した発話（声紋一致・補正・自動登録・合流）は触らない——
    そちらは席の平均より強い、という §27.12 の判断を変えていない。
    """

    def __init__(self, seat: SeatAudio, *,
                 schedule: tuple[float, ...] = RETRO_SCHEDULE_SEC,
                 interval: float = RETRO_INTERVAL_SEC) -> None:
        self.seat = seat
        # 校正: 2分・5分は handoff §28.2 の実測点。それ以降を interval ごとに
        # 繰り返すのは、終了時一括（93.1%）が5分時点（89.5%）を上回る＝
        # 参照が育つほど良くなるという同じ測定からの外挿。
        self.schedule = tuple(schedule)
        self.interval = float(interval)
        self._embeddings: dict[int, np.ndarray] = {}   # ms -> 声紋
        self._next_idx = 0
        self._next_at = self.schedule[0] if self.schedule else self.interval
        self._lock = threading.RLock()

    def remember(self, ms: int | None, emb: np.ndarray | None) -> None:
        """後で決め直せるように、この発話の声紋を控える."""
        if ms is None or emb is None:
            return
        with self._lock:
            self._embeddings[int(ms)] = emb

    def due(self, elapsed_sec: float) -> bool:
        """貼り直しの時刻に達したか（達したら次回の時刻へ進める）."""
        with self._lock:
            if elapsed_sec < self._next_at:
                return False
            self._next_idx += 1
            if self._next_idx < len(self.schedule):
                self._next_at = self.schedule[self._next_idx]
            else:
                self._next_at = elapsed_sec + self.interval
            return True

    def revise(self) -> dict[int, str]:
        """控えてある声紋を今の参照で判定し直し、{ms: 新しいキー} を返す.

        呼び出し側（`SessionState.apply_retro_attribution`）が records を
        書き換える。ここは判定だけで副作用を持たない。
        """
        with self._lock:
            items = list(self._embeddings.items())
        out: dict[int, str] = {}
        for ms, emb in items:
            got = self.seat.nearest_from(emb)
            if got is not None:
                out[ms] = got[0]
        return out

    def rename(self, old: str, new: str) -> None:
        """付け替えは `SeatAudio` 側が持つので、ここは何もしない（明示）."""

    def reset(self) -> None:
        with self._lock:
            self._embeddings.clear()
            self._next_idx = 0
            self._next_at = (self.schedule[0] if self.schedule
                             else self.interval)
