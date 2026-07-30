"""WebSocket受信ループ + flush処理."""
from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from ._session_state import SessionState
    from .stt import STTBackend

import contextlib

from ._attribution import (
    _VOICEPRINT_RELIABLE_KINDS,
    decide_speaker,
    impure_lowsim,
    voiceprint_endorses,
)
from ._constants import (
    _BACKCHANNEL_RE,
    RESET,
    UNSURE_SPEAKER,
    fmt_ts,
)
from ._seat_audio import declines_short
from ._speaker_keys import is_ai_key
from ._ui import _print_line
from ._voice_profiles import _best_text_similarity

# 発話の話者帰属決定（声紋→Resolver→クラスタ確定/匿名キー）は _attribution.py の
# decide_speaker に一本化した（2026-07-17 再編。従来ここにあった if 連鎖が
# 「事実上の統合層」だった。docs/design/attribution_logic_review_2026-07.md §2）。
_UNKNOWN_STT_SPEAKERS = {"", "none", "null", "unknown", "uu", UNSURE_SPEAKER}
# 帰属の根拠がSTTラベルしか無い声紋判定の種別（handoff §27.12）。ラベル不純は
# そのラベルが複数人を混載していると分かっている状態、ラベル継続は声紋照合が
# 成立せずラベルの過去の対応を引き継いでいるだけの状態。どちらも「ラベルに
# 基づく推測」なので、席の実音声と直接比べたほうが強い（実測: この2種を
# 席の音声で決め直すと 正解 71.0%→79.2% / 誤帰属 19.7%→13.6%）。
_LABEL_ONLY_KINDS = frozenset({"ラベル不純", "ラベル継続"})
RecvStatus = Literal["ok", "finished", "disconnected"]


def _is_unknown_stt_speaker(speaker) -> bool:
    return str(speaker).strip().lower() in _UNKNOWN_STT_SPEAKERS


def _stt_speaker_key(speaker) -> str:
    if _is_unknown_stt_speaker(speaker):
        return UNSURE_SPEAKER
    return "#" + str(speaker)


class RecvLoop:
    """STTからのトークンストリームを処理し、発話を確定(flush)してrecordsに追加する.

    内部トークン形式（STTBackend.parse_messageの出力）を受け取り、
    声紋判定・エコー除去・records追加・ファイル保存を行う。
    STTプロバイダには依存しない（バックエンドが変換済みトークンを供給）。
    """

    _FLUSH_TIMEOUT = 30.0     # トークンが来なくなってからの強制flush（秒）
    _FLUSH_SOFT_CHARS = 500   # この文字数を超えたら文の切れ目でflush
    _FLUSH_HARD_CHARS = 1000  # この文字数を超えたら問答無用で強制flush

    def __init__(self, state: SessionState, args, backend: STTBackend):
        self.state = state
        self.args = args
        self.backend = backend
        self.cur_speaker = None
        self.cur_text = ""
        self.cur_ms: int | None = None
        self.cur_end: int | None = None
        self.cur_last_token_time: float = time.monotonic()
        self.recent_segs: list[tuple] = []

    def _note_echo_drop(self, src: str, *, sim: float | None = None,
                        key: str | None = None) -> None:
        """エコーとして捨てた発話を diag に1行残す.

        捨てた発話は records にも turns にも残らないので、ここで書かないと
        「記録が無いのに登録通知だけある」状態を後から追えない。テキスト安全網
        （agent/partner/retired）と声紋照合の3経路で同じ形の行を出す——経路に
        よって記録の有無が違うと、記録から挙動を再生できない（handoff §23）。
        """
        row: dict[str, object] = {
            "ms": self.cur_ms, "end": self.cur_end,
            "type": "echo_drop", "src": src,
            "text": self.cur_text.strip()[:40],
        }
        if sim is not None:
            row["sim"] = round(sim, 3)
        if key is not None:
            row["key"] = key
        with contextlib.suppress(OSError), \
                open(self.state.diag_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def overlaps_other(self, start, end, label) -> bool:
        if start is None or end is None:
            return False
        return any(lbl != label and min(e, end) - max(s, start) > 0
                   for s, e, lbl in self.recent_segs)

    def _link_mint_to_cluster(self, name: str) -> None:
        """鋳造したての人物を、同一人物の席持ちクラスタへ統合する（opt-in）.

        二重帳簿の根治（handoff_2026-07-25_dual_ledger_rootcure.md 案B）。
        声紋側が新しい戸籍 人物N を作った直後に、その鋳造したてのプロファイルと
        席を持つ各クラスタの蓄積声紋を**対称比較**し、同一人物と判定できたら
        クラスタ側の席を人物Nへ rekey する（統合の単一入口は従来どおり
        SessionState.rekey）。結果として 1人=1クラスタ=1戸籍=1席 になり、
        席の二重取りで実在者が締め出される問題が消える。

        既定は無効（--vp-mint-cluster-link で有効化）。cluster_namer が無い構成
        （Soniox単独・pyannote単独）は呼ばれても即 return するため挙動不変。
        判定は鋳造の瞬間の1回きり（繰り返し判定は分離が消える。
        _constants.PYANNOTE_CLUSTER_MINT_LINK_MIN_SIM の校正メモ参照）。
        """
        s = self.state
        namer = s.cluster_namer
        tracker = s.tracker
        if (namer is None or tracker is None
                or not getattr(self.args, "vp_mint_cluster_link", False)):
            return
        prof = tracker.profiles.get(name)
        if prof is None:
            return

        def _key_of(raw_cluster: str) -> str | None:
            # 席＝現在この生クラスタが持っている表示キー。確定名があればそちら。
            return (namer.confirmed_name(raw_cluster)
                    or s.diarization_speaker_keys.get(raw_cluster))

        hit = namer.link_minted_profile(prof, _key_of)
        if hit is None:
            return
        raw_cluster, seat_key, sim = hit
        if seat_key != name:
            # 席（@diar:N 等）の過去分・台帳・色をまとめて人物Nへ寄せる。
            s.rekey(seat_key, name)
        with s.state_lock:
            s.diarization_speaker_keys[raw_cluster] = name
        namer.adopt_confirmed(raw_cluster, name)
        with contextlib.suppress(OSError), \
                open(s.diag_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ms": self.cur_ms, "end": self.cur_end,
                "type": "mint_cluster_link", "cluster": raw_cluster,
                "seat": seat_key, "name": name, "sim": round(sim, 3),
            }, ensure_ascii=False, default=str) + "\n")
        if self.args.vp_debug:
            _print_line(f"# 鋳造リンク: {seat_key}({raw_cluster}) を {name} へ統合"
                        f"（対称類似{sim:.2f}）")

    def _maybe_retro_reattribute(self) -> None:
        """時刻が来たら、序盤に決めた帰属を今の参照で貼り直す（handoff §28）.

        誤りはセッション序盤に極端に偏る（実測: 開始0-1分は正解29%、
        5-10分は90%）。システムは収束していて悪いのは立ち上がりだけなので、
        参照が育った時点で決め直すと 79.2%→89.5%（5分時点）になる。

        表示済みの行の話者名が変わるため、変わったことをシステム行として
        タイムラインに残す（黙って書き換えない）。
        """
        s = self.state
        if s.retro is None or self.cur_ms is None:
            return
        if not s.retro.due(self.cur_ms / 1000.0):
            return
        applied = s.apply_retro_attribution(s.retro.revise())
        # 貼り直しで発言の無くなったキーが表示文字を押さえ続けると、参加者が
        # 1人しか居ないのに「参加者B」から始まる。空いた文字を詰め直す。
        s.compact_anonymous_labels()
        if not applied:
            return
        # diag に残す。発話ごとの行は flush 時点の final_key しか持たないので、
        # これが無いと記録から最終状態を復元できず、オフライン採点が遡及訂正の
        # ぶんだけ低く出る（handoff §28.10）。
        with contextlib.suppress(OSError), \
                open(s.diag_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ms": self.cur_ms, "type": "retro_reattribution",
                "changed": len(applied),
                "pairs": [[m, k] for m, k in sorted(applied.items())],
            }, ensure_ascii=False, default=str) + "\n")
        s.add_sys(self.cur_ms,
                  f"これまでの声を聞き直して、{len(applied)}件の話者を再判定しました")
        _print_line(f"# 遡及訂正: {len(applied)}件の話者を再判定しました")

    # ------------------------------------------------------------------
    # flush の各段（順番と各段の役割は flush の docstring を参照）
    # ------------------------------------------------------------------

    def _clear_current(self, *, reset_timer: bool = False) -> None:
        """組み立て中の発話を捨てる（確定でも破棄でも共通の後始末）.

        `reset_timer` はトークンが来なくなってからの強制flush（`_FLUSH_TIMEOUT`）
        の起点。**エコー破棄では触らない**——破棄はタイマーを進める理由に
        ならず、触ると強制flushの間隔が黙って延びる。
        """
        self.cur_text = ""
        self.cur_ms = None
        self.cur_end = None
        if reset_timer:
            self.cur_last_token_time = time.monotonic()

    def _text_echo_match(self, *, use_intervals: bool) -> tuple[str, float] | None:
        """テキストの近さでAIのエコーを見つける（安全網。声紋より前に効かせる）.

        声紋トラッカーの副作用（文字数の蓄積・自動登録）より前に評価する。
        エコーと判定したら classify を呼ばずに捨てるので、漏れ込んだAI音声で
        匿名話者が育って登録される事故を防げる（D2）。

        AI再生区間との重なりでエコー窓を判定する（P2-1）。STT確定が遅れて
        壁時計のエコー窓を過ぎた回り込みも、発話区間 [cur_ms, cur_end] が
        記録済みの再生区間と重なれば拾う。ms が無い/記録が無いときは従来の
        壁時計（in_echo_window）に倒す。

        戻り値: (どの経路で見つけたか, 類似度) または None。
        """
        s = self.state
        for name, src in (("agent", s.agent), ("partner", s.partner)):
            if src is None:
                continue
            if name == "agent":
                in_echo = (s.overlaps_ai_speech(self.cur_ms, self.cur_end,
                                                source="agent")
                           if use_intervals else src.in_echo_window)
                if not in_echo:
                    continue
            sim = src._best_similarity(self.cur_text)
            if sim > 0.35:
                return name, sim
        # パートナー切断直後もエコー参照を短時間保持する（P2-4）。partner が
        # None になってもテキスト安全網が効くよう、TTL 内の退役テキストとも照合。
        retired = s.recent_retired_echo_texts()
        if retired:
            sim = _best_text_similarity(self.cur_text.strip(), retired)
            if sim > 0.35:
                return "retired", sim
        return None

    def _utterance_audio(self) -> np.ndarray:
        """この発話の区間の音声を、STTへ送った音声のバッファから切り出す.

        ms はここへ送ったバイト位置そのもの（16kHz・16bit なので 32 bytes/ms）。
        区間が不明・長さゼロなら空配列を返す——呼び先（声紋・席）は空を
        「計算できない」として扱う。
        """
        s = self.state
        if not (self.cur_ms is not None and self.cur_end is not None
                and self.cur_end > self.cur_ms):
            return np.zeros(0, dtype=np.float32)
        with s.buf_lock:
            abs_start = self.cur_ms * 32
            abs_end = self.cur_end * 32
            rel_start = max(abs_start - s.asr_pcm_buf_offset, 0)
            rel_end = max(abs_end - s.asr_pcm_buf_offset, 0)
            seg = bytes(s.asr_pcm_buf[rel_start: rel_end])
        return np.frombuffer(seg, dtype="<i2").astype(np.float32) / 32768.0

    def _classify_utterance(self, wav, *, label: str, is_backchannel: bool,
                            use_intervals: bool) -> tuple:
        """声紋層にこの発話を判定させる（戻り: キー・診断・記録用・入力条件）.

        登録は声ごとの累積文字数で判定するので、この発話の文字数を渡す。
        F2: エコー窓中（AIが発話中/直後）は enroll=False で蓄積・自動登録を
        抑止する。室内音響でAI声紋照合(AI_THRESH)が外れた漏れ込みが「新規
        話者の蓄積」に化けるのを塞ぐ。話者判定自体は従来どおり行う。正当な
        人間発話の登録がエコー窓ぶん遅れるのは許容（登録は累積制のため軽微）。
        count: 相槌は照合ごとスキップ。

        agent 側は再生区間の重なりで判定（P2-1, ms/記録が無ければ壁時計）。
        partner 側は従来どおり壁時計。
        """
        s = self.state
        agent, partner = s.agent, s.partner
        agent_active = (
            s.overlaps_ai_speech(self.cur_ms, self.cur_end, source="agent")
            if use_intervals
            else (agent is not None
                  and (agent.ai_speaking or agent.in_echo_window))
        )
        ai_active = (
            agent_active
            or (partner is not None
                and (partner.ai_speaking or partner.in_echo_window))
        )
        overlapped = self.overlaps_other(self.cur_ms, self.cur_end, label)
        enroll = (not is_backchannel) and not ai_active
        sp_id = s.tracker.classify(
            wav, self.cur_speaker,
            overlapped=overlapped,
            count=not is_backchannel,
            enroll=enroll,
            chars=len(self.cur_text.strip()))
        # classify に実際に渡した条件（オフライン再生用, handoff §23）。
        # overlapped は発話系列から再現できるが、enroll はエコー窓＝AI再生
        # 区間に依存し記録からは再現できないため必ず残す。
        flags = {"ov": overlapped, "enr": enroll,
                 "chars": len(self.cur_text.strip())}
        return sp_id, s.tracker.last, {}, flags

    def _apply_voiceprint_effects(self, d, rec_extra: dict) -> str | None:
        """声紋判定に伴う台帳の更新と通知（戻り: 鋳造した人物キー、無ければ None）."""
        s = self.state
        if d and d["kind"] == "補正":
            # peek_disp_name（割当てなし）を使う。ここは説明文のためだけの
            # 参照で、constrain より前にラベル文字を確保してしまうと、席の
            # 無いキーが席持ちとして居座る（下の 自動登録 と同じ事故）。
            note = (f"声紋でラベル{d['label']}の取り違えを修正"
                    f"（類似{d['sim']:.2f}、"
                    f"放置なら{s.peek_disp_name(d['prev'])}の発言になっていた）")
            rec_extra["vp"] = "補正"
            rec_extra["note"] = note
            _print_line(f"# ⚡補正: {note}")
        elif d and d["kind"] == "自動登録":
            if d["rename"]:
                s.rekey(*d["rename"])
            self._link_mint_to_cluster(d["name"])
            # **通知は constrain の後に出す**。ここで disp_name を呼ぶと、
            # 席が満杯でも新しい人物にラベル文字が付いてしまい、
            # constrain の「既にラベルを持つ人は常に通す」規則によって
            # 上限を超えた席が恒久的に居座る（実会話で --max-speakers 3 に
            # 対し 参加者D まで出た。handoff §28.7）。
            return d["name"]
        elif d and d["kind"] == "合流":
            if d["rename"]:
                s.rekey(*d["rename"])
            if self.args.vp_debug:
                _print_line(f"# 合流: ラベル{d['label']}→{d['name']}")
        elif self.args.vp_debug and d:
            extra = f" 類似{d['sim']:.2f}({d['name']})" if "sim" in d else ""
            _print_line(f"# vp判定[{d['kind']}]{extra}")
        return None

    def _decide_and_constrain(self, sp_id, *, d, wav, label: str,
                              is_backchannel: bool, rec_extra: dict,
                              classify_flags: dict) -> tuple[str, str, dict]:
        """帰属を決め（`decide_speaker`）、人数上限を適用する.

        戻り値は (上限適用前のキー, 適用後のキー, diag へ併記する入力)。
        上限前後の**両方**を返すのは、diag に併記して「resolver は正しいキーを
        選んだのに上限で未確定に落ちた」事象を後から切り分けるため。従来の
        diag は上限前の key しか持たず、この区別ができなかった。

        `diag_extra` には「判定の入力」を集める（records ではなく diag へ）。
        目的はオフライン再生: これが無いとクラスタ層の入力が実行後に失われ、
        記録から本番コードを回せない（handoff §23）。判定の**出力**の側
        （speaker_source / reason）も併記する——records は終了時に永続化
        されないため、これが無いと「この誤帰属は門番で止められる経路か、
        それとも STT フォールバックか」を後から分けられない（§26.6）。
        短いキー名なのは diag が1発話1行で膨らむため。
        """
        s = self.state
        diag_extra: dict[str, object] = dict(classify_flags)
        sp_id = decide_speaker(
            s, sp_id=sp_id, d=d, wav=wav,
            start_ms=self.cur_ms, end_ms=self.cur_end,
            rec_extra=rec_extra, vp_debug=self.args.vp_debug,
            diag_extra=diag_extra, stt_label=label,
        )
        if rec_extra.get("speaker_source") is not None:
            diag_extra["src"] = rec_extra["speaker_source"]
            diag_extra["why"] = rec_extra.get("speaker_reason")
        if self.cur_ms is not None and self.cur_end is not None:
            self.recent_segs.append((self.cur_ms, self.cur_end, label))
            del self.recent_segs[:-12]
        final_sp_id = s.constrain_human_speaker_key(
            UNSURE_SPEAKER if is_backchannel else sp_id)
        return sp_id, final_sp_id, diag_extra

    def _assign_seat(self, final_sp_id: str, *, sp_id, d, wav,
                     rec_extra: dict, diag_extra: dict) -> str:
        """席上限で落ちた発話・ラベル頼りの発話を、席の実音声で決め直す.

        上流はキーを決めていたのに席上限で落ちた発話は、実測で**全て**
        `@diar:N`＝既に席を持つ人の分裂だった（新しい参加者ではない）。
        参加人数の設定上そこに新しい参加者は入れないので、残る問いは
        「席を持つN人のうち誰か」だけになる。この発話1件に限って席の実音声と
        比べ、最も似た人へ寄せる（確定は書かない＝可逆。§15.12 の「不可逆な
        操作は高確信を要求」と衝突しない。handoff §27）。
        """
        s = self.state
        # 「蓄積中」の門番（handoff §27.11）。声紋が育っていない発話の帰属は
        # 裏付け（1位候補が帰属先と一致）が無いと当てにならず、実測で
        # 裏付けあり 12正解/1誤り に対し裏付けなし 2正解/29誤り だった。
        # 単独では「誤帰属 -2.7pt と引き換えに未確定 +2.7pt」の交換にしか
        # ならないので §27.6 で一度保留にしたが、切った先を下の席の音声が
        # 拾い直すので純増になる（正解 +9.0pt・誤帰属は横ばい）。
        # §18.8 の3d門番と同じ述語（voiceprint_endorses）を使う。3dより
        # 適用範囲が広い（経路を問わない）のは、そう測ったから。
        if (final_sp_id != UNSURE_SPEAKER and d is not None
                and d.get("kind") == "蓄積中"
                and not voiceprint_endorses(d, sp_id)):
            final_sp_id = UNSURE_SPEAKER
            rec_extra["speaker_source"] = "accumulating_without_endorsement"
            rec_extra["speaker_confidence"] = 0.0
            rec_extra["speaker_reason"] = (
                "voiceprint_still_accumulating_without_endorsement")
        kind = d.get("kind") if d is not None else None
        if kind in _LABEL_ONLY_KINDS:
            # 根拠がSTTラベルしかない kind は、上流のキーを信用せず
            # 席の実音声で決め直す（handoff §27.12）。
            #   ラベル不純: そのラベルが複数人を混載していると分かっている
            #   ラベル継続: 声紋照合が成立せず、ラベルの過去の対応を
            #               引き継いでいるだけ
            # どちらも「ラベルに基づく推測」であり、声を直接比べたほうが
            # 強い。実測で 正解 71.0%→79.2% / 誤帰属 19.7%→13.6%
            # （検証4本では 82.0%）。棄権していた分（未確定）も
            # 決めていた分（誤帰属の48%を占めていた）も同じ規則で扱う。
            reason = "label_only_kind_resolved_by_seat_audio"
        elif final_sp_id == UNSURE_SPEAKER and sp_id != UNSURE_SPEAKER:
            # 上流は決めていたのに席上限で落ちた分（§27.8 の本体）。
            reason = "seat_full_nearest_seat_audio"
        else:
            # 参照は「声紋層が高信頼だった発話」だけで作る。全発話で作ると
            # 席の参照そのものが汚れる（実測: ある席は GT 純度 38%）。
            # 高信頼4種に絞ると純度は 95-100% に上がり、寄せ先の的中も
            # 67%→70%、誤帰属の増分も 3.9→3.4pt に下がる（handoff §27.9）。
            if (final_sp_id != UNSURE_SPEAKER
                    and kind in _VOICEPRINT_RELIABLE_KINDS):
                s.seat_audio.observe(final_sp_id, wav)
            return final_sp_id
        # 声紋は1回だけ計算し、判定にも遡及訂正の控えにも使い回す。
        emb = s.seat_audio.embed(wav)
        dur_ms = (None if self.cur_ms is None or self.cur_end is None
                  else max(0, int(self.cur_end) - int(self.cur_ms)))
        if s.retro is not None:
            s.retro.remember(self.cur_ms, emb, dur_ms)
        picked = s.seat_audio.nearest_from(emb) if emb is not None else None
        if picked is None:
            # なぜ席で判定できなかったかを残す（診断のみ）。実会話で
            # 「ラベル不純/継続」178件のうち53件が判定できず未確定の
            # まま残っており、それが未確定18%の主因だった。短い発話に
            # 偏る（中央値0.30秒 対 0.66秒）が、合成音声では0.06秒でも
            # 埋め込みは計算できるため、原因が声紋計算なのか席の不足
            # なのかを記録から切り分ける（handoff §28.13）。
            diag_extra["seat_miss"] = (
                "no_embedding" if emb is None
                else f"few_seats:{s.seat_audio.n_ready()}")
            return final_sp_id
        if declines_short(picked, dur_ms):
            # 短くて僅差なら名前を出さない（handoff §36）。相槌の長さの発話で
            # 誤った名前を出すより、未確定のほうが読み手を惑わせない。
            diag_extra["seat"] = s.seat_audio.last_pick
            diag_extra["seat_miss"] = f"short_margin:{picked[2]:.3f}"
            rec_extra["speaker_source"] = "seat_assign_declined_short"
            rec_extra["speaker_confidence"] = 0.0
            rec_extra["speaker_reason"] = "short_utterance_margin_too_small"
            return UNSURE_SPEAKER
        rec_extra["speaker_source"] = "seat_assign"
        rec_extra["speaker_confidence"] = round(picked[1], 3)
        rec_extra["speaker_reason"] = reason
        diag_extra["seat"] = s.seat_audio.last_pick
        return picked[0]

    def _write_utterance_diag(self, *, label: str, sp_id, final_sp_id,
                              d, diag_extra: dict) -> None:
        """1発話1行の diag を書く（判定の入力と出力）.

        判定の根拠は `d`（この発話の classify 結果）だけを書く。
        `tracker.last` を直に読むと、STT が話者ラベルを返さなかった発話
        （classify を呼ばない経路。`d` は None にしてある）で**前の発話の
        判定**が書かれてしまい、記録が実態とずれる。採点も分析も diag の
        kind を信じて動くので、ここがずれると全部が静かに狂う。
        """
        with contextlib.suppress(OSError), \
                open(self.state.diag_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ms": self.cur_ms, "end": self.cur_end,
                                "label": label,
                                "key": sp_id, "final_key": final_sp_id,
                                **(d or {}), **diag_extra},
                               ensure_ascii=False, default=str) + "\n")

    def _write_cluster_naming_diag(self) -> None:
        """クラスタ確定イベントを diag に残し、実地検証で観測可能にする.

        (handoff §4-2) 書いたら消費（None化）して同一イベントの重複出力を
        防ぐ。cluster_namer が無い経路（従来モード）は不変。
        """
        s = self.state
        namer_diag = getattr(s.cluster_namer, "last_match", None)
        if namer_diag is None:
            return
        s.cluster_namer.last_match = None
        with contextlib.suppress(OSError), \
                open(s.diag_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ms": self.cur_ms, "end": self.cur_end,
                                "type": "cluster_naming", **namer_diag},
                               ensure_ascii=False, default=str) + "\n")

    def _commit_record(self, sp_id: str, rec_extra: dict) -> None:
        """確定した発話を records に積み、画面と外部フックへ流す."""
        from das.asr.live import ON_UTTERANCE

        s = self.state
        with s.state_lock:
            s.records.append({"ms": self.cur_ms, "end_ms": self.cur_end,
                              "speaker": sp_id, "text": self.cur_text.strip(),
                              **rec_extra})
            c = s.color_of(sp_id)
        if ON_UTTERANCE is not None:
            with contextlib.suppress(Exception):
                ON_UTTERANCE(s.disp_name(sp_id), self.cur_text.strip())
        _print_line(f"{c}[{fmt_ts(self.cur_ms)}] {s.disp_name(sp_id)}{RESET}: "
                    f"{self.cur_text.strip()}")

    def flush(self):
        """組み立て中の発話を確定し、話者を決めて records に積む.

        段の順番には理由がある（入れ替えると壊れる）:

          1. エコー破棄（テキスト）— 声紋の副作用より**前**。エコーで匿名話者が
             育って自動登録されるのを防ぐ（D2）
          2. 音声の切り出し → 3. 声紋判定 → 4. エコー破棄（声紋）
          5. 声紋に伴う台帳更新（補正・鋳造・合流）。鋳造の通知だけは後回し
          6. 帰属の決定（`decide_speaker`）→ 7. 人数上限（`constrain`）
          8. 席の実音声による決め直し（上限で落ちた分・ラベル頼りの分）
          9. 記録（diag）→ 10. records へ積む → 11. 遡及訂正 → 12. 保存
        """
        s = self.state
        if not self.cur_text.strip():
            self._clear_current(reset_timer=True)
            return
        label = str(self.cur_speaker)
        stt_speaker_unknown = _is_unknown_stt_speaker(self.cur_speaker)
        tracker = s.tracker
        # 相槌（「はい」等）は声紋の人物確定に使わず、UIでも薄く折りたためるよう印を付ける
        _is_backchannel = bool(_BACKCHANNEL_RE.match(self.cur_text.strip()))
        _minted_key: str | None = None
        _ms_known = self.cur_ms is not None and self.cur_end is not None
        _use_intervals = _ms_known and s.has_ai_speech_intervals()

        # --- 1. テキスト類似度によるエコー破棄（安全網, F2で前倒し） ---
        echo = self._text_echo_match(use_intervals=_use_intervals)
        if echo is not None:
            if self.args.vp_debug:
                _print_line(f"# テキスト安全網エコー除去({echo[0]})"
                            f" sim={echo[1]:.2f}"
                            f" ({self.cur_text.strip()[:40]}...)")
            self._note_echo_drop(echo[0], sim=echo[1])
            self._clear_current()
            return

        if tracker is not None:
            # --- 2-3. 音声の切り出しと声紋判定 ---
            wav = self._utterance_audio()
            if stt_speaker_unknown:
                sp_id, d, rec_extra, _classify_flags = UNSURE_SPEAKER, None, {}, {}
            else:
                sp_id, d, rec_extra, _classify_flags = self._classify_utterance(
                    wav, label=label, is_backchannel=_is_backchannel,
                    use_intervals=_use_intervals)
            # --- 4. 声紋によるAIエコー破棄 ---
            if sp_id is not None and is_ai_key(sp_id):
                if self.args.vp_debug:
                    _print_line(f"# AI声紋エコー除去: sp={sp_id}"
                                f" ({self.cur_text.strip()[:40]}...)")
                self._note_echo_drop(
                    "voiceprint", key=str(sp_id),
                    sim=d.get("sim") if isinstance(d, dict) else None)
                self._clear_current()
                return
            # --- 5. 声紋判定に伴う台帳更新・通知 ---
            _minted_key = self._apply_voiceprint_effects(d, rec_extra)
        else:
            sp_id = _stt_speaker_key(self.cur_speaker)
            rec_extra: dict[str, object] = {}
            d = None
            wav = None
            _classify_flags = {}

        # --- 6-7. 帰属の決定と人数上限 ---
        sp_id, final_sp_id, diag_extra = self._decide_and_constrain(
            sp_id, d=d, wav=wav, label=label, is_backchannel=_is_backchannel,
            rec_extra=rec_extra, classify_flags=_classify_flags)

        # --- 8. 席の実音声による決め直し（ハイブリッド限定） ---
        if s.seat_audio is not None and not _is_backchannel:
            final_sp_id = self._assign_seat(
                final_sp_id, sp_id=sp_id, d=d, wav=wav,
                rec_extra=rec_extra, diag_extra=diag_extra)

        # --- 8b. 未登録話者の門番（ハイブリッド限定, handoff §47） ---
        # ラベル不純で、長い発話なのに best sim が低い声は未登録の公算が高く、
        # 写像・席のどちらで寄せても誤帰属にしかならない（未登録の声には正解の
        # 出口が無い）。席の決め直しの**後**に掛けるのは、講義 2026-07-30 で
        # 席の決め直し自体が誤った（近い席しか選べない）ため。speaker_source を
        # 専用値にして、遡及訂正がこの未確定を席の参照で復活させないようにする。
        if (s.cluster_namer is not None and final_sp_id != UNSURE_SPEAKER
                and d is not None
                and impure_lowsim(d.get("kind"), len(self.cur_text.strip()),
                                  d.get("sim"))):
            final_sp_id = UNSURE_SPEAKER
            rec_extra["speaker_source"] = "impure_lowsim_guard"
            rec_extra["speaker_confidence"] = 0.0
            rec_extra["speaker_reason"] = "impure_label_long_low_sim"
            diag_extra["src"] = "impure_lowsim_guard"
            diag_extra["why"] = "impure_label_long_low_sim"

        # --- 9. 記録 ---
        if tracker is not None and (d is not None or stt_speaker_unknown):
            self._write_utterance_diag(label=label, sp_id=sp_id,
                                       final_sp_id=final_sp_id, d=d,
                                       diag_extra=diag_extra)
        self._write_cluster_naming_diag()
        if _is_backchannel:
            # 相槌は、話している人とは別人の可能性が高い（Aの話中にBが「はい」）。
            # 未確定化は final_sp_id の計算（constrain 入力を UNSURE にする）で
            # 済んでいるため、ここでは UI 折りたたみ用の bc フラグだけ付ける
            # (attribution_logic_review_2026-07.md D1: 旧 sp_id 上書きは
            # 直後の代入で消えるデッドコードだったので削除)。
            rec_extra["bc"] = True

        # --- 10-12. records へ積み、遡及訂正して保存する ---
        sp_id = final_sp_id   # constrain 済み（diag の final_key と同一値）
        if _minted_key is not None and sp_id == _minted_key:
            # 席を得られた場合だけ「追跡開始」を告げる。落ちた場合は席が無い
            # という警告（_note_constrain_drop）が別に出る。
            _disp = s.disp_name(_minted_key)
            s.add_sys(self.cur_ms, f"この声を「{_disp}」として追跡開始"
                                   "（名前は右側の登録欄から設定できます）")
            _print_line(f"# この声を「{_disp}」として追跡します"
                        "（名前は右側の登録欄から設定できます）")
        self._commit_record(sp_id, rec_extra)
        self._maybe_retro_reattribute()
        s.save()
        self._clear_current(reset_timer=True)

    def run(self, ws) -> RecvStatus:
        """WebSocket受信ループのメイン.

        stop（終了）または reset_requested（STT作り直し）で正常に抜ける。
        後者の場合、呼び出し側（run_session）が新しい ws を張り直す。
        """
        args = self.args
        try:
            while not self.state.stop.is_set() and not self.state.reset_requested.is_set():
                try:
                    raw = ws.recv()
                except Exception as e:
                    # 停止/作り直しで ws が閉じられた場合は正常終了扱い
                    if self.state.stop.is_set() or self.state.reset_requested.is_set():
                        break
                    _print_line(f"# STT WebSocket切断: {e}。再接続します")
                    return "disconnected"
                res = self.backend.parse_message(json.loads(raw), args.lang)
                if res.get("error_code") is not None:
                    _print_line(f"# エラー: {res['error_code']} - {res.get('error_message')}")
                    break
                partial = ""
                partial_sp = self.cur_speaker
                for token in res.get("tokens", []):
                    text = token.get("text") or ""
                    if text == "<end>":
                        self.flush()
                        continue
                    if not text:
                        continue
                    if token.get("is_final"):
                        sp = token.get("speaker")
                        if sp != self.cur_speaker:
                            self.flush()
                            self.cur_speaker = sp
                        if self.cur_ms is None:
                            self.cur_ms = self.state.stt_abs_ms(token.get("start_ms"))
                        if token.get("end_ms") is not None:
                            self.cur_end = self.state.stt_abs_ms(token["end_ms"])
                        self.cur_text += text
                        self.cur_last_token_time = time.monotonic()
                    else:
                        partial += text
                        partial_sp = token.get("speaker") or partial_sp
                # --- 強制flush（区切りは Soniox の <end>(エンドポイント)＋話者変化＋文字数）---
                if self.cur_text:
                    clen = len(self.cur_text)
                    if ((time.monotonic() - self.cur_last_token_time > self._FLUSH_TIMEOUT
                            or clen > self._FLUSH_HARD_CHARS)
                            or (clen > self._FLUSH_SOFT_CHARS
                                and self.cur_text.rstrip()[-1:] in "。？！.?!\n")):
                        self.flush()
                self.state.show_partial(partial_sp if partial else self.cur_speaker,
                                        self.cur_text + partial)
                if res.get("finished"):
                    self.flush()
                    _print_line("# 終了")
                    return "finished"
        except KeyboardInterrupt:
            pass
        finally:
            self.flush()
        return "ok"
