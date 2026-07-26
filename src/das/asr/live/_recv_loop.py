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
    voiceprint_endorses,
)
from ._constants import (
    _BACKCHANNEL_RE,
    RESET,
    UNSURE_SPEAKER,
    fmt_ts,
)
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

    def flush(self):
        from das.asr.live import ON_UTTERANCE

        s = self.state
        if not self.cur_text.strip():
            self.cur_text = ""
            self.cur_ms = None
            self.cur_end = None
            self.cur_last_token_time = time.monotonic()
            return
        label = str(self.cur_speaker)
        stt_speaker_unknown = _is_unknown_stt_speaker(self.cur_speaker)
        tracker = s.tracker
        agent = s.agent
        partner = s.partner
        # 相槌（「はい」等）は声紋の人物確定に使わず、UIでも薄く折りたためるよう印を付ける
        _is_backchannel = bool(_BACKCHANNEL_RE.match(self.cur_text.strip()))
        # --- テキスト類似度エコー判定（安全網, F2で前倒し） ---
        # 声紋トラッカーの副作用（文字数蓄積・自動登録）より前に評価する。エコーと
        # 判定したら classify を呼ばずに破棄し、漏れ込んだAI音声で匿名話者が蓄積・
        # 自動登録されるのを防ぐ（D2）。判定に必要なのは cur_text と agent/partner
        # だけで、声紋判定への依存はない。
        # AI再生区間との重なりでエコー窓を判定する（P2-1）。STT確定が遅れて壁時計の
        # エコー窓を過ぎた回り込みも、発話区間 [cur_ms, cur_end] が記録済みの再生区間と
        # 重なれば拾う。ms が無い/記録が無いときは従来の壁時計（in_echo_window）に倒す。
        _ms_known = self.cur_ms is not None and self.cur_end is not None
        _use_intervals = _ms_known and s.has_ai_speech_intervals()
        for _src_name, _src in [("agent", agent), ("partner", partner)]:
            if _src is None:
                continue
            if _src_name == "agent":
                _agent_echo = (s.overlaps_ai_speech(self.cur_ms, self.cur_end,
                                                    source="agent")
                               if _use_intervals else _src.in_echo_window)
                if not _agent_echo:
                    continue
            sim = _src._best_similarity(self.cur_text)
            if sim > 0.35:
                if self.args.vp_debug:
                    _print_line(f"# テキスト安全網エコー除去({_src_name})"
                                f" sim={sim:.2f}"
                                f" ({self.cur_text.strip()[:40]}...)")
                # 破棄した発話も echo_drop として diag に1行残す（「記録が無いのに
                # 登録通知だけある」状態を後から追えるようにする）。
                with contextlib.suppress(OSError), \
                        open(s.diag_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "ms": self.cur_ms, "end": self.cur_end,
                        "type": "echo_drop", "src": _src_name,
                        "sim": round(sim, 3),
                        "text": self.cur_text.strip()[:40],
                    }, ensure_ascii=False, default=str) + "\n")
                self.cur_text = ""
                self.cur_ms = None
                self.cur_end = None
                return
        # パートナー切断直後もエコー参照を短時間保持する（P2-4）。partner が None に
        # なってもテキスト安全網が効くよう、TTL 内の退役テキストとも照合する。
        _retired = s.recent_retired_echo_texts()
        if _retired:
            sim = _best_text_similarity(self.cur_text.strip(), _retired)
            if sim > 0.35:
                if self.args.vp_debug:
                    _print_line(f"# テキスト安全網エコー除去(retired)"
                                f" sim={sim:.2f}"
                                f" ({self.cur_text.strip()[:40]}...)")
                with contextlib.suppress(OSError), \
                        open(s.diag_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "ms": self.cur_ms, "end": self.cur_end,
                        "type": "echo_drop", "src": "retired",
                        "sim": round(sim, 3),
                        "text": self.cur_text.strip()[:40],
                    }, ensure_ascii=False, default=str) + "\n")
                self.cur_text = ""
                self.cur_ms = None
                self.cur_end = None
                return
        if tracker is not None:
            if self.cur_ms is not None and self.cur_end is not None and self.cur_end > self.cur_ms:
                with s.buf_lock:
                    abs_start = self.cur_ms * 32
                    abs_end = self.cur_end * 32
                    rel_start = max(abs_start - s.asr_pcm_buf_offset, 0)
                    rel_end = max(abs_end - s.asr_pcm_buf_offset, 0)
                    seg = bytes(s.asr_pcm_buf[rel_start: rel_end])
                wav = np.frombuffer(seg, dtype="<i2").astype(np.float32) / 32768.0
            else:
                wav = np.zeros(0, dtype=np.float32)
            _classify_flags: dict[str, object] = {}
            if stt_speaker_unknown:
                sp_id = UNSURE_SPEAKER
                d = None
                rec_extra: dict[str, object] = {}
            else:
                # 登録は声ごとの累積文字数で判定するので、この発話の文字数を渡す。
                # F2: エコー窓中（AIが発話中/直後）は count=False で蓄積・自動登録を
                # 抑止する。室内音響でAI声紋照合(AI_THRESH)が外れた漏れ込みが「新規
                # 話者の蓄積」に化けるのを塞ぐ。話者判定自体は従来どおり行う。正当な
                # 人間発話の登録がエコー窓ぶん遅れるのは許容（登録は累積制のため軽微）。
                # agent 側は再生区間の重なりで判定（P2-1, ms/記録が無ければ壁時計）。
                # partner 側は従来どおり壁時計。
                _agent_active = (
                    s.overlaps_ai_speech(self.cur_ms, self.cur_end, source="agent")
                    if _use_intervals
                    else (agent is not None
                          and (agent.ai_speaking or agent.in_echo_window))
                )
                _ai_active = (
                    _agent_active
                    or (partner is not None
                        and (partner.ai_speaking or partner.in_echo_window))
                )
                # count: 相槌は照合ごとスキップ。enroll: エコー窓中は照合・補正は
                # するが蓄積・登録はしない（P2-2）。エコー窓直後の人間の返答が声紋
                # 補正なしのラベル追従に落ちるのを防ぐ。
                _overlapped = self.overlaps_other(self.cur_ms, self.cur_end, label)
                _enroll = (not _is_backchannel) and not _ai_active
                sp_id = tracker.classify(
                    wav, self.cur_speaker,
                    overlapped=_overlapped,
                    count=not _is_backchannel,
                    enroll=_enroll,
                    chars=len(self.cur_text.strip()))
                d = tracker.last
                rec_extra: dict[str, object] = {}
                # classify に実際に渡した条件（オフライン再生用, handoff §23）。
                # overlapped は発話系列から再現できるが、enroll はエコー窓＝AI再生
                # 区間に依存し記録からは再現できないため必ず残す。
                _classify_flags = {"ov": _overlapped, "enr": _enroll,
                                   "chars": len(self.cur_text.strip())}
            # --- 声紋ベースのAIエコー除去 ---
            if sp_id is not None and is_ai_key(sp_id):
                if self.args.vp_debug:
                    _print_line(f"# AI声紋エコー除去: sp={sp_id}"
                                f" ({self.cur_text.strip()[:40]}...)")
                self.cur_text = ""
                self.cur_ms = None
                self.cur_end = None
                return
            if d and d["kind"] == "補正":
                note = (f"声紋でラベル{d['label']}の取り違えを修正"
                        f"（類似{d['sim']:.2f}、放置なら{s.disp_name(d['prev'])}の発言になっていた）")
                rec_extra = {"vp": "補正", "note": note}
                _print_line(f"# ⚡補正: {note}")
            elif d and d["kind"] == "自動登録":
                if d["rename"]:
                    s.rekey(*d["rename"])
                self._link_mint_to_cluster(d["name"])
                display_name = s.disp_name(d["name"])
                s.add_sys(self.cur_ms, f"この声を「{display_name}」として追跡開始"
                                       "（名前は右側の登録欄から設定できます）")
                _print_line(f"# この声を「{display_name}」として追跡します"
                            "（名前は右側の登録欄から設定できます）")
            elif d and d["kind"] == "合流":
                if d["rename"]:
                    s.rekey(*d["rename"])
                if self.args.vp_debug:
                    _print_line(f"# 合流: ラベル{d['label']}→{d['name']}")
            elif self.args.vp_debug and d:
                extra = f" 類似{d['sim']:.2f}({d['name']})" if "sim" in d else ""
                _print_line(f"# vp判定[{d['kind']}]{extra}")
        else:
            sp_id = _stt_speaker_key(self.cur_speaker)
            rec_extra: dict[str, object] = {}
            d = None
            wav = None
            _classify_flags = {}
        # --- 話者帰属の決定（声紋→Resolver→クラスタ確定/匿名キー） ---
        # 判定フローは _attribution.decide_speaker に一本化（構成ごとの分岐・
        # 各ステップの根拠はモジュール docstring 参照）。constrain（参加人数
        # 上限・closed roster）はこの後の final_sp_id 計算で適用する。
        # diag_extra には「判定の入力」を集める（records ではなく diag に出す）。
        # 目的はオフライン再生: これが無いとクラスタ層の入力が実行後に失われ、
        # 記録から本番コードを回せない（handoff §23）。
        diag_extra: dict[str, object] = dict(_classify_flags)
        sp_id = decide_speaker(
            s, sp_id=sp_id, d=d, wav=wav,
            start_ms=self.cur_ms, end_ms=self.cur_end,
            rec_extra=rec_extra, vp_debug=self.args.vp_debug,
            diag_extra=diag_extra,
        )
        # 判定がどの経路で決まったかを diag にも残す（診断のみ・挙動不変）。
        # speaker_source は records にしか無く、records は終了時に永続化されない
        # （transcripts に残るのは diag/turns/wav だけ）。そのため
        # eval/decompose_attribution.py が「この誤帰属は 3d の門番で止められる
        # 経路か、それとも STT フォールバックか」を分けられなかった
        # （handoff §26.6）。短いキー名なのは diag が1発話1行で膨らむため。
        if rec_extra.get("speaker_source") is not None:
            diag_extra["src"] = rec_extra["speaker_source"]
            diag_extra["why"] = rec_extra.get("speaker_reason")
        if self.cur_ms is not None and self.cur_end is not None:
            self.recent_segs.append((self.cur_ms, self.cur_end, label))
            del self.recent_segs[:-12]
        # constrain 後に records へ入る最終キーを先に計算し、diag へ併記する。
        # 従来の diag は constrain 前の key しか持たず、「resolver は正しいキーを
        # 選んだのに constrain で未確定に落ちた」事象の切り分けができなかった
        # (docs/design/handoff_2026-07-14_unregistered_speakers.md 参照)。
        # 既存フィールド（key 等）は変えず final_key を追加のみ（diag 消費側の互換維持）。
        final_sp_id = s.constrain_human_speaker_key(
            UNSURE_SPEAKER if _is_backchannel else sp_id)
        # --- 席落ちの割当て（クラスタ分裂の回収。handoff §27。ハイブリッド限定） ---
        # 上流はキーを決めていたのに席上限で落ちた発話は、実測で**全て**
        # @diar:N＝既に席を持つ人の分裂だった。参加人数の設定上そこに新しい
        # 参加者は入れないので、残る問いは「席を持つN人のうち誰か」だけになる。
        # この発話1件に限って席の実音声と比べ、最も似た人へ寄せる（確定は
        # 書かない＝可逆。§15.12 の「不可逆な操作は高確信を要求」と衝突しない）。
        if s.seat_audio is not None and not _is_backchannel:
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
            _kind = d.get("kind") if d is not None else None
            if _kind in _LABEL_ONLY_KINDS:
                # 根拠がSTTラベルしかない kind は、上流のキーを信用せず
                # 席の実音声で決め直す（handoff §27.12）。
                #   ラベル不純: そのラベルが複数人を混載していると分かっている
                #   ラベル継続: 声紋照合が成立せず、ラベルの過去の対応を
                #               引き継いでいるだけ
                # どちらも「ラベルに基づく推測」であり、声を直接比べたほうが
                # 強い。実測で 正解 71.0%→79.2% / 誤帰属 19.7%→13.6%
                # （検証4本では 82.0%）。棄権していた分（未確定）も
                # 決めていた分（誤帰属の48%を占めていた）も同じ規則で扱う。
                _reason = "label_only_kind_resolved_by_seat_audio"
            elif final_sp_id == UNSURE_SPEAKER and sp_id != UNSURE_SPEAKER:
                # 上流は決めていたのに席上限で落ちた分（§27.8 の本体）。
                _reason = "seat_full_nearest_seat_audio"
            else:
                _reason = None
                # 参照は「声紋層が高信頼だった発話」だけで作る。全発話で作ると
                # 席の参照そのものが汚れる（実測: ある席は GT 純度 38%）。
                # 高信頼4種に絞ると純度は 95-100% に上がり、寄せ先の的中も
                # 67%→70%、誤帰属の増分も 3.9→3.4pt に下がる（handoff §27.9）。
                if (final_sp_id != UNSURE_SPEAKER
                        and _kind in _VOICEPRINT_RELIABLE_KINDS):
                    s.seat_audio.observe(final_sp_id, wav)
            if _reason is not None:
                picked = s.seat_audio.nearest(wav)
                if picked is not None:
                    final_sp_id = picked[0]
                    rec_extra["speaker_source"] = "seat_assign"
                    rec_extra["speaker_confidence"] = round(picked[1], 3)
                    rec_extra["speaker_reason"] = _reason
                    diag_extra["seat"] = s.seat_audio.last_pick
        if tracker is not None and tracker.last is not None:
            try:
                with open(s.diag_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"ms": self.cur_ms, "end": self.cur_end, "label": label,
                                        "key": sp_id, "final_key": final_sp_id,
                                        **tracker.last, **diag_extra},
                                       ensure_ascii=False, default=str) + "\n")
            except OSError:
                pass
        # クラスタ確定イベントを diag に残し、実地検証で観測可能にする
        # (docs/design/handoff_2026-07-14_unregistered_speakers.md §4-2)。
        # 書いたら消費（None化）して同一イベントの重複出力を防ぐ。
        # cluster_namer が無い経路（従来モード）は不変。
        _namer_diag = getattr(s.cluster_namer, "last_match", None)
        if _namer_diag is not None:
            s.cluster_namer.last_match = None
            with contextlib.suppress(OSError), \
                    open(s.diag_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ms": self.cur_ms, "end": self.cur_end,
                                    "type": "cluster_naming", **_namer_diag},
                                   ensure_ascii=False, default=str) + "\n")
        if _is_backchannel:
            # 相槌は、話している人とは別人の可能性が高い（Aの話中にBが「はい」）。
            # 未確定化は final_sp_id の計算（constrain 入力を UNSURE にする）で
            # 済んでいるため、ここでは UI 折りたたみ用の bc フラグだけ付ける
            # (docs/design/attribution_logic_review_2026-07.md D1: 旧 sp_id 上書きは
            # 直後の代入で消えるデッドコードだったので削除)。
            rec_extra["bc"] = True
        sp_id = final_sp_id   # constrain 済み（diag の final_key と同一値）
        with s.state_lock:
            s.records.append({"ms": self.cur_ms, "end_ms": self.cur_end,
                              "speaker": sp_id, "text": self.cur_text.strip(),
                              **rec_extra})
            c = s.color_of(sp_id)
        if ON_UTTERANCE is not None:
            with contextlib.suppress(Exception):
                ON_UTTERANCE(s.disp_name(sp_id), self.cur_text.strip())
        _print_line(f"{c}[{fmt_ts(self.cur_ms)}] {s.disp_name(sp_id)}{RESET}: {self.cur_text.strip()}")
        s.save()
        self.cur_text = ""
        self.cur_ms = None
        self.cur_end = None
        self.cur_last_token_time = time.monotonic()

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
