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

from ._constants import (
    _BACKCHANNEL_RE,
    PYANNOTE_CLUSTER_OVERLAP_MIN_RATIO,
    RESET,
    UNSURE_SPEAKER,
    fmt_ts,
)
from ._diarization import TimeSegment, has_overlapping_speakers
from ._ui import _print_line
from ._voice_profiles import _best_text_similarity

_VOICEPRINT_RELIABLE_KINDS = {"声紋一致", "補正", "自動登録", "合流"}
# 声紋で判定できない発話は VoiceProfiles._classify が「ラベル継続」（そのSTTラベル
# の、声紋照合の成功で確定した現在の対応先）を返す（2026-07-14 再設計。照合失敗で
# 対応を破棄する旧仕様は同一人物を #ラベルと人物Nに分裂させ、オフライン再生評価
# eval/replay_attribution.py で 1:1帰属精度44%→継続化を含む再設計で79%）。
# ラベル継続・蓄積中は _VOICEPRINT_RELIABLE_KINDS に含めない＝Resolver 上は
# 高信頼の声紋判定として扱わない。相槌レコードの最終表示を未確定へ落とす規則は
# 本ファイル flush 側にある（相槌は聞き手が打つ＝直前話者とは別人が多い）。
# かつてここにあったハイブリッド限定の _HYBRID_UNTRUSTED_FOLLOW_KINDS による抑制は
# 冗長になったため撤去（二重実装を残さない）。ハイブリッドの帰属優先度
# 「声紋一致 > pyannoteクラスタ(名寄せ済み) > 未確定」は tracker が UNSURE を
# 返すことで従来どおり成立する（UNSURE は stt_fallback の参加者化もしない）。
_UNKNOWN_STT_SPEAKERS = {"", "none", "null", "unknown", "uu", UNSURE_SPEAKER}
RecvStatus = Literal["ok", "finished", "disconnected"]


def _is_unknown_stt_speaker(speaker) -> bool:
    return str(speaker).strip().lower() in _UNKNOWN_STT_SPEAKERS


def _stt_speaker_key(speaker) -> str:
    if _is_unknown_stt_speaker(speaker):
        return UNSURE_SPEAKER
    return "#" + str(speaker)


def _merged_diarization_speaker_key(s: SessionState, raw_cluster: str,
                                    source: str, speaker: str,
                                    *, duration_ms: int) -> str:
    """クラスタ間名寄せを反映した匿名キー解決（s.cluster_namer 有効時のみ呼ぶ）.

    設計: docs/design/handoff_2026-07-14_unregistered_speakers.md §3。
    - 名寄せ成立済み (canonical != raw): 新規参加者を作らず canonical のキーへ
      帰属させる。吸収側に別キーが発行済みなら rekey で過去レコードごと遡及統合
      する（§3 の3）。どちらも未キーなら canonical の source/speaker で
      key_for_diarization_speaker を呼び、ヒステリシスの pending を canonical に
      集約する。
    - 名寄せ不成立: 参加人数上限まで人間スロットが埋まっている場合のみ、最近傍
      クラスタの既存キーへ統合を試みる（§3 の2: 昇格の厳格化）。ただし類似度が
      namer.merge_sim（名寄せ用の類似度下限、既定は tracker.dedupe と同値）
      未満なら「全く似ていない新話者」なので統合しない。それも不可なら従来どおり
      key_for_diarization_speaker へ（最終的に constrain_human_speaker_key で
      未確定に落ちる＝安全側の既存挙動）。
    """
    namer = s.cluster_namer
    canonical = namer.canonical_cluster(raw_cluster)
    if canonical != raw_cluster:
        # 吸収側に溜まっていたヒステリシス pending を canonical へ合算する
        # （同一人物なので分裂で参加者化が二重に遅れないようにする。§3 参照）。
        s.merge_diarization_pending(raw_cluster, canonical)
        canonical_key = s.diarization_speaker_keys.get(canonical)
        absorbed_key = s.diarization_speaker_keys.pop(raw_cluster, None)
        if canonical_key is not None:
            if absorbed_key is not None and absorbed_key != canonical_key:
                # 吸収側に既に @diar:N を発行済み → 過去レコードごと遡及統合。
                s.rekey(absorbed_key, canonical_key)
            return canonical_key
        if absorbed_key is not None:
            # canonical 側が未キーなら吸収側の発行済みキーを canonical へ付け替えて
            # 再利用する（過去レコードのキーを安定に保つ）。
            s.diarization_speaker_keys[canonical] = absorbed_key
            return absorbed_key
        c_source, _, c_speaker = canonical.partition(":")
        return s.key_for_diarization_speaker(c_source, c_speaker,
                                             duration_ms=duration_ms)
    if (raw_cluster not in s.diarization_speaker_keys
            and s.human_slot_budget_exhausted()):
        nearest = namer.nearest_cluster(raw_cluster)
        if nearest is not None:
            nearest_cluster, nearest_sim = nearest
            # 下限閾値は namer.merge_sim（名寄せと同じ独立ノブ。既定は従来どおり
            # tracker.dedupe と同値＝挙動不変。review C6/P4 で文脈分離）。
            # 無条件の最近傍統合だと類似度0.0でも既存参加者に張り付くため、
            # merge_sim 未満は統合せず従来経路へ（constrain で未確定＝安全側）。
            if nearest_sim >= namer.merge_sim:
                nearest_key = s.diarization_speaker_keys.get(nearest_cluster)
                if nearest_key is not None:
                    return nearest_key
    return s.key_for_diarization_speaker(source, speaker, duration_ms=duration_ms)


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
                sp_id = tracker.classify(
                    wav, self.cur_speaker,
                    overlapped=self.overlaps_other(self.cur_ms, self.cur_end, label),
                    count=not _is_backchannel,
                    enroll=(not _is_backchannel) and not _ai_active,
                    chars=len(self.cur_text.strip()))
                d = tracker.last
                rec_extra: dict[str, object] = {}
            # --- 声紋ベースのAIエコー除去 ---
            if (sp_id is not None
                    and sp_id.startswith("__") and sp_id.endswith("__")):
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
        # 相槌でも「クラスタ根拠」があれば帰属を通すためのフラグ（下の相槌規則参照。
        # Chiba実測 2026-07-16_1551: 未確定21件中18件が1秒未満の相槌で、Sonioxは
        # 3人の相槌を同一STTラベルに混ぜる＝STTラベル由来の推測は不信のままだが、
        # pyannoteクラスタは声で束ねるため相槌でも分離できる。
        # handoff_2026-07-14_unregistered_speakers.md §15.4）。
        bc_cluster_evidence = False
        if (self.cur_ms is not None and self.cur_end is not None
                and self.cur_end > self.cur_ms):
            voiceprint_speaker = None
            voiceprint_confidence = None
            if (d and d.get("kind") in _VOICEPRINT_RELIABLE_KINDS
                    and sp_id is not None
                    and not str(sp_id).startswith("#")):
                voiceprint_speaker = str(sp_id)
                # VoiceProfiles側のしきい値を通った判定なのでResolver上は高信頼扱いにする。
                # 意図的に固定値 1.0 を渡す＝「信頼4種の声紋判定は diarization に
                # 無条件で勝つ」。Resolver の voiceprint_high_confidence(0.70) は
                # この経路では比較として機能しない（実simを渡すとしきい値が生きて
                # 挙動が変わるので、変更時は要再評価。
                # docs/design/attribution_logic_review_2026-07.md C8）。
                voiceprint_confidence = 1.0
            diarization_events = s.diarization_window(self.cur_ms, self.cur_end)
            resolved = s.speaker_resolver.resolve(
                utterance=TimeSegment(self.cur_ms, self.cur_end),
                stt_speaker=str(sp_id),
                diarization_events=diarization_events,
                voiceprint_speaker=voiceprint_speaker,
                voiceprint_confidence=voiceprint_confidence,
            )
            if resolved.source != "stt":
                if resolved.source == "voiceprint":
                    sp_id = resolved.speaker
                    rec_extra["speaker_source"] = resolved.source
                    rec_extra["speaker_confidence"] = round(resolved.confidence, 3)
                    rec_extra["speaker_reason"] = resolved.reason
                else:
                    rec_extra["diarization_raw_speaker"] = resolved.speaker
                    # --- ハイブリッド構成: pyannoteクラスタ単位の声紋名前付け ---
                    # (docs/design/pyannote_live1_trial_2026-07-09.md §8.4/§9)。
                    # s.cluster_namer は --vp-cluster-naming 指定時のみ設定される。
                    # 未設定なら以下は素通りし、従来どおり key_for_diarization_speaker
                    # による匿名キー付与だけで完結する（Soniox単独/他provider配線は不変）。
                    raw_cluster = f"{resolved.source}:{resolved.speaker}"
                    cluster_overlap = False
                    cluster_name = None
                    if s.cluster_namer is not None:
                        cluster_overlap = has_overlapping_speakers(
                            diarization_events, self.cur_ms, self.cur_end,
                            min_ratio=PYANNOTE_CLUSTER_OVERLAP_MIN_RATIO,
                        )
                        cluster_name = s.cluster_namer.observe(
                            raw_cluster, wav, overlapped=cluster_overlap)
                    if cluster_name is not None:
                        # クラスタの累積声紋が確定名に達した。以後このクラスタの発話は
                        # この名前に帰属する。過去にこのクラスタへ既に匿名キー
                        # (@diar:N)を発行済みだった場合、既存の rekey 機構で過去分も
                        # まとめて確定名へ付け替える（設計点4: 低コストな遡及リネーム）。
                        # クラスタ間名寄せで吸収されたクラスタはキーが canonical 側で
                        # 管理されるため、raw/canonical 両方のキーを確定名へ統合する
                        # (docs/design/handoff_2026-07-14_unregistered_speakers.md §3)。
                        # 統合後は diarization_speaker_keys も確定名に付け替え、
                        # 以後の最近傍統合が古い @diar:N を復活させないようにする。
                        _canonical = s.cluster_namer.canonical_cluster(raw_cluster)
                        for _cluster in {raw_cluster, _canonical}:
                            prior_key = s.diarization_speaker_keys.get(_cluster)
                            if prior_key is not None and prior_key != cluster_name:
                                s.rekey(prior_key, cluster_name)
                                s.diarization_speaker_keys[_cluster] = cluster_name
                        sp_id = cluster_name
                        bc_cluster_evidence = True
                        rec_extra["speaker_source"] = "cluster_voiceprint"
                        rec_extra["speaker_confidence"] = 1.0
                        rec_extra["speaker_reason"] = "pyannote_cluster_voiceprint_confirmed"
                    elif cluster_overlap:
                        # 重複発話（複数の生クラスタが同時にこの区間を占める）区間は
                        # 声が混ざり声紋があてにならないため、安全側で未確定にする
                        # （設計点5）。
                        sp_id = UNSURE_SPEAKER
                        rec_extra["speaker_source"] = "cluster_overlap"
                        rec_extra["speaker_confidence"] = 0.0
                        rec_extra["speaker_reason"] = "multiple_diarization_speakers_overlap"
                    else:
                        if s.cluster_namer is not None:
                            # クラスタ間名寄せを反映したキー解決（遡及統合・
                            # max-speakers超過時の最近傍統合を含む。§3 参照）。
                            # cluster_namer が無い場合は下の従来コードのままで、
                            # Soniox単独/pyannote単独の挙動は一切変えない。
                            sp_id = _merged_diarization_speaker_key(
                                s, raw_cluster, resolved.source, resolved.speaker,
                                duration_ms=self.cur_end - self.cur_ms,
                            )
                            # クラスタ由来の匿名キー（@diar:N / canonical）も相槌の
                            # 帰属根拠として信用する（UNSURE ならどのみち未確定）。
                            bc_cluster_evidence = sp_id != UNSURE_SPEAKER
                        else:
                            sp_id = s.key_for_diarization_speaker(
                                resolved.source, resolved.speaker,
                                duration_ms=self.cur_end - self.cur_ms,
                            )
                        rec_extra["speaker_source"] = resolved.source
                        rec_extra["speaker_confidence"] = round(resolved.confidence, 3)
                        rec_extra["speaker_reason"] = resolved.reason
                if self.args.vp_debug and resolved.source != "voiceprint":
                    # peek_disp_name（割当てなし）: この時点の sp_id は constrain 前で、
                    # 未確定に落ちる可能性がある。debug 表示のために disp_name で
                    # ラベル文字を先食いすると幻キーがスロットを消費する
                    # (docs/design/handoff_2026-07-14_unregistered_speakers.md 参照)。
                    _print_line(
                        f"# diarization: {s.peek_disp_name(sp_id)} ({resolved.speaker})"
                        f" conf={rec_extra.get('speaker_confidence', resolved.confidence):.2f}"
                        f" {rec_extra.get('speaker_reason', resolved.reason)}"
                    )
            elif (s.diarization_provider is not None
                  and voiceprint_speaker is None
                  and sp_id != UNSURE_SPEAKER):
                rec_extra["stt_raw_speaker"] = resolved.speaker
                sp_id = s.key_for_stt_fallback_speaker(
                    resolved.speaker, duration_ms=self.cur_end - self.cur_ms
                )
                rec_extra["speaker_source"] = "stt_fallback"
                rec_extra["speaker_confidence"] = 0.0
                rec_extra["speaker_reason"] = "diarization_no_confident_overlap_stt_fallback"
        if self.cur_ms is not None and self.cur_end is not None:
            self.recent_segs.append((self.cur_ms, self.cur_end, label))
            del self.recent_segs[:-12]
        # constrain 後に records へ入る最終キーを先に計算し、diag へ併記する。
        # 従来の diag は constrain 前の key しか持たず、「resolver は正しいキーを
        # 選んだのに constrain で未確定に落ちた」事象の切り分けができなかった
        # (docs/design/handoff_2026-07-14_unregistered_speakers.md 参照)。
        # 既存フィールド（key 等）は変えず final_key を追加のみ（diag 消費側の互換維持）。
        # 相槌規則: 相槌は聞き手が打つ＝直前話者と別人が多く、Sonioxは3人の相槌を
        # 同一STTラベルに混ぜる（Chiba実測）ため、STTラベル・声紋継続由来の帰属は
        # 未確定に落とす（従来どおり）。ただし pyannote クラスタ由来の根拠
        # （確定名 cluster_voiceprint / クラスタ匿名キー）がある場合だけは通す:
        # クラスタは「声の束ね」で相槌でも話者分離できることが実証済み
        # （trial §8「未確定回収はpyannote優位」、handoff §15.4）。
        final_sp_id = s.constrain_human_speaker_key(
            sp_id if (not _is_backchannel or bc_cluster_evidence)
            else UNSURE_SPEAKER)
        if tracker is not None and tracker.last is not None:
            try:
                with open(s.diag_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"ms": self.cur_ms, "end": self.cur_end, "label": label,
                                        "key": sp_id, "final_key": final_sp_id,
                                        **tracker.last},
                                       ensure_ascii=False, default=str) + "\n")
            except OSError:
                pass
        # 名寄せ・クラスタ確定イベントを diag に残し、実地検証で観測可能にする
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
