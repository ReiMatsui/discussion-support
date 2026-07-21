"""発話単位の話者帰属決定（flush の統合層）.

3つのID空間（STTラベル→声紋 sp_map / pyannote生クラスタ→@diar:N / 人物N・実名
プロファイル）を発話ごとに突き合わせて最終キーを決める判定は、従来
RecvLoop.flush() 内の手続き的な if 連鎖として実装されており、それが「事実上の
統合層」になっていた（docs/design/attribution_logic_review_2026-07.md §2 の指摘）。
本モジュールはその判定を1本の明示的なフロー（decide_speaker）として切り出した
もの（2026-07-17 再編）。**挙動は flush 時代と同一**（回帰ゲート: テストスイート
全件＋eval/replay_attribution.py 5本の出力一致。handoff §17 のベースライン表）。

判定フロー（上から順に評価し、確定した段階で終了）:

  0. 発話区間が不明/長さゼロ → 声紋段(classify)の結果をそのまま使う
  1. SpeakerResolver が情報源を選ぶ（声紋 > diarization > STT。
     信頼4種の声紋判定は conf=1.0 固定で渡す＝diarization に無条件で勝つ）
  2. 声紋が勝った → その人物キー
  3. diarization が勝った（ハイブリッド構成では続けてクラスタ層で解決）:
     3a. クラスタ確定名あり → 確定名（過去の匿名キーは rekey で遡及統合）
     3b. 重なり発話 → 未確定（声が混ざり声紋があてにならない）
     3c. それ以外 → 匿名キー解決（_anonymous_cluster_key: 名寄せ・
         最近傍統合・ヒステリシスを含む。cluster_namer が無い構成は
         SessionState.key_for_diarization_speaker へ直行）
     3d. 不純ラベル門番（ハイブリッドのみ, handoff §18.8）: 声紋層が
         「ラベル不純」で棄権した発話は、声紋1位候補が回収先と一致する
         ときだけ 3a/3c の結果を採用し、それ以外は未確定に差し替える
         （台帳・蓄積の副作用は 3a-3c のまま保存）
  4. STT に落ちた（diarization 供給ありで重なりが無く、声紋も無い）
     → STT フォールバックキー
  最後に呼び出し側（flush）が constrain_human_speaker_key で参加人数上限・
  closed roster を適用する（相槌は constrain 入力を未確定に差し替える）。

P1（発行前スロット判定, handoff §16。2026-07-16 見送り・設計保存）を将来
導入する場合、変更点は 3c の _anonymous_cluster_key の内部（発行前に
スロット判定し、満杯なら発行せず待つ）と、相槌を pending に加算しない分岐
（flush 側）に閉じる。ステップ0-2・3a-3b はそのまま。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from ._session_state import SessionState

from ._constants import (
    CLUSTER_IMPURE_RECOVERY_ENDORSE_MIN_SIM,
    PYANNOTE_CLUSTER_OVERLAP_MIN_RATIO,
    UNSURE_SPEAKER,
)
from ._diarization import TimeSegment, has_overlapping_speakers
from ._ui import _print_line

_VOICEPRINT_RELIABLE_KINDS = {"声紋一致", "補正", "自動登録", "合流"}
# 声紋で判定できない発話は VoiceProfiles._classify が「ラベル継続」（そのSTTラベル
# の、声紋照合の成功で確定した現在の対応先）を返す（2026-07-14 再設計。照合失敗で
# 対応を破棄する旧仕様は同一人物を #ラベルと人物Nに分裂させ、オフライン再生評価
# eval/replay_attribution.py で 1:1帰属精度44%→継続化を含む再設計で79%）。
# ラベル継続・蓄積中は _VOICEPRINT_RELIABLE_KINDS に含めない＝Resolver 上は
# 高信頼の声紋判定として扱わない。相槌レコードの最終表示を未確定へ落とす規則は
# RecvLoop.flush 側にある（相槌は聞き手が打つ＝直前話者とは別人が多い）。
# かつてここにあったハイブリッド限定の _HYBRID_UNTRUSTED_FOLLOW_KINDS による抑制は
# 冗長になったため撤去（二重実装を残さない）。ハイブリッドの帰属優先度
# 「声紋一致 > pyannoteクラスタ(名寄せ済み) > 未確定」は tracker が UNSURE を
# 返すことで従来どおり成立する（UNSURE は stt_fallback の参加者化もしない）。


def _anonymous_cluster_key(s: SessionState, raw_cluster: str,
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
      クラスタの既存キーへ統合を試みる（§3 の2: 昇格の厳格化）。ただし
      namer.merge_sim が None（既定＝統合無効, handoff §15.12）または類似度が
      それ未満なら統合しない。その場合は従来どおり key_for_diarization_speaker へ
      （最終的に constrain_human_speaker_key で未確定に落ちる＝安全側の既存挙動）。
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
            # 下限閾値は namer.merge_sim（名寄せと同じ独立ノブ）。None は統合無効
            # ＝既定（クラスタ埋め込み同士の比較に安全な閾値が存在しないことが
            # 実測で判明したため。handoff §15.12、_cluster_naming.py 参照）。
            if namer.merge_sim is not None and nearest_sim >= namer.merge_sim:
                nearest_key = s.diarization_speaker_keys.get(nearest_cluster)
                if nearest_key is not None:
                    return nearest_key
    return s.key_for_diarization_speaker(source, speaker, duration_ms=duration_ms)


def _voiceprint_claim(d, sp_id) -> tuple[str | None, float | None]:
    """ステップ1に渡す声紋の主張 (speaker, confidence) を返す.

    信頼4種（声紋一致・補正・自動登録・合流）の判定のみを Resolver への
    主張として通す。confidence は意図的に固定値 1.0＝「信頼4種の声紋判定は
    diarization に無条件で勝つ」。Resolver の voiceprint_high_confidence(0.70)
    はこの経路では比較として機能しない（実simを渡すとしきい値が生きて
    挙動が変わるので、変更時は要再評価。
    docs/design/attribution_logic_review_2026-07.md C8）。
    """
    if (d and d.get("kind") in _VOICEPRINT_RELIABLE_KINDS
            and sp_id is not None
            and not str(sp_id).startswith("#")):
        return str(sp_id), 1.0
    return None, None


def _cluster_attribution(s: SessionState, resolved, *, d, wav,
                         start_ms: int, end_ms: int,
                         diarization_events, rec_extra: dict) -> str:
    """ステップ3: diarization が勝った発話をクラスタ層で解決し、不純門番を適用する.

    3a-3c で通常どおりキーを解決した後、ステップ3d（不純ラベル門番,
    handoff §18.8）が最終キーだけを差し替える。台帳・蓄積の副作用
    （observe / rekey / ヒステリシス pending / キー発行）は従来どおり
    実行される——オフライン反実仮想（記録ランの最終ラベルのみ差し替えて
    再採点）と実装の意味論を一致させるため、判定は出力段に置く。
    """
    sp_id = _cluster_attribution_raw(
        s, resolved, wav=wav, start_ms=start_ms, end_ms=end_ms,
        diarization_events=diarization_events, rec_extra=rec_extra)
    # --- 3d. 不純ラベル門番（ハイブリッド構成のみ。handoff §18.8） ---
    # 声紋層が「ラベル不純」（直近の照合成功が複数人物に割れたSTTラベル）で
    # 棄権した発話のクラスタ回収は、Chiba 12会話の実測で通算正解45%
    # （0632では0/17）と当てにならず、誤帰属の主経路だった（§18.6）。
    # ただし「その発話自身の声紋1位候補が回収先と一致」する回収は
    # 開発5会話で正解37/誤り6、検証5会話でも成立。弱い声紋の裏付けが
    # あるときだけ回収を通し、それ以外は未確定に落とす（誤帰属＞未確定の
    # 優先。§15.3）。pyannote単独・Soniox単独（cluster_namer なし）は不変。
    if (s.cluster_namer is not None and sp_id != UNSURE_SPEAKER
            and d and d.get("kind") == "ラベル不純"):
        endorsed = (d.get("name") == sp_id
                    and float(d.get("sim") or 0.0)
                    >= CLUSTER_IMPURE_RECOVERY_ENDORSE_MIN_SIM)
        if not endorsed:
            rec_extra["speaker_source"] = "cluster_impure_label"
            rec_extra["speaker_confidence"] = 0.0
            rec_extra["speaker_reason"] = (
                "impure_stt_label_without_voiceprint_endorsement")
            return UNSURE_SPEAKER
    return sp_id


def _cluster_attribution_raw(s: SessionState, resolved, *, wav,
                             start_ms: int, end_ms: int,
                             diarization_events, rec_extra: dict) -> str:
    """ステップ3a-3c: クラスタ確定名／重なり未確定／匿名キーの解決本体.

    ハイブリッド構成 (--vp-cluster-naming, s.cluster_namer あり) では
    pyannoteクラスタ単位の声紋名前付け
    (docs/design/pyannote_live1_trial_2026-07-09.md §8.4/§9) を経由する。
    cluster_namer が無ければ従来どおり key_for_diarization_speaker による
    匿名キー付与だけで完結する（Soniox単独/他provider配線は不変）。
    """
    rec_extra["diarization_raw_speaker"] = resolved.speaker
    raw_cluster = f"{resolved.source}:{resolved.speaker}"
    cluster_overlap = False
    cluster_name = None
    if s.cluster_namer is not None:
        cluster_overlap = has_overlapping_speakers(
            diarization_events, start_ms, end_ms,
            min_ratio=PYANNOTE_CLUSTER_OVERLAP_MIN_RATIO,
        )
        cluster_name = s.cluster_namer.observe(
            raw_cluster, wav, overlapped=cluster_overlap)
    if cluster_name is not None:
        # --- 3a. クラスタの累積声紋が確定名に達した ---
        # 以後このクラスタの発話はこの名前に帰属する。過去にこのクラスタへ既に
        # 匿名キー(@diar:N)を発行済みだった場合、既存の rekey 機構で過去分も
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
        rec_extra["speaker_source"] = "cluster_voiceprint"
        rec_extra["speaker_confidence"] = 1.0
        rec_extra["speaker_reason"] = "pyannote_cluster_voiceprint_confirmed"
        return cluster_name
    if cluster_overlap:
        # --- 3b. 重複発話（複数の生クラスタが同時にこの区間を占める） ---
        # 声が混ざり声紋があてにならないため、安全側で未確定にする（設計点5）。
        rec_extra["speaker_source"] = "cluster_overlap"
        rec_extra["speaker_confidence"] = 0.0
        rec_extra["speaker_reason"] = "multiple_diarization_speakers_overlap"
        return UNSURE_SPEAKER
    # --- 3c. 匿名キー解決 ---
    if s.cluster_namer is not None:
        # クラスタ間名寄せを反映したキー解決（遡及統合・max-speakers超過時の
        # 最近傍統合を含む。§3 参照）。cluster_namer が無い場合は下の従来
        # コードのままで、Soniox単独/pyannote単独の挙動は一切変えない。
        sp_id = _anonymous_cluster_key(
            s, raw_cluster, resolved.source, resolved.speaker,
            duration_ms=end_ms - start_ms,
        )
    else:
        sp_id = s.key_for_diarization_speaker(
            resolved.source, resolved.speaker,
            duration_ms=end_ms - start_ms,
        )
    rec_extra["speaker_source"] = resolved.source
    rec_extra["speaker_confidence"] = round(resolved.confidence, 3)
    rec_extra["speaker_reason"] = resolved.reason
    return sp_id


def decide_speaker(s: SessionState, *, sp_id, d, wav: np.ndarray | None,
                   start_ms: int | None, end_ms: int | None,
                   rec_extra: dict, vp_debug: bool) -> str:
    """発話の話者キーを決める統合層（constrain 前まで。フローはモジュール docstring）.

    引数:
      sp_id: 声紋段（VoiceProfiles.classify / STTラベル正規化）の結果キー
      d: tracker.last（声紋判定の診断辞書。tracker 無しなら None）
      wav: 発話区間の音声（クラスタ層の蓄積・照合に使う）
      rec_extra: records に併記する判定根拠（本関数が追記する）
    戻り値: constrain_human_speaker_key 適用前の話者キー。
    """
    if start_ms is None or end_ms is None or end_ms <= start_ms:
        return sp_id   # ステップ0: 区間不明は声紋段の結果のまま
    voiceprint_speaker, voiceprint_confidence = _voiceprint_claim(d, sp_id)
    diarization_events = s.diarization_window(start_ms, end_ms)
    resolved = s.speaker_resolver.resolve(
        utterance=TimeSegment(start_ms, end_ms),
        stt_speaker=str(sp_id),
        diarization_events=diarization_events,
        voiceprint_speaker=voiceprint_speaker,
        voiceprint_confidence=voiceprint_confidence,
    )
    if resolved.source != "stt":
        if resolved.source == "voiceprint":
            # --- 2. 声紋が勝った ---
            sp_id = resolved.speaker
            rec_extra["speaker_source"] = resolved.source
            rec_extra["speaker_confidence"] = round(resolved.confidence, 3)
            rec_extra["speaker_reason"] = resolved.reason
        else:
            # --- 3. diarization が勝った ---
            sp_id = _cluster_attribution(
                s, resolved, d=d, wav=wav, start_ms=start_ms, end_ms=end_ms,
                diarization_events=diarization_events, rec_extra=rec_extra)
        if vp_debug and resolved.source != "voiceprint":
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
        # --- 4. STT に落ちた ---
        rec_extra["stt_raw_speaker"] = resolved.speaker
        sp_id = s.key_for_stt_fallback_speaker(
            resolved.speaker, duration_ms=end_ms - start_ms
        )
        rec_extra["speaker_source"] = "stt_fallback"
        rec_extra["speaker_confidence"] = 0.0
        rec_extra["speaker_reason"] = "diarization_no_confident_overlap_stt_fallback"
    return sp_id
