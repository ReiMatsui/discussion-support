"""SessionState の脱線検出シード（Fix 8）のユニットテスト."""
from __future__ import annotations

import datetime
from types import SimpleNamespace

from das.asr.live._session_state import SessionState


def _make_state() -> SessionState:
    return SessionState(
        args=object(),
        started=datetime.datetime(2026, 1, 1),
        out_path="/tmp/o.md",
        html_path="/tmp/o.html",
        diag_path="/tmp/o.diag",
        turns_path="/tmp/o.turns",
        wav_path="/tmp/o.wav",
    )


def test_seed_topic_adds_when_empty():
    s = _make_state()
    s.seed_topic("AIツール導入の是非")
    assert [t["topic"] for t in s.topics] == ["AIツール導入の是非"]
    assert s.topics[0]["speaker"] == "議題"


def test_retired_echo_texts_respect_ttl():
    """退役エコーテキストは TTL(10s) 内だけ照合対象に含まれる（P2-4）."""
    import time
    s = _make_state()
    s.add_retired_echo_texts(["まず、今日の目的を確認しましょう", ""])
    now = time.monotonic()
    # 空文字は除外され、1件だけ保持
    assert s.recent_retired_echo_texts(now=now) == ["まず、今日の目的を確認しましょう"]
    # TTL を過ぎたら対象外
    assert s.recent_retired_echo_texts(now=now + 11.0) == []


def test_anonymous_label_reused_after_merge_no_skip():
    """幻の話者が統合で消えたら、その文字を次の新規話者が再利用する（飛びの解消）."""
    s = _make_state()
    assert s.disp_name("#1") == "参加者A"
    assert s.disp_name("#2") == "参加者B"   # ファシリテーターの声などの幻
    assert s.disp_name("#3") == "参加者C"
    s.rekey("#2", "#1")                      # 幻 #2 を #1 に統合 → B が解放される
    assert s.disp_name("#4") == "参加者B"    # 参加者D に飛ばず B を再利用


def test_anonymous_labels_no_collision_after_merge():
    """統合後、既存キーの表示と新規キーの表示が衝突しない（重複の回帰）."""
    s = _make_state()
    for k in ("#1", "#2", "#3"):
        s.disp_name(k)
    s.rekey("#2", "#1")
    assert {s.disp_name("#3"), s.disp_name("#4")} == {"参加者C", "参加者B"}


def test_real_name_releases_letter():
    """実名を付けたキーの文字は解放され、新規キーが再利用する（表示は実名のまま）."""
    s = _make_state()
    assert s.disp_name("#1") == "参加者A"
    assert s.disp_name("#2") == "参加者B"
    s.set_display_name("#1", "松井")
    assert s.disp_name("#1") == "松井"       # 表示は実名のまま
    assert s.disp_name("#3") == "参加者A"    # 解放された A を再利用


def test_system_anonymous_name_does_not_release_letter():
    """話者N/人物N などのシステム匿名名では文字を解放しない（実名のみ解放）."""
    s = _make_state()
    assert s.disp_name("#1") == "参加者A"
    s.set_display_name("#1", "人物3")        # システム匿名名
    assert "#1" in s.anonymous_labels        # 解放されない
    assert s.disp_name("#2") == "参加者B"    # A は使用中のまま


def test_merge_into_anonymous_keeps_carryover():
    """統合先に文字が無い匿名キーへの統合は、従来どおり文字を引き継ぐ（setdefault経路）."""
    s = _make_state()
    assert s.disp_name("#1") == "参加者A"
    s.rekey("#1", "#9")                      # #9 は未表示（ラベル無し）
    assert s.disp_name("#9") == "参加者A"    # #1 の A を引き継ぐ


def test_constrain_after_label_release_with_max_speakers():
    """max_speakers 指定下でも、統合で文字が解放された後の判定が正しい."""
    s = _make_state()
    s.args = SimpleNamespace(diarization_max_speakers=2)
    assert s.disp_name("#1") == "参加者A"
    assert s.disp_name("#2") == "参加者B"    # 幻
    s.rekey("#2", "#1")                      # 幻を統合 → B 解放
    assert s.constrain_human_speaker_key("#3") == "#3"   # 上限2以内で通る
    assert s.disp_name("#3") == "参加者B"    # 解放された B を再利用


def test_peek_disp_name_is_read_only():
    """peek_disp_name は割当てを行わない: 未割当てキーは「?」、辞書は不変."""
    s = _make_state()
    assert s.peek_disp_name("#1") == "?"        # 未割当て → 中立表示
    assert s.anonymous_labels == {}             # 副作用なし
    assert s.disp_name("#1") == "参加者A"       # 本表示（flush経路）は従来どおり割当て
    assert s.peek_disp_name("#1") == "参加者A"  # 既割当てはその文字を返す


def test_show_partial_phantom_key_does_not_consume_label_slot():
    """partial 表示だけのSTTラベル（幻キー）が文字とスロットを消費しない.

    バグ修正（2026-07-14 実セッション）: show_partial が disp_name（新規割当て）を
    呼んでいたため、議事録に一度も現れない partial 限りの幻キーが「参加者B」の
    文字と max_speakers スロットを恒久的に占有した
    (docs/design/handoff_2026-07-14_unregistered_speakers.md 参照)。
    """
    s = _make_state()
    s.args = SimpleNamespace(diarization_max_speakers=2)
    s.show_partial("7", "話している途中のテキスト")   # 幻ラベル: flush に到達しない
    assert s.anonymous_labels == {}                    # 文字を消費しない
    assert s.partial_speaker == "?"                    # 中立表示（許容仕様）
    # 幻キーがスロットを食わないので、後続の実参加者2人は従来どおり通る
    assert s.constrain_human_speaker_key("#1") == "#1"
    assert s.disp_name("#1") == "参加者A"
    assert s.constrain_human_speaker_key("#2") == "#2"
    assert s.disp_name("#2") == "参加者B"


def test_rekey_migrates_display_name_and_cleans_old_entry():
    """rekey は names も colors と同じ流儀で移行し、old 側の残留を掃除する（F7）."""
    s = _make_state()
    s.names["#1"] = "田中"
    s.rekey("#1", "#2")
    assert s.names == {"#2": "田中"}         # 移行される（new 側に名前が無い場合）
    s.names["#3"] = "鈴木"
    s.rekey("#3", "#2")
    assert s.names == {"#2": "田中"}         # new 側に名前があれば old 側は捨てる


def test_diarization_key_seq_never_reissues_key_after_pop():
    """名寄せで keys が pop されても、使用中の @diar:N を別人へ再発行しない（F1）."""
    s = _make_state()
    assert s.key_for_diarization_speaker("pyannote", "A") == "@diar:1"
    assert s.key_for_diarization_speaker("pyannote", "B") == "@diar:2"
    assert s.key_for_diarization_speaker("pyannote", "C") == "@diar:3"
    # クラスタ間名寄せ相当: B が A に吸収されエントリが pop される
    s.diarization_speaker_keys.pop("pyannote:B")
    # len ベース採番なら使用中の @diar:3 を再発行していた（キー衝突の回帰）
    assert s.key_for_diarization_speaker("pyannote", "D") == "@diar:4"


class _FakePyannoteProvider:
    """pyannote provider をhysteresis判定用に模したダミー(.name==\"pyannote\")."""

    name = "pyannote"


class _FakeOtherProvider:
    """pyannote以外のprovider（AssemblyAI等）を模したダミー."""

    name = "assemblyai"


def test_key_for_diarization_speaker_hysteresis_below_threshold_stays_unsure():
    """pyannote使用時、累積発話が3秒未満の新規ラベルは@diar:Nを発行せずUNSURE_SPEAKERのまま."""
    s = _make_state()
    s.diarization_provider = _FakePyannoteProvider()
    assert s.key_for_diarization_speaker("pyannote", "SPEAKER_00", duration_ms=1000) == "?"
    assert s.key_for_diarization_speaker("pyannote", "SPEAKER_00", duration_ms=1500) == "?"
    # 累計2.5秒 < 3.0秒 なのでまだ@diar:Nは発行されない
    assert "pyannote:SPEAKER_00" not in s.diarization_speaker_keys


def test_key_for_diarization_speaker_hysteresis_above_threshold_registers_participant():
    """累積発話が3秒に達したら@diar:Nを新規発行し、以後は安定して同じキーを返す."""
    s = _make_state()
    s.diarization_provider = _FakePyannoteProvider()
    assert s.key_for_diarization_speaker("pyannote", "SPEAKER_00", duration_ms=1500) == "?"
    key = s.key_for_diarization_speaker("pyannote", "SPEAKER_00", duration_ms=1600)
    assert key.startswith("@diar:")
    # 一度確定したら、以後は同じ生ラベルに対して同じキーを安定して返す
    assert s.key_for_diarization_speaker("pyannote", "SPEAKER_00", duration_ms=50) == key


def test_key_for_diarization_speaker_no_hysteresis_for_non_pyannote_provider():
    """pyannote以外のproviderでは従来どおり即時に@diar:Nを発行する（挙動を変えない）."""
    s = _make_state()
    s.diarization_provider = _FakeOtherProvider()
    key = s.key_for_diarization_speaker("assemblyai", "A", duration_ms=10)
    assert key.startswith("@diar:")


def test_key_for_diarization_speaker_no_hysteresis_without_provider():
    """diarization_provider未設定（従来のデフォルト）でも即時発行する."""
    s = _make_state()
    key = s.key_for_diarization_speaker("stt", "A", duration_ms=10)
    assert key.startswith("@diar:")


def test_reset_drains_summarize_requests():
    """会議リセットで整理介入の要求キューも drain される（C3）."""
    s = _make_state()
    s.summarize_requests.put({"focus": "論点の整理"})
    s.reset_for_new_meeting()
    assert s.summarize_requests.empty()


def test_delivery_event_includes_timing(tmp_path):
    """delivery イベントに speak_start_latency_ms などの timing を残せる（Phase4観測）."""
    import json
    s = _make_state()
    s.interventions_path = str(tmp_path / "o.interventions.jsonl")
    s.add_facilitator_delivery_event("本題に戻しましょう",
                                     timing={"speak_start_latency_ms": 420.0})
    with open(s.interventions_path, encoding="utf-8") as f:
        lines = [json.loads(x) for x in f]
    assert lines[-1]["type"] == "delivery"
    assert lines[-1]["timing"]["speak_start_latency_ms"] == 420.0


def test_delivery_event_omits_timing_when_absent(tmp_path):
    """timing 未指定なら delivery イベントに timing キーを付けない（既存互換）."""
    import json
    s = _make_state()
    s.interventions_path = str(tmp_path / "o.interventions.jsonl")
    s.add_facilitator_delivery_event("本題に戻しましょう")
    with open(s.interventions_path, encoding="utf-8") as f:
        lines = [json.loads(x) for x in f]
    assert "timing" not in lines[-1]


def test_seed_topic_noop_for_empty_input():
    s = _make_state()
    s.seed_topic("")
    s.seed_topic(None)
    assert s.topics == []


def test_seed_topic_does_not_override_existing():
    s = _make_state()
    s.topics.append({"topic": "既存論点", "speaker": "話者1"})
    s.seed_topic("議題テーマ")
    assert [t["topic"] for t in s.topics] == ["既存論点"]


# --- M6: partial 受信で沈黙タイマーを更新 ---------------------------------

def test_show_partial_updates_silence_timer_on_new_text():
    """非空の新しい partial を受けたら沈黙タイマーを更新する（発話中を沈黙と誤認しない）."""
    s = _make_state()
    s._last_utt_time[0] = 0.0
    s.show_partial("#1", "この論点はもう少し")
    assert s._last_utt_time[0] > 0.0


def test_show_partial_does_not_update_on_repeated_text():
    """同一 partial の再送では更新しない（沈黙が永久に 0 に張り付くのを防ぐ）."""
    s = _make_state()
    s.show_partial("#1", "同じ文字列")
    s._last_utt_time[0] = 0.0            # 変化検出のため一旦戻す
    s.show_partial("#1", "同じ文字列")   # 同一 partial の再送
    assert s._last_utt_time[0] == 0.0


def test_show_partial_empty_does_not_update():
    """空（strip 後空）の partial では更新しない."""
    s = _make_state()
    s._last_utt_time[0] = 0.0
    s.show_partial("#1", "   ")
    assert s._last_utt_time[0] == 0.0


def test_show_partial_records_change_time():
    """F3: partial が変化したら変化時刻(_last_partial_change)を記録する."""
    s = _make_state()
    s._last_partial_change = 0.0
    s.show_partial("#1", "喋っている途中")
    assert s._last_partial_change > 0.0


# --- F3: アクティブな partial をフロア占有として扱う ----------------------

def test_effective_silence_is_zero_while_active_partial():
    """partial 非空かつ直近更新中は「フロア占有」= 沈黙 0 を返す."""
    import time as _t

    from das.asr.live._workers import _effective_silence
    s = _make_state()
    last = [0.0]                # 実際には大きな沈黙が経過している状態
    now = _t.monotonic()
    s.partial_text = "まだ喋っている途中で"
    s._last_partial_change = now
    assert _effective_silence(s, now, last) == 0.0


def test_effective_silence_normal_when_no_partial():
    """partial が空なら従来どおり now - last_utt_time を返す."""
    import time as _t

    from das.asr.live._workers import _effective_silence
    s = _make_state()
    last = [0.0]
    now = _t.monotonic()
    s.partial_text = ""
    assert _effective_silence(s, now, last) == now - last[0]


def test_effective_silence_ignores_stale_partial():
    """partial が10秒以上変化していなければ stale として無視し、通常の沈黙を返す."""
    import time as _t

    from das.asr.live._workers import _effective_silence
    s = _make_state()
    last = [0.0]
    now = _t.monotonic()
    s.partial_text = "クリアされずに固着した partial"
    s._last_partial_change = now - 11.0  # 10秒超前
    assert _effective_silence(s, now, last) == now - last[0]
