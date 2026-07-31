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

    name = "otherprov"


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
    key = s.key_for_diarization_speaker("otherprov", "A", duration_ms=10)
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

    from das.asr.live._intervention import _effective_silence
    s = _make_state()
    last = [0.0]                # 実際には大きな沈黙が経過している状態
    now = _t.monotonic()
    s.partial_text = "まだ喋っている途中で"
    s._last_partial_change = now
    assert _effective_silence(s, now, last) == 0.0


def test_effective_silence_normal_when_no_partial():
    """partial が空なら従来どおり now - last_utt_time を返す."""
    import time as _t

    from das.asr.live._intervention import _effective_silence
    s = _make_state()
    last = [0.0]
    now = _t.monotonic()
    s.partial_text = ""
    assert _effective_silence(s, now, last) == now - last[0]


def test_effective_silence_ignores_stale_partial():
    """partial が10秒以上変化していなければ stale として無視し、通常の沈黙を返す."""
    import time as _t

    from das.asr.live._intervention import _effective_silence
    s = _make_state()
    last = [0.0]
    now = _t.monotonic()
    s.partial_text = "クリアされずに固着した partial"
    s._last_partial_change = now - 11.0  # 10秒超前
    assert _effective_silence(s, now, last) == now - last[0]


# ---------------------------------------------------------------------------
# constrain の可視化（2026-07-25: 上限1のまま2人会話→2人目が無警告で全滅の対策）
# ---------------------------------------------------------------------------

def _make_state_with_diag(tmp_path, max_speakers=1):
    from types import SimpleNamespace
    return SessionState(
        args=SimpleNamespace(diarization_max_speakers=max_speakers),
        started=datetime.datetime(2026, 1, 1),
        out_path=str(tmp_path / "o.md"),
        html_path=str(tmp_path / "o.html"),
        diag_path=str(tmp_path / "o.diag"),
        turns_path=str(tmp_path / "o.turns"),
        wav_path=str(tmp_path / "o.wav"),
    )


def test_constrain_drop_writes_diag_and_keeps_behavior(tmp_path):
    """上限で落とした事実が diag に構造化され、返り値（挙動）は従来どおり."""
    import json as _json
    s = _make_state_with_diag(tmp_path, max_speakers=1)
    s.records.append({"ms": 0, "end_ms": 500, "speaker": "@diar:1", "text": "x"})
    s.disp_name("@diar:1")   # スロット1を占有
    assert s.constrain_human_speaker_key("@diar:2") == "?"   # 挙動は不変
    with open(tmp_path / "o.diag") as f:
        events = [_json.loads(line) for line in f]
    assert events and events[0]["type"] == "constrain_drop"
    assert events[0]["key"] == "@diar:2"
    assert events[0]["max_speakers"] == 1
    assert events[0]["slots"] == ["参加者A"]


def test_constrain_drop_warns_once_after_repeats(tmp_path):
    """同一キーが3回落ちたら一度だけ sys 警告が出る（連発はしない）."""
    s = _make_state_with_diag(tmp_path, max_speakers=1)
    s.records.append({"ms": 0, "end_ms": 500, "speaker": "@diar:1", "text": "x"})
    s.disp_name("@diar:1")
    for _ in range(5):
        s.constrain_human_speaker_key("@diar:2")
    sys_msgs = [r["sys"] for r in s.records if "sys" in r]
    assert len(sys_msgs) == 1
    assert "想定話者数の上限" in sys_msgs[0]
    # 警告レコードにはタイムスタンプ（経過ms）が付く
    warn = next(r for r in s.records if "sys" in r)
    assert isinstance(warn["ms"], int)


def test_constrain_drop_state_cleared_on_reset(tmp_path):
    """会議リセットで警告状態と回数がクリアされ、次の会議で再警告できる."""
    s = _make_state_with_diag(tmp_path, max_speakers=1)
    s.records.append({"ms": 0, "end_ms": 500, "speaker": "@diar:1", "text": "x"})
    s.disp_name("@diar:1")
    for _ in range(3):
        s.constrain_human_speaker_key("@diar:2")
    assert s.constrain_warned is True
    s.reset_for_new_meeting()
    assert s.constrain_warned is False
    assert s.constrain_drop_counts == {}


def test_constrain_unified_seat_rule_counterfactual_1723(tmp_path):
    """統一席ルールの実セッション反実仮想（2026-07-25_1723, max=2）.

    観測: 席2つ（@diar:1=A, 人物1=B）が埋まった後、声紋の二重登録で生まれた
    人物2 が旧実装では素通しして「参加者C」を作った。統一ルールでは未確定に
    落ち、表示される人間は設定どおり2人を超えない。
    """
    from types import SimpleNamespace
    s = SessionState(
        args=SimpleNamespace(diarization_max_speakers=2),
        started=datetime.datetime(2026, 7, 25),
        out_path=str(tmp_path / "o.md"), html_path=str(tmp_path / "o.html"),
        diag_path=str(tmp_path / "o.diag"), turns_path=str(tmp_path / "o.turns"),
        wav_path=str(tmp_path / "o.wav"))
    assert s.constrain_human_speaker_key("@diar:1") == "@diar:1"
    s.records.append({"ms": 0, "end_ms": 500, "speaker": "@diar:1", "text": "x"})
    s.disp_name("@diar:1")                      # 参加者A
    assert s.constrain_human_speaker_key("人物1") == "人物1"
    s.records.append({"ms": 600, "end_ms": 900, "speaker": "人物1", "text": "y"})
    s.disp_name("人物1")                        # 参加者B
    # 3人目（声紋の二重登録）は席が無いので未確定 → 参加者Cは生まれない
    assert s.constrain_human_speaker_key("人物2") == "?"
    # 既存の2人はその後も安定して通る
    assert s.constrain_human_speaker_key("@diar:1") == "@diar:1"
    assert s.constrain_human_speaker_key("人物1") == "人物1"


def test_sys_records_do_not_consume_a_seat(tmp_path):
    """システムメッセージが席を1つ食い潰さない（想定話者数の目減り対策）.

    records には話者を持たない add_sys のレコードが混ざる。統一席ルール
    （2026-07-25）が records を走査するようになった際、これらを話者キー "" と
    して数えていたため、空文字が1席として居座り上限が実質1人ぶん減っていた。
    sys は鋳造のたびに必ず出る（「この声を『参加者A』として追跡開始」）ので、
    上限2の2人会話では2人目が常に締め出されていた。
    """
    s = _make_state_with_diag(tmp_path, max_speakers=2)
    s.records.append({"ms": 0, "end_ms": 500, "speaker": "@diar:1", "text": "x"})
    s.disp_name("@diar:1")                      # 参加者A（席1）
    assert len(s._known_human_slots()) == 1
    s.add_sys(100, "この声を「参加者A」として追跡開始")
    assert len(s._known_human_slots()) == 1     # sys は席を作らない
    # 2人目は席が空いているので通る（旧実装では "?" に落ちていた）
    assert s.constrain_human_speaker_key("@diar:2") == "@diar:2"


def test_saved_html_colors_stable_across_rekey(tmp_path):
    """保存用HTMLの色が rekey（統合/リネーム）で他の話者へずれない（監査E）.

    旧実装は list(colors).index() だったため、rekey の pop で後続話者の色が
    全員ずれた（ライブUI側は C11/P2 で修正済み、保存HTML側だけ旧実装が残存）。
    """
    from types import SimpleNamespace
    s = SessionState(
        args=SimpleNamespace(diarization_max_speakers=None),
        started=datetime.datetime(2026, 7, 25),
        out_path=str(tmp_path / "o.md"), html_path=str(tmp_path / "o.html"),
        diag_path=str(tmp_path / "o.diag"), turns_path=str(tmp_path / "o.turns"),
        wav_path=str(tmp_path / "o.wav"))
    s.records = [
        {"speaker": "#1", "text": "a", "ms": 0, "end_ms": 500},
        {"speaker": "#2", "text": "b", "ms": 500, "end_ms": 1000},
    ]
    color2_before = s.html_color("#2")
    s.write_html(live=False)
    s.rekey("#1", "田中")          # 先頭キーの統合（旧実装ならここで #2 の色がずれた）
    s.write_html(live=False)
    assert s.html_color("#2") == color2_before
    with open(tmp_path / "o.html", encoding="utf-8") as f:
        assert color2_before in f.read()


def test_show_partial_writes_text_and_timestamp_under_one_lock(tmp_path):
    """partial のテキストと更新時刻が state_lock 下で組にして書かれる.

    _workers._effective_silence は「非空の partial が直近に更新されていれば
    フロアは埋まっている」と判定する（F3）。書き手がロックを取らないと、読み手が
    state_lock を取っていても「新しいテキスト＋古いタイムスタンプ」の組を読み、
    発話中なのにフロアが空いたと誤認して介入が発話に被さる（M6/F3 が塞いだはずの
    穴が書き手側で開いていた。2026-07-25 監査）。
    """
    import threading

    s = _make_state_with_diag(tmp_path, max_speakers=2)

    s.show_partial("1", "途中経過のテキスト")
    assert s.partial_text == "途中経過のテキスト"
    assert s._last_partial_change > 0.0

    # 別スレッドが state_lock を保持している間は show_partial が進めない
    # （＝書き込みが本当にロック下にある）。修正前はロック外なので即座に完了した。
    s.state_lock.acquire()
    done: list[bool] = []

    def _writer() -> None:
        s.show_partial("1", "次の途中経過")
        done.append(True)

    t = threading.Thread(target=_writer, daemon=True)
    t.start()
    t.join(timeout=0.3)
    assert done == [], "state_lock 保持中に partial が書き換わった（ロック外の書き込み）"
    assert s.partial_text == "途中経過のテキスト"
    s.state_lock.release()
    t.join(timeout=1.0)
    assert done == [True]
    assert s.partial_text == "次の途中経過"


def test_diarization_key_issue_is_serialized_with_rekey(tmp_path):
    """クラスタ台帳の発行が state_lock 下で行われる（rekey の走査との競合防止）.

    台帳 diarization_speaker_keys は recvスレッドが書き、rekey（UIの /rename・
    /activate 経由＝別スレッド）が state_lock 下で `.items()` を走査して書き換える。
    発行側がロックを取らないと、走査中の挿入で「dictionary changed size during
    iteration」が起きて rekey が落ちる（2026-07-25 監査）。
    """
    import threading

    s = _make_state_with_diag(tmp_path, max_speakers=None)
    assert s.key_for_diarization_speaker("pyannote", "A") == "@diar:1"

    s.state_lock.acquire()
    issued: list[str] = []

    def _issuer() -> None:
        issued.append(s.key_for_diarization_speaker("pyannote", "B"))

    t = threading.Thread(target=_issuer, daemon=True)
    t.start()
    t.join(timeout=0.3)
    assert issued == [], "state_lock 保持中に新しいクラスタキーが発行された"
    s.state_lock.release()
    t.join(timeout=1.0)
    assert issued == ["@diar:2"]


# ---------------------------------------------------------------------------
# 表示ラベルの詰め直し（handoff §28.6）
# ---------------------------------------------------------------------------

def _label_state(tmp_path):
    import datetime

    from das.asr.live._session_state import SessionState

    class _Args:
        lang = "ja"
        vp_debug = False
        diarization = "pyannote"
        diarization_max_speakers = 3
        vp_cluster_naming = True
        stt = "soniox"

    st = SessionState(
        args=_Args(), started=datetime.datetime(2026, 7, 26),
        out_path=str(tmp_path / "o.md"), html_path=str(tmp_path / "o.html"),
        diag_path=str(tmp_path / "o.diag"), turns_path=str(tmp_path / "o.turns"),
        wav_path=str(tmp_path / "o.wav"), serve=False)
    st.save = lambda *a, **k: None
    return st


def test_labels_start_from_a_after_utterances_move_away(tmp_path):
    """発言の無くなったキーが押さえていた文字を解放する.

    席の割当てや遡及訂正で発話が別のキーへ移ると、元のキーは1件も発話を
    持たないまま文字だけ押さえ続ける。その結果、参加者が1人しか居ないのに
    「参加者B」から始まる（2026-07-26 に実会話で確認）。
    """
    st = _label_state(tmp_path)
    # #1 が先に参加者A を取り、その後で発話が 人物2 へ移った状況
    assert st.disp_name("#1") == "参加者A"
    assert st.disp_name("人物2") == "参加者B"
    st.records = [{"ms": 1000, "speaker": "人物2", "text": "こんにちは"}]

    assert st.compact_anonymous_labels() > 0
    assert st.disp_name("人物2") == "参加者A"     # 1人なのでAから始まる
    assert "#1" not in st.anonymous_labels        # 幻のキーは台帳から消える


def test_compaction_keeps_the_order_people_appeared_in(tmp_path):
    """初出順は保つ（既に見えている人同士の文字が入れ替わらない）."""
    st = _label_state(tmp_path)
    for k in ("#1", "人物1", "人物2", "人物3"):
        st.disp_name(k)                            # A/B/C/D を確保
    st.records = [{"ms": 1000, "speaker": "人物2", "text": "先に喋った"},
                  {"ms": 2000, "speaker": "人物1", "text": "後から喋った"}]

    st.compact_anonymous_labels()

    assert st.disp_name("人物2") == "参加者A"      # 初出が早い方がA
    assert st.disp_name("人物1") == "参加者B"
    assert st.anonymous_labels.keys() == {"人物1", "人物2"}


def test_compaction_is_idempotent(tmp_path):
    """既に詰まっていれば何もしない（毎回の呼び出しで表示が揺れない）."""
    st = _label_state(tmp_path)
    st.records = [{"ms": 1000, "speaker": "人物1", "text": "あ"},
                  {"ms": 2000, "speaker": "人物2", "text": "い"}]
    st.disp_name("人物1")
    st.disp_name("人物2")
    st.compact_anonymous_labels()
    before = dict(st.anonymous_labels)

    assert st.compact_anonymous_labels() == 0
    assert st.anonymous_labels == before


def test_previously_registered_voice_still_needs_a_free_seat(tmp_path):
    """過去の会議で登録された人物も、席が無ければ入れない（handoff §28.9）.

    voices.json は**セッションをまたいで**プロファイルを保持する。以前は
    「実名リネーム済みの人物は既存の席保持者」として実名キーを無条件に
    通していたため、今回まだ一度も喋っていない過去の登録者が上限を素通り
    した。実会話では上限3に対し8人が表示され、そこには名前の聞き間違いで
    できた「壁」「朱色」も含まれていた。
    """
    s = _make_state_with_diag(tmp_path, max_speakers=2)
    s.records.append({"ms": 0, "end_ms": 500, "speaker": "@diar:1", "text": "x"})
    s.disp_name("@diar:1")
    s.records.append({"ms": 600, "end_ms": 900, "speaker": "松井", "text": "y"})

    # ここまでで2席（参加者A と 松井）。過去の登録者は入れない
    assert s.constrain_human_speaker_key("壁") == "?"
    # 既に喋っている実名は当然そのまま通る
    assert s.constrain_human_speaker_key("松井") == "松井"


def test_renamed_person_keeps_their_seat(tmp_path):
    """会議中にリネームした人は席を失わない（rekey はラベルを解放するため）."""
    s = _make_state_with_diag(tmp_path, max_speakers=1)
    s.records.append({"ms": 0, "end_ms": 500, "speaker": "人物1", "text": "x"})
    s.disp_name("人物1")
    s.rekey("人物1", "田中")

    assert s.constrain_human_speaker_key("田中") == "田中"
    assert s.constrain_human_speaker_key("人物2") == "?"   # 上限1なので他は入れない
