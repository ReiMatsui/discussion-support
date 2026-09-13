"""介入の指示文（agents/_notes.py）の文面を固定する."""
from __future__ import annotations

from das.asr.live.agents import _notes

TOPICS = [
    {"topic": "AIツール導入の是非", "speaker": "議題"},
    {"topic": "公園でのピクニックに行く場所の話", "speaker": "参加者A"},
]
PENDING = [{"speaker": "参加者A", "text": "おにぎりがいいですね"}]


def test_drift_directive_names_the_agenda_and_steers_back():
    """脱線の指示は議題を名指しして「戻す側に立つ」ことを求める.

    2026-09-13 のシミュレーションで、旧文面（「今の流れを踏まえて会話を前に
    進める」）のままだとピクニックの話を進行してしまった。
    """
    conv = _notes.compose_trigger_notes(
        _notes.format_utterance_context(PENDING), topics=TOPICS, drift_reason="弁当話に脱線")
    directive, context = _notes.split_directive_and_context(conv)
    assert "会議の議題は「AIツール導入の是非」です" in directive
    assert "議題に戻す短い一言" in directive
    assert "議題へ戻すことを優先" in directive          # 論点の注記も脱線時は戻す側
    assert "新しい論点は尊重" not in directive
    assert context.startswith("[参加者発話]") and "おにぎり" in context


def test_drift_directive_without_agenda_still_steers_back():
    conv = _notes.compose_trigger_notes("", topics=None, drift_reason="雑談")
    assert "会議の議題から離れています" in conv and "戻す側" in conv


def test_topics_note_respects_new_topics_when_not_drifting():
    conv = _notes.compose_trigger_notes("", topics=TOPICS, invite_target="参加者B")
    assert "新しい論点は尊重" in conv
    assert "[声かけ] 参加者Bさん" in conv
    assert "脱線" not in conv


def test_agenda_of_prefers_seeded_agenda():
    assert _notes.agenda_of(TOPICS) == "AIツール導入の是非"
    assert _notes.agenda_of([{"topic": "x", "speaker": "参加者A"}]) == ""
    assert _notes.agenda_of(None) == ""
