"""パッケージが import でき、設定が読めることを確認する最低限のスモークテスト。"""

from __future__ import annotations


def test_package_importable() -> None:
    import das

    assert das.__version__


def test_settings_loadable() -> None:
    from das.settings import get_settings

    settings = get_settings()
    assert settings.openai_api_key == "test-openai-key"
    assert settings.linking_threshold == 0.6


def test_types_importable() -> None:
    from das.types import AddEdge, AddNode, Intervention, Tick, Utterance

    u = Utterance(turn_id=1, speaker="A", text="hello")
    assert u.text == "hello"
    assert AddNode and AddEdge and Intervention and Tick
