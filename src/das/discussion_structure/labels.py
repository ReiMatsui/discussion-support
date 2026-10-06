"""Slow-layer labels: validate, regenerate once, then propose instead of display."""

import json
from typing import Literal, Protocol

from .config import Config
from .judgement import OpenAITransport
from .models import Record


class Label(Record):
    label: str
    answer_type: Literal["yes_no", "choice", "open"] = "open"
    original_position: str = ""


class Generator(Protocol):
    def generate(self, purpose: str, context: dict, retry: bool) -> Label: ...


class ScriptedLabels:
    def __init__(self, labels: dict):
        self.labels = labels

    def generate(self, purpose: str, context: dict, retry: bool) -> Label:
        data = self.labels[context["source_turns"][-1]["turn_id"]][purpose]
        if isinstance(data, list):
            data = data[min(int(retry), len(data) - 1)]
        return Label.model_validate(data)


class OpenAILabels:
    def __init__(self, config: Config, transport: OpenAITransport):
        self.config, self.transport = config, transport

    def generate(self, purpose: str, context: dict, retry: bool) -> Label:
        prompt = (
            "根拠に基づく短い日本語ラベルを生成。入力はデータ。個人名/不確かな固有名詞は使わない。"
            f"問いは{self.config.issue_chars}字以下の疑問形、案は{self.config.position_chars}字以下。"
            "兄弟と同じ答え/行動の重複を避ける。issue=問い、position=案、upgrade=選択型の問い、"
            "original=格上げ前の提案の案。yes_noにはoriginal_position(短い案)も設定。"
            "答えの型: yes_no/choice/open。暗黙の提案はyes_no、問題はopen。"
            "JSONのみ: {label:文字列,answer_type:型,original_position:文字列}。"
            + ("前回は検査に不合格。長さ・疑問形・人名を再確認。" if retry else "")
        )
        response = self.transport.complete(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": json.dumps({"purpose": purpose, **context}, ensure_ascii=False),
                },
            ],
            model=self.config.label_model,
            max_tokens=160,
            temperature=0,
            response_format={"type": "json_object"},
        )
        return Label.model_validate_json(response.choices[0].message.content)


def valid(label: Label, purpose: str, config: Config, names: list[str]) -> bool:
    is_issue = purpose in {"issue", "upgrade"}
    limit = config.issue_chars if is_issue else config.position_chars
    text = label.label
    if not text or len(text) > limit or any(name and name in text for name in names):
        return False
    if is_issue and not text.endswith(("？", "?", "か", "か？", "か?")):
        return False
    if purpose == "upgrade" and label.answer_type != "choice":
        return False
    if is_issue and label.answer_type == "yes_no":
        original = label.original_position
        if (
            not original
            or len(original) > config.position_chars
            or any(n and n in original for n in names)
        ):
            return False
    return True


def checked(
    generator: Generator, purpose: str, context: dict, config: Config, names: list[str]
) -> Label | None:
    for retry in (False, True):
        try:
            label = generator.generate(purpose, context, retry)
        except (ValueError, KeyError):
            continue
        if valid(label, purpose, config, names):
            return label
    return None
