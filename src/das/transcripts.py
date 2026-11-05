"""議論グラフ・評価で共用する、Utterance形式のJSONL読込み。"""
from __future__ import annotations

import json
from pathlib import Path

from das.types import Utterance


def load_transcript(path: Path) -> list[Utterance]:
    """空行を除き、各行をUtteranceとして検証する。認識器の生ログとは形式が異なる。"""
    with path.open(encoding="utf-8") as file:
        return [Utterance.model_validate(json.loads(line)) for line in file if line.strip()]
