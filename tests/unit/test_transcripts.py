import json

import pytest
from pydantic import ValidationError

from das.transcripts import load_transcript


def test_shared_reader_skips_blank_lines_and_validates(tmp_path):
    path = tmp_path / "turns.jsonl"
    path.write_text("\n" + json.dumps({"turn_id": 1, "speaker": "A", "text": "意見です"}) + "\n\n")
    result = load_transcript(path)
    assert len(result) == 1 and result[0].speaker == "A" and result[0].text == "意見です"


@pytest.mark.parametrize(
    "line,error", [("bad json", json.JSONDecodeError), ("{}", ValidationError)]
)
def test_invalid_transcript_still_fails(tmp_path, line, error):
    path = tmp_path / "turns.jsonl"
    path.write_text(line)
    with pytest.raises(error):
        load_transcript(path)
