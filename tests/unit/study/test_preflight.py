from __future__ import annotations


def test_twenty_minute_virtual_sessions(tmp_path):
    from das.study.preflight import run_preflight

    report = run_preflight(tmp_path / "load")
    assert report["status"] == "passed"
    assert [r["voice_requests"] for r in report["runs"]] == [12, 0]
    assert all(r["human_turns"] == 240 for r in report["runs"])

