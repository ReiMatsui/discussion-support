"""試行GUIの録画保存・起動条件。実API・実カメラ・実マイクは使わない。"""
from __future__ import annotations

import json
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import Mock

import pytest

from das.pilot.recording import VideoRecording
from das.pilot.server import make_server
from das.pilot.session import PilotSession

ROOT = Path(__file__).resolve().parents[3]


def session(tmp_path):
    return PilotSession(ROOT, tmp_path / "pilot")


def start(pilot, **changes):
    return pilot.start(**dict(scenario="scripted", condition="record_only", mime="video/webm", consent=True, **changes))


def test_video_order_completion_and_no_overwrite(tmp_path):
    video = VideoRecording(tmp_path, "video/webm")
    video.append(0, b"first")
    with pytest.raises(ValueError, match="順序"):
        video.append(0, b"duplicate")
    with pytest.raises(ValueError, match="一致"):
        video.finish(2)
    video.append(1, b"last")
    video.finish(2)
    video.close_incomplete()
    assert video.path.read_bytes() == b"firstlast"
    assert json.loads(video.manifest_path.read_text())["complete"] is True
    with pytest.raises(ValueError, match="終了"):
        video.append(2, b"later")
    with pytest.raises(FileExistsError):
        VideoRecording(tmp_path, "video/webm")


def test_incomplete_recording_and_size_limit(tmp_path, monkeypatch):
    from das.pilot import recording

    monkeypatch.setattr(recording, "MAX_VIDEO_BYTES", 3)
    video = VideoRecording(tmp_path, "video/mp4")
    video.append(0, b"abc")
    with pytest.raises(ValueError, match="上限"):
        video.append(1, b"d")
    video.close_incomplete()
    assert json.loads(video.manifest_path.read_text())["complete"] is False
    assert video.path.read_bytes() == b"abc"


def test_gate_does_not_start_external_process(tmp_path, monkeypatch):
    pilot = session(tmp_path)
    process = Mock(side_effect=AssertionError("must not launch"))
    monkeypatch.setattr(subprocess, "Popen", process)
    with pytest.raises(ValueError, match="了承"):
        pilot.start(scenario="scripted", condition="record_only", mime="video/webm", consent=False)
    with pytest.raises(ValueError, match="接続確認"):
        pilot.start(scenario="free", condition="system", mime="video/webm", consent=True)
    info = start(pilot)
    with pytest.raises(ValueError, match="一致"):
        pilot.append_video("wrong", 0, b"one")
    pilot.append_video(info["id"], 0, b"one")
    pilot.finish_video(info["id"], 1)
    pilot.note("decision", "C社を選択")
    stopped = pilot.stop()
    assert stopped["video_complete"] and pilot.view()["active"] is None
    assert pilot.completed_video(info["id"])[0].read_bytes() == b"one"
    assert not process.called
    assert "C社を選択" in (pilot.output / "pilot_events.jsonl").read_text()


def test_reload_or_stop_preserves_incomplete_status(tmp_path):
    pilot = session(tmp_path)
    info = start(pilot)
    pilot.append_video(info["id"], 0, b"partial")
    assert not pilot.stop("browser_lost")["video_complete"]
    with pytest.raises(ValueError):
        pilot.completed_video(info["id"])
    assert pilot.stop() == {"ok": True}


def test_launch_uses_current_study_mode_and_bounded_config(tmp_path, monkeypatch):
    pilot = session(tmp_path)
    pilot.model_check = {"status": "completed"}
    child = Mock()
    child.poll.return_value = 0
    child.returncode = 0
    launch = Mock(return_value=child)
    monkeypatch.setattr(subprocess, "Popen", launch)
    info = pilot.start(scenario="free", condition="system", mime="video/webm", consent=True)
    command = launch.call_args.args[0]
    assert command[1:3] == ["-m", "das.asr.live"]
    assert command[command.index("--study-condition") + 1] == "system"
    assert "--no-setup" in command and "--simulate" not in command
    config = json.loads((Path(info["directory"]) / "task.json").read_text())
    assert config["review_call_limit"] == 60 and config["options"] == ["A社", "B社", "C社"]
    assert info["max_seconds"] == 480
    pilot.stop()


def test_connection_check_reuses_probe_and_sanitizes_failure(tmp_path, monkeypatch):
    pilot = session(tmp_path)
    def run(command, **kwargs):
        assert command[1:4] == ["-m", "das.study.bounded_probe", "--execute"]
        Path(command[-1]).write_text(json.dumps({"status":"failed", "http_status":401,
                                               "message":"APIキーを確認してください"}))
        return Mock(returncode=2)
    monkeypatch.setattr(subprocess, "run", run)
    result = pilot.check_model()
    assert result["status"] == "failed" and result["http_status"] == 401
    assert not pilot.checking and pilot.active is None


def test_timeout_check_clears_busy_state(tmp_path, monkeypatch):
    pilot = session(tmp_path)
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("mock", 45)
    monkeypatch.setattr(subprocess, "run", timeout)
    assert pilot.check_model()["status"] == "failed"
    assert not pilot.checking


def test_http_record_save_requires_token(tmp_path):
    pilot = session(tmp_path)
    server = make_server(pilot, 0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval":0.01}, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    def post(path, body, token=None):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        req = urllib.request.Request(base + path, data=data, headers={"X-Pilot-Token":token or "invalid"})
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read())
    try:
        with urllib.request.urlopen(base) as response:
            assert pilot.token in response.read().decode()
        with pytest.raises(urllib.error.HTTPError) as exc:
            post("/api/start", {})
        assert exc.value.code == 403
        info = post("/api/start", {"scenario":"scripted", "condition":"record_only",
                                   "mime":"video/webm", "consent":True}, pilot.token)
        post(f"/api/video/{info['id']}/0", b"movie", pilot.token)
        post("/api/video-finish", {"id":info["id"], "chunks":1}, pilot.token)
        assert post("/api/stop", {}, pilot.token)["video_complete"]
        with urllib.request.urlopen(base + "/video/" + info["id"]) as response:
            assert response.read() == b"movie"
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/video/../../.env")
    finally:
        pilot.stop()
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_pilot_command_only_opens_launcher(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from das.cli import app
    from das.pilot import server

    serve = Mock()
    monkeypatch.setattr(server, "serve", serve)
    result = CliRunner().invoke(app, ["pilot", "--output", str(tmp_path), "--no-open"])
    assert result.exit_code == 0, result.output
    assert serve.call_args.kwargs["open_browser"] is False
    assert serve.call_args.args[0].active is None
