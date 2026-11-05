from __future__ import annotations

import json
import threading
from types import SimpleNamespace

from click.testing import CliRunner

from tests.support.study import runtime


def test_study_cli_has_explicit_conditions(tmp_path, monkeypatch):
    from das.asr.live import _bootstrap, main

    f = tmp_path / "cfg.json"
    f.write_text("{}")
    captured = []
    monkeypatch.setattr(_bootstrap, "run_session", lambda args: captured.append(args))
    result = CliRunner().invoke(main, ["--study-config", str(f), "--study-condition", "none"])
    assert result.exit_code == 0 and captured[0].study_condition == "none"


def test_condition_changing_ui_calls_rejected():
    from das.asr.live._ui import _UIRequestHandler

    h = object.__new__(_UIRequestHandler)
    h._state = SimpleNamespace(study=object())
    h.path = "/api/mode"
    out = []
    h._json = lambda code, data: out.append((code, data))
    h._dispatch_post()
    assert out[0][0] == 409


def test_legacy_workers_not_started_in_study_mode(monkeypatch):
    from das.asr.live import _bootstrap

    threads = []

    class FakeThread:
        def __init__(self, **kw):
            threads.append(kw)

        def start(self):
            pass

    monkeypatch.setattr(_bootstrap.threading, "Thread", FakeThread)
    _bootstrap._start_llm_workers(
        SimpleNamespace(study=SimpleNamespace(condition="none")),
        SimpleNamespace(no_llm=False),
        oai_key="fake",
        oai_model="fake",
        out_path="unused",
        explicit_agenda=False,
    )
    assert len(threads) == 1 and threads[0]["target"].__name__ == "run_study_worker"


def test_participant_http_routes_without_private_material(tmp_path):
    import urllib.request
    from http.server import ThreadingHTTPServer

    from das.asr.live._ui import _UIHandler

    runtime_obj = runtime(tmp_path, "none")
    fake = SimpleNamespace(study=runtime_obj)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _UIHandler.create(fake))
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(base + "/study") as response:
            html = response.read().decode()
        with urllib.request.urlopen(base + "/api/study") as response:
            payload = json.loads(response.read())
        assert "textContent" in html and "innerHTML" not in html
        assert payload["records"] == [] and "scores" not in payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

