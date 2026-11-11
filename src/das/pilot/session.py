"""試行ごとの音声本体プロセスとローカル記録を管理する。"""
from __future__ import annotations

import datetime
import json
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from .recording import VideoRecording

PUBLIC_TASK = {
    "title": "3人で参加する仕事体験先を1社選ぶ",
    "options": [
        {"name": "A社", "facts": "3日間・対面／片道60分・分析実務／指導は初日のみ・謝礼合計24,000円"},
        {"name": "B社", "facts": "5日間・オンライン・開発実践／毎日指導あり・謝礼合計5,000円"},
        {"name": "C社", "facts": "4日間・対面／片道20分・見学と小課題／指導は2回・謝礼合計12,000円"},
    ],
}


def utc_now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class PilotSession:
    def __init__(self, root: Path, output: Path):
        self.root = root.resolve()
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.active = None
        self.rounds = []
        self.model_check = {"status": "not_checked", "message": "APIキーを更新し、接続確認を先に行ってください。"}
        self.checking = False

    def event(self, event_type, **fields):
        with (self.output / "pilot_events.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps({"at": utc_now(), "type": event_type, **fields}, ensure_ascii=False) + "\n")

    def check_model(self):
        with self.lock:
            if self.checking or self.active:
                raise ValueError("試行中または確認中には接続確認できません")
            self.checking = True
        result_file = self.output / f"model_check_{time.time_ns()}.json"
        try:
            command = [sys.executable, "-m", "das.study.bounded_probe", "--execute", "--out", str(result_file)]
            subprocess.run(command, cwd=self.root, capture_output=True, timeout=45, check=False)
            report = json.loads(result_file.read_text()) if result_file.exists() else {}
            code = report.get("http_status")
            ok = report.get("status") == "completed"
            self.model_check = {
                "status": "completed" if ok else "failed", "http_status": code,
                "message": "接続確認が通りました。動画・音声はまだ確認していません。" if ok else
                report.get("message", "接続確認が失敗しました。API設定または利用上限を確認してください。"),
                "result_file": str(result_file),
            }
            self.event("model_check", **self.model_check)
            return self.model_check
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            self.model_check = {"status": "failed", "message": "接続確認を完了できませんでした。設定と結果ファイルを確認してください。",
                                "error_type": type(exc).__name__}
            self.event("model_check", **self.model_check)
            return self.model_check
        finally:
            self.checking = False

    def start(self, *, scenario, condition, mime, consent):
        if scenario not in {"scripted", "free"} or condition not in {"system", "none", "record_only"}:
            raise ValueError("試行の設定が不正です")
        if consent is not True:
            raise ValueError("記録する全員の記録・外部送信への了承を確認してください")
        with self.lock:
            if self.active or self.checking:
                raise ValueError("先の試行を終了してください")
            if condition == "system" and self.model_check.get("status") != "completed":
                raise ValueError("APIキー更新後の接続確認を先に通してください")
            if shutil.disk_usage(self.output).free < 1024 ** 3:
                raise ValueError("保存先に1GB以上の空き容量が必要です")
            round_id = datetime.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
            directory = self.output / round_id
            directory.mkdir()
            video = VideoRecording(directory, mime)
            info = {
                "id": round_id, "scenario": scenario, "condition": condition,
                "started_at": utc_now(), "max_seconds": 480 if scenario == "free" else 300,
                "directory": str(directory), "consent_confirmed": True,
                "native_port": None, "status": "recording",
            }
            process = None
            log_file = None
            try:
                if condition != "record_only":
                    config = json.loads((self.root / "study/pilot_jobs.json").read_text())
                    config["session_id"] = round_id
                    config_path = directory / "task.json"
                    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2))
                    info["native_port"] = free_port()
                    command = [
                        sys.executable, "-m", "das.asr.live", "--study-config", str(config_path),
                        "--study-condition", condition, "--out", str(directory / "conversation.md"),
                        "--port", str(info["native_port"]), "--no-open", "--no-setup",
                        "--diarization-max-speakers", "3",
                    ]
                    log_file = (directory / "console.log").open("wb")
                    process = subprocess.Popen(command, cwd=self.root, stdout=log_file, stderr=subprocess.STDOUT)
                self.active = {"info": info, "video": video, "process": process,
                               "log_file": log_file, "monotonic_start": time.monotonic()}
                self.rounds.append(info)
                self._save_round(info)
                self.event("round_started", **info)
            except Exception:
                if process and process.poll() is None:
                    self._stop_process(process)
                self.active = None
                video.close_incomplete()
                if log_file:
                    log_file.close()
                raise
            return info.copy()

    @staticmethod
    def _save_round(info):
        (Path(info["directory"]) / "round.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")

    def append_video(self, round_id, sequence, data):
        with self.lock:
            if not self.active or self.active["info"]["id"] != round_id:
                raise ValueError("録画対象が一致しません")
            self.active["video"].append(sequence, data)

    def finish_video(self, round_id, chunks):
        with self.lock:
            if not self.active or self.active["info"]["id"] != round_id:
                raise ValueError("録画対象が一致しません")
            self.active["video"].finish(chunks)
            self.event("video_completed", round_id=round_id, chunks=chunks)

    def note(self, kind, text=""):
        with self.lock:
            if kind not in {"intervention", "problem", "decision", "note"} or len(text) > 1000:
                raise ValueError("観察メモが不正です")
            self.event("observation", round_id=self.active["info"]["id"] if self.active else None,
                       kind=kind, text=text)

    @staticmethod
    def _stop_process(process):
        for action, timeout in ((lambda: process.send_signal(signal.SIGINT), 8),
                                (process.terminate, 3), (process.kill, 3)):
            if process.poll() is not None:
                return
            try:
                action()
                process.wait(timeout=timeout)
                return
            except ProcessLookupError:
                return
            except subprocess.TimeoutExpired:
                continue

    def stop(self, reason="operator"):
        with self.lock:
            active = self.active
            if not active:
                return {"ok": True}
            process = active["process"]
            info = active["info"]
            if process and process.poll() is None:
                try:
                    request = urllib.request.Request(f"http://127.0.0.1:{info['native_port']}/api/stop",
                                                     data=b"{}", headers={"Content-Type": "application/json"})
                    urllib.request.urlopen(request, timeout=2).close()
                except Exception:
                    pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._stop_process(process)
            active["video"].close_incomplete()
            if active["log_file"]:
                active["log_file"].close()
            info.update(status="stopped", ended_at=utc_now(), stop_reason=reason,
                        video_complete=active["video"].complete,
                        native_returncode=process.returncode if process else None)
            self._save_round(info)
            self.event("round_stopped", **info)
            self.active = None
            return info.copy()

    def completed_video(self, round_id):
        with self.lock:
            info = next((r for r in self.rounds if r["id"] == round_id), None)
            if not info or not info.get("video_complete"):
                raise ValueError("保存が完了した動画がありません")
            directory = Path(info["directory"])
            manifest = json.loads((directory / "video.json").read_text())
            return directory / manifest["file"], manifest["mime"]

    def view(self):
        with self.lock:
            active = self.active
            result = {"task": PUBLIC_TASK, "output": str(self.output), "model_check": self.model_check,
                      "checking": self.checking, "rounds": list(self.rounds), "active": None}
            if active:
                process = active["process"]
                info = active["info"].copy()
                info.update(elapsed_seconds=round(time.monotonic() - active["monotonic_start"], 1),
                            video_bytes=active["video"].bytes, video_chunks=active["video"].chunks,
                            native_running=process is not None and process.poll() is None)
                result["active"] = info
            return result

    def native_view(self):
        with self.lock:
            info = self.active["info"].copy() if self.active else None
        if not info or not info["native_port"]:
            return {}
        result = {}
        for path, key in [("/api/state", "transcript"), ("/api/study", "study")]:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{info['native_port']}{path}", timeout=0.5) as response:
                    result[key] = json.loads(response.read())
            except Exception:
                pass
        return result
