"""Zero-dependency local dashboard server (stdlib only).

Serves a single-page UI plus a tiny JSON API over the JSONL event logs that scenario
and society runs write. The page polls ``/api/runs/<id>/events?after=<seq>`` so a run
that is still in progress appears live, and finished runs replay the same way.
"""
from __future__ import annotations

import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from genesis.events import DEFAULT_RUNS_DIR, iter_runs, read_events, summarize_run

STATIC_DIR = Path(__file__).resolve().parent / "static"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def make_handler(runs_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        server_version = "GenesisMonitor/1.0"

        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def _send(self, status, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, status=HTTPStatus.OK):
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def do_GET(self):  # noqa: N802 - stdlib naming
            url = urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            if not parts or parts == ["index.html"]:
                page = (STATIC_DIR / "index.html").read_bytes()
                return self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
            if parts == ["api", "runs"]:
                runs = [s for s in (summarize_run(d) for d in iter_runs(runs_dir)) if s]
                runs.sort(key=lambda r: r["started_at"], reverse=True)
                return self._json({"runs": runs})
            if len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "events":
                run_id = parts[2]
                if not RUN_ID_RE.match(run_id):
                    return self._json({"error": "bad run id"}, HTTPStatus.BAD_REQUEST)
                path = runs_dir / run_id / "events.jsonl"
                if not path.exists():
                    return self._json({"error": "unknown run"}, HTTPStatus.NOT_FOUND)
                try:
                    after = int(parse_qs(url.query).get("after", ["0"])[0])
                except ValueError:
                    after = 0
                events = read_events(path, after)
                finished = bool(events) and events[-1]["type"] == "run_end" or any(
                    e["type"] == "run_end" for e in events)
                return self._json({"events": events, "finished": finished})
            if len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "download":
                run_id = parts[2]
                if not RUN_ID_RE.match(run_id):
                    return self._json({"error": "bad run id"}, HTTPStatus.BAD_REQUEST)
                path = runs_dir / run_id / "events.jsonl"
                if not path.exists():
                    return self._json({"error": "unknown run"}, HTTPStatus.NOT_FOUND)
                body = path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Disposition", f'attachment; filename="{run_id}-events.jsonl"')
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    return Handler


def serve(runs_dir=None, host="127.0.0.1", port=8765) -> ThreadingHTTPServer:
    runs_dir = Path(runs_dir or DEFAULT_RUNS_DIR)
    runs_dir.mkdir(parents=True, exist_ok=True)
    return ThreadingHTTPServer((host, port), make_handler(runs_dir))
