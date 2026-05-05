#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lifers — 本地 HTTP 网关（中文代号：终身监禁者）

默认监听 127.0.0.1:55555；仅供受控环境使用，勿将 0.0.0.0 暴露公网。

GET  /  /health  /lifers   — 服务元数据
POST /v1/step  /step      — 请求体同 agent_bridge_once：
                             {"text":"...", "contextFiles":[]}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]


def _ensure_path() -> None:
    sys.path.insert(0, str(ROOT))


def _json_response(handler: BaseHTTPRequestHandler, code: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


class LifersGateHandler(BaseHTTPRequestHandler):
    server_version = "lifers-gate/1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path in ("/", "/health", "/lifers"):
            host, port = self.server.server_address[:2]
            _json_response(
                self,
                200,
                {
                    "ok": True,
                    "service": "lifers",
                    "codename_zh": "终身监禁者",
                    "bind": f"{host}:{port}",
                    "lifers_root": str((ROOT).resolve()),
                    "post_step": 'POST /v1/step  Content-Type: application/json  {"text":"","contextFiles":[]}',
                },
            )
        else:
            _json_response(self, 404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if path not in ("/v1/step", "/step"):
            _json_response(self, 404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length).decode("utf-8", errors="replace")

        _ensure_path()
        os.environ.setdefault("LIFERS_ROOT", str(ROOT.resolve()))
        from lifers_brain.bridge_turn import lifers_turn_from_json_body

        out = lifers_turn_from_json_body(ROOT, raw)
        code = 200 if out.get("ok") else 200
        _json_response(self, code, out)


def main() -> None:
    parser = argparse.ArgumentParser(description="lifers gate (终身监禁者) HTTP service")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (use 0.0.0.0 only in trusted LAN)")
    parser.add_argument("--port", type=int, default=55555, help="listen port (default 55555)")
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), LifersGateHandler)
    print(
        json.dumps(
            {
                "lifers": "gate_start",
                "codename_zh": "终身监禁者",
                "listen": f"{args.host}:{args.port}",
                "root": str(ROOT.resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[lifers] stop", flush=True)


if __name__ == "__main__":
    main()
