#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lifers — 本地 HTTP 网关（中文代号：终身监禁者）

默认监听 127.0.0.1:55555；仅供受控环境使用，勿将 0.0.0.0 暴露公网。

GET  /  /health  /lifers   — 服务元数据
POST /v1/step  /step      — 请求体同 agent_bridge_once：
                             {"text":"...", "contextFiles":[]}
POST /v1/stream /stream  — 简化本地流式字符：{"text":"...","maxChars":200}
                             响应 Transfer-Encoding: chunked，text/plain UTF-8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterator


ROOT = Path(__file__).resolve().parents[1]

# Persistent agent cache (reused across requests to avoid reloading 1.5GB weights)
_GATE_AGENT = None
_GATE_AGENT_MODEL = None


def _get_gate_agent():
    global _GATE_AGENT, _GATE_AGENT_MODEL
    model = os.environ.get("MODEL", "lifers")
    if _GATE_AGENT is not None and _GATE_AGENT_MODEL == model:
        return _GATE_AGENT
    from lifers.agent import LifersAgent
    from lifers.local_brain import AgentConfig
    from lifers.model_names import canonical_brain_model
    sandbox = os.environ.get("SANDBOX", "0") == "1"
    cfg = AgentConfig(root_dir=ROOT, model=canonical_brain_model(model), sandbox=sandbox)
    _GATE_AGENT = LifersAgent(cfg)
    _GATE_AGENT_MODEL = model
    return _GATE_AGENT


def _gate_step(root: Path, raw_json: str) -> dict:
    from lifers.taskflow.classify import split_user_message
    from lifers.taskflow.dialogue_router import infer_dialogue_route
    from lifers.taskflow.context import TaskContext
    from lifers.taskflow.handlers import build_default_dispatcher
    import json as _json
    if not raw_json.strip():
        return {"ok": False, "text": "", "error": "empty body"}
    try:
        data = _json.loads(raw_json)
    except _json.JSONDecodeError as e:
        return {"ok": False, "text": "", "error": f"invalid json: {e}"}
    text = str(data.get("text", "")).strip()
    if not text:
        return {"ok": False, "text": "", "error": "empty text"}
    ctx_files = data.get("contextFiles") or []
    # Build context prefix
    prefix_parts = []
    for rel in (ctx_files if isinstance(ctx_files, list) else []):
        if not isinstance(rel, str) or not rel.strip():
            continue
        rel_n = rel.replace("\\", "/").lstrip("/")
        target = (root / rel_n).resolve()
        try:
            if str(target).startswith(str(root.resolve())) and target.is_file():
                blob = target.read_text(encoding="utf-8", errors="ignore")[:6000]
                if blob:
                    prefix_parts.append(f"--- context file: {rel_n} ---\n{blob}")
        except Exception:
            pass
    full = text
    if prefix_parts:
        full = "\n\n".join(prefix_parts) + "\n\n--- user message ---\n" + text

    agent = _get_gate_agent()
    user_text, has_ctx = split_user_message(full)
    route = infer_dialogue_route(user_text, has_ctx, emit=True, planner=agent.planner)
    ctx = TaskContext(
        agent=agent, agent_input=full, user_text=user_text,
        kind=route.kind, dialogue_route_reason=route.reason,
        dialogue_route_notes_zh=route.notes_zh,
    )
    dispatcher = build_default_dispatcher()
    res = dispatcher.dispatch(route.kind, ctx)
    return {"ok": True, "text": res.reply}


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


def _chunked_plain_utf8(handler: BaseHTTPRequestHandler, char_iter: Iterator[str]) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Transfer-Encoding", "chunked")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    for s in char_iter:
        b = s.encode("utf-8")
        handler.wfile.write(f"{len(b):x}\r\n".encode("ascii"))
        handler.wfile.write(b)
        handler.wfile.write(b"\r\n")
    handler.wfile.write(b"0\r\n\r\n")


class LifersGateHandler(BaseHTTPRequestHandler):
    server_version = "lifers-gate/1"

    def log_message(self, fmt: str, *args: Any) -> None:
        from lifers.silent_mode import is_silent
        if is_silent():
            return  # 静默模式压制HTTP日志
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
                    "capabilities": {
                        "auth": True, "tls": os.environ.get("LIFERS_TLS", "") == "1",
                        "cluster": os.environ.get("LIFERS_CLUSTER", "") == "1",
                        "websocket": True, "fleet": True,
                    },
                    "endpoints": {
                        "step": "POST /v1/step",
                        "stream": "POST /v1/stream",
                        "auth_login": "POST /auth/login",
                        "auth_register": "POST /auth/register",
                        "auth_api_key": "POST /auth/api-key",
                        "fleet_robots": "GET /v1/fleet/robots",
                        "fleet_health": "GET /v1/fleet/health",
                        "cluster_status": "GET /v1/cluster/status",
                    },
                },
            )
        elif path.startswith("/v1/fleet/"):
            # 舰队管理端点
            from lifers.fleet import fleet_http_handler
            from lifers.auth import require_auth
            headers = {k: v for k, v in self.headers.items()}
            auth_payload = require_auth(headers)
            if not auth_payload:
                _json_response(self, 401, {"error": "Unauthorized"})
                return
            sub_path = "/" + path.split("/", 3)[-1] if len(path.split("/")) > 3 else path
            result = fleet_http_handler("GET", path, auth_payload)
            _json_response(self, 200, result)
        elif path.startswith("/v1/cluster/"):
            from lifers.cluster import cluster_http_handler
            result = cluster_http_handler("GET", path, {})
            _json_response(self, 200, result)
        else:
            _json_response(self, 404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")

        # ---- 认证端点 (无需预认证) ----
        if path in ("/auth/login", "/auth/register", "/auth/api-key"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            body = json.loads(raw) if raw.strip() else {}
            from lifers.auth import auth_http_handler
            headers = {k: v for k, v in self.headers.items()}
            result = auth_http_handler("POST", path, headers, body)
            if isinstance(result, tuple):
                data, code = result
                _json_response(self, code, data)
            else:
                _json_response(self, 200, result)
            return

        # ---- 舰队管理端点 ----
        if path.startswith("/v1/fleet/"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            body = json.loads(raw) if raw.strip() else {}
            from lifers.fleet import fleet_http_handler
            from lifers.auth import require_auth
            headers = {k: v for k, v in self.headers.items()}
            auth_payload = require_auth(headers)
            if not auth_payload:
                _json_response(self, 401, {"error": "Unauthorized"})
                return
            result = fleet_http_handler("POST", path, auth_payload, body)
            if isinstance(result, tuple):
                data, code = result
                _json_response(self, code, data)
            else:
                _json_response(self, 200, result)
            return

        # ---- 原有端点 ----
        if path not in ("/v1/step", "/step", "/v1/stream", "/stream"):
            _json_response(self, 404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length).decode("utf-8", errors="replace")

        _ensure_path()
        os.environ.setdefault("LIFERS_ROOT", str(ROOT.resolve()))

        if path in ("/v1/stream", "/stream"):
            from lifers.bridge_turn import iter_stream_simple_chars

            if not raw.strip():
                _json_response(self, 400, {"ok": False, "error": "empty body"})
                return
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                _json_response(self, 400, {"ok": False, "error": f"invalid json: {e}"})
                return
            text = str(data.get("text", ""))
            try:
                max_chars = int(data.get("maxChars", data.get("max_chars", 200)))
            except (TypeError, ValueError):
                max_chars = 200
            max_chars = max(1, min(max_chars, 8000))
            try:
                _chunked_plain_utf8(self, iter_stream_simple_chars(ROOT, text, max_chars=max_chars))
            except Exception as e:
                _json_response(self, 500, {"ok": False, "error": str(e)})
            return

        # Use persistent agent cache for faster subsequent requests
        try:
            out = _gate_step(ROOT, raw)
        except Exception:
            from lifers.bridge_turn import lifers_turn_from_json_body
            out = lifers_turn_from_json_body(ROOT, raw)
        code = 200 if out.get("ok") else 200
        _json_response(self, code, out)


def main() -> None:
    parser = argparse.ArgumentParser(description="lifers gate (终身监禁者) HTTP service")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (use 0.0.0.0 only in trusted LAN)")
    parser.add_argument("--port", type=int, default=55555, help="listen port (default 55555)")
    parser.add_argument("--silent", action="store_true", help="静默模式（输出写入文件）")
    args = parser.parse_args()

    if args.silent:
        os.environ["LIFERS_SILENT"] = "1"
    from lifers.silent_mode import is_silent, service_banner
    if is_silent():
        from lifers.silent_mode import setup_silent
        setup_silent("gate")

    httpd = ThreadingHTTPServer((args.host, args.port), LifersGateHandler)
    print(service_banner("gate_start", codename_zh="终身监禁者", listen=f"{args.host}:{args.port}", root=str(ROOT.resolve())), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\n{"lifers":"gate_stop"}', flush=True)


if __name__ == "__main__":
    main()
