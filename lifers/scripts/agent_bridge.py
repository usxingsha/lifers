"""
lifers/scripts/agent_bridge.py
──────────────────────────────────────
Persistent HTTP+SSE bridge between VSCodium extension and Python backend.
Replaces agent_bridge_once.py (single-shot) with a proper server.

Endpoints
─────────
POST /chat          { "text":"...", "session_id":"...", "npc":"..." }
GET  /health        { "status":"ok", "uptime_s":N }
POST /npc/suspend   { "npc":"..." }
GET  /npc/list      { "active":[...] }
POST /reset         { "session_id":"..." }
"""
from __future__ import annotations
import json, logging, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.input_validator  import validate_input
from core.inference_router import InferenceRouter, Route
from core.output_formatter import from_stack as formatter_from_stack
from core.npc.npc_manager  import NPCManager

log        = logging.getLogger("agent_bridge")
_STACK_P   = ROOT / "config" / "stack.json"
with _STACK_P.open(encoding="utf-8") as _f:
    STACK: dict = json.load(_f)

_router    = InferenceRouter(stack_path=_STACK_P)
_formatter = formatter_from_stack(STACK)
_npc       = NPCManager(STACK)
_start_ts  = time.time()
_sessions: dict[str, list[dict]] = {}


# ── Inference via LifersAgent (real model call) ────────────────────────────────

_AGENT = None

def _get_agent():
    global _AGENT
    if _AGENT is None:
        from lifers.agent import LifersAgent
        from lifers.local_brain import AgentConfig
        from lifers.model_names import canonical_brain_model
        model = canonical_brain_model(os.environ.get("MODEL", "lifers"))
        sandbox = os.environ.get("SANDBOX", "0") == "1"
        _AGENT = LifersAgent(AgentConfig(root_dir=ROOT, model=model, sandbox=sandbox))
    return _AGENT

def _infer(prompt: str, meta: dict) -> str:
    route = meta.get("route", Route.LOCAL)
    if route == Route.LOCAL or route == Route.NPC:
        agent = _get_agent()
        try:
            max_chars = int(os.environ.get("LIFERS_QUICK_CHAT_OUT_CHARS", "200"))
        except ValueError:
            max_chars = 200
        return agent.quick_chat(prompt)
    if route == Route.REMOTE:
        return agent.step(prompt)
    return agent.step(prompt)


# ── HTTP Handler ──────────────────────────────────────────────────────────────

class Bridge(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        log.debug(fmt, *args)

    def do_OPTIONS(self):
        self.send_response(204)
        for h, v in [("Access-Control-Allow-Origin", "*"),
                     ("Access-Control-Allow-Methods", "GET,POST,OPTIONS"),
                     ("Access-Control-Allow-Headers", "Content-Type")]:
            self.send_header(h, v)
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._json({"status": "ok", "uptime_s": round(time.time() - _start_ts, 1)})
        elif self.path == "/npc/list":
            self._json({"active": _npc.list_active()})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        body = self._body()
        if   self.path == "/chat":        self._chat(body)
        elif self.path == "/npc/suspend": _npc.suspend(body.get("npc","")); self._json({"ok":True})
        elif self.path == "/reset":
            sid = body.get("session_id","default"); _sessions.pop(sid, None)
            self._json({"ok": True, "session_id": sid})
        else:
            self._json({"error": "not found"}, 404)

    # ── /chat ─────────────────────────────────────────────────────────────────

    def _chat(self, body: dict):
        raw      = body.get("text", "").strip()
        sid      = body.get("session_id", "default")
        npc_name = body.get("npc", "").strip()

        # 1. Validate
        v = validate_input(raw)
        if v.rejected:
            return self._sse_err(v.errors[0])
        if v.warnings:
            log.warning("Input warnings: %s", v.warnings)

        # 2. Route
        ctx      = {"session_id": sid}
        if npc_name:
            ctx["active_npc"] = npc_name
        decision = _router.route(v.text, context=ctx)
        log.info("Route→%s intent=%s", decision.route, decision.intent)

        # 3. Build prompt
        if decision.route == Route.NPC and npc_name:
            sess = _npc.get_or_create(npc_name)
            if not sess:
                return self._sse_err(f"NPC '{npc_name}' not found")
            prompt, npc_meta = sess.build_prompt(v.text)
            decision.meta.update(npc_meta)
        else:
            hist = _sessions.setdefault(sid, [])
            hist.append({"role": "user", "content": v.text})
            prompt = _build_chat_prompt(hist)

        # 4. Infer
        meta = {**decision.meta, "route": decision.route}
        raw_response = _infer(prompt, meta)

        # 5. Format
        result = _formatter.format(raw_response)
        if result.hallucination_flag:
            log.warning("Hallucination flag: %s", result.warnings)

        # 6. Persist
        if decision.route == Route.NPC and npc_name:
            s2 = _npc.get_or_create(npc_name)
            if s2:
                s2.record(v.text, result.text)
        else:
            _sessions.setdefault(sid, []).append(
                {"role": "assistant", "content": result.text})

        # 7. Stream
        self._sse_ok(result.text, {
            "confidence":        result.confidence,
            "hallucination_flag": result.hallucination_flag,
            "warnings":          result.warnings,
            "route":             decision.route.value,
            "intent":            decision.intent.value,
        })

    # ── SSE helpers ───────────────────────────────────────────────────────────

    def _sse_ok(self, text: str, meta: Optional[dict] = None):
        self._sse_headers()
        for i in range(0, len(text), 40):
            self.wfile.write(f"data: {json.dumps({'chunk': text[i:i+40]})}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(f"data: {json.dumps({'done': True, 'meta': meta or {}})}\n\n".encode())
        self.wfile.flush()

    def _sse_err(self, msg: str):
        self._sse_headers()
        self.wfile.write(f"data: {json.dumps({'error': msg})}\n\n".encode())
        self.wfile.flush()

    def _sse_headers(self):
        self.send_response(200)
        self.send_header("Content-Type",  "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _json(self, data: dict, code: int = 200):
        payload = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except json.JSONDecodeError:
            return {}


def _build_chat_prompt(history: list[dict]) -> str:
    lines = []
    for m in history[-20:]:
        tag = "<|user|>" if m["role"] == "user" else "<|assistant|>"
        lines.append(f"{tag}\n{m['content']}")
    lines.append("<|assistant|>\n")
    return "\n".join(lines)


def main():
    host = STACK.get("gate", {}).get("host", "127.0.0.1")
    port = int(STACK.get("gate", {}).get("port", 55555))
    srv  = HTTPServer((host, port), Bridge)
    log.info("Bridge running on http://%s:%d", host, port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log.info("Bridge stopped")
        srv.server_close()


if __name__ == "__main__":
    main()
