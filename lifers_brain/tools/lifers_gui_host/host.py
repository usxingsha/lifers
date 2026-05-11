"""
Lifers 自研 GUI 宿主：单进程 HTTP 服务
- 静态 UI（/、/static/*）
- POST /api/bridge  — 请求体与 agent_bridge_once / lifers_gate 相同 JSON
- GET  /api/editor-settings — 来自仓库 tools/vscodium_editor_defaults.json 的主题映射

默认仅监听 127.0.0.1；与扩展一致可向子逻辑注入环境变量（每请求临时覆盖）。
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import webbrowser
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict


def _pkg_static_dir() -> Path:
    return Path(__file__).resolve().parent / "static"


def _json(handler: BaseHTTPRequestHandler, code: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(brain_root: Path, repo_root: Path):
    static_root = _pkg_static_dir()

    class H(BaseHTTPRequestHandler):
        server_version = "lifers-gui-host/1"

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
            if path == "/health":
                _json(
                    self,
                    200,
                    {
                        "ok": True,
                        "service": "lifers_gui_host",
                        "lifersRoot": str(brain_root.resolve()),
                    },
                )
                return
            if path == "/api/editor-settings":
                from tools.lifers_gui_host.vscodium_settings import (
                    gui_theme_from_defaults,
                    load_vscodium_defaults_json,
                )

                raw = load_vscodium_defaults_json(repo_root)
                _json(
                    self,
                    200,
                    {
                        "ok": True,
                        "lifersRoot": str(brain_root.resolve()),
                        "repoRoot": str(repo_root.resolve()),
                        "theme": gui_theme_from_defaults(raw),
                        "defaultsKeys": sorted(raw.keys())[:80],
                    },
                )
                return
            if path in ("/", "/index.html"):
                self._send_file(static_root / "index.html", "text/html; charset=utf-8")
                return
            if path.startswith("/static/"):
                rel = path[len("/static/") :].lstrip("/")
                if ".." in rel or rel.startswith("/"):
                    self.send_error(400)
                    return
                fp = (static_root / rel).resolve()
                try:
                    fp.relative_to(static_root.resolve())
                except ValueError:
                    self.send_error(403)
                    return
                if not fp.is_file():
                    self.send_error(404)
                    return
                ctype, _ = mimetypes.guess_type(str(fp))
                self._send_file(fp, ctype or "application/octet-stream")
                return
            self.send_error(404)

        def _send_file(self, fp: Path, ctype: str) -> None:
            data = fp.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0].rstrip("/")
            if path != "/api/bridge":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            raw = self.rfile.read(length).decode("utf-8", errors="replace")

            root = brain_root.resolve()
            sys.path.insert(0, str(root))
            prev: Dict[str, str | None] = {}
            env_keys = (
                "LIFERS_ROOT",
                "SANDBOX",
                "MODEL",
                "LIFERS_FORCE_LOCAL_ONLY",
                "LIFERS_TASKFLOW",
                "LIFERS_QUICK_CHAT_LEARN",
                "LIFERS_MICRO_THINK_EVERY",
                "LIFERS_MAX_SPEED",
                "LIFERS_QUICK_WEB",
                "LIFERS_HTTP_DIRECT",
            )
            for k in env_keys:
                prev[k] = os.environ.get(k)

            os.environ["LIFERS_ROOT"] = str(root)
            os.environ.setdefault("SANDBOX", "1")
            os.environ.setdefault("MODEL", "lifers")
            os.environ.setdefault("LIFERS_FORCE_LOCAL_ONLY", "1")
            os.environ.setdefault("LIFERS_TASKFLOW", "1")
            os.environ.setdefault("LIFERS_QUICK_CHAT_LEARN", "0")
            os.environ.setdefault("LIFERS_MICRO_THINK_EVERY", "999")
            os.environ.setdefault("LIFERS_MAX_SPEED", "1")
            os.environ.setdefault("LIFERS_QUICK_WEB", "0")

            try:
                from lifers_brain.bridge_turn import lifers_turn_from_json_body

                out = lifers_turn_from_json_body(root, raw)
            except Exception as e:
                out = {"ok": False, "text": "", "error": f"{type(e).__name__}: {e}"}
            finally:
                for k, v in prev.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

            _json(self, 200, out)

    return H


def build_httpd(brain_root: Path, repo_root: Path, host: str, port: int) -> ThreadingHTTPServer:
    handler = make_handler(brain_root, repo_root)
    return ThreadingHTTPServer((host, port), handler)


def serve_background(httpd: ThreadingHTTPServer) -> threading.Thread:
    t = threading.Thread(target=httpd.serve_forever, name="lifers-gui-host", daemon=True)
    t.start()
    return t


def main() -> int:
    here = Path(__file__).resolve().parent
    brain_default = here.parents[2]
    parser = argparse.ArgumentParser(description="Lifers self GUI + bridge host")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--brain-root", type=Path, default=brain_default, help="lifers_brain 根（含 config/stack.json）")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--webview",
        action="store_true",
        help="使用 pywebview 内嵌窗口（需 pip install pywebview；无则退出码 3）",
    )
    args = parser.parse_args()

    brain_root: Path = args.brain_root.resolve()
    sys.path.insert(0, str(brain_root))
    repo_root = brain_root.parent
    if not (brain_root / "scripts" / "agent_bridge_once.py").is_file():
        print(f"invalid --brain-root (missing agent_bridge_once.py): {brain_root}", file=sys.stderr)
        return 2

    httpd = build_httpd(brain_root, repo_root, args.host, args.port)
    url = f"http://{args.host}:{args.port}/"
    print(json.dumps({"lifers": "gui_host", "listen": f"{args.host}:{args.port}", "url": url}, ensure_ascii=False), flush=True)

    if args.webview:
        try:
            import webview  # type: ignore
        except ImportError:
            print("pywebview 未安装：pip install pywebview", file=sys.stderr)
            return 3
        serve_background(httpd)
        try:
            webview.create_window("Lifers · GUI Host", url, width=1280, height=840)
            webview.start()
        finally:
            httpd.shutdown()
        return 0

    if not args.no_browser and args.host in ("127.0.0.1", "localhost"):
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[lifers] gui host stop", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
