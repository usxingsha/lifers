"""
Lifers 自研 GUI 宿主：单进程 HTTP 服务
- 静态 UI（/、/static/*）
- POST /api/bridge  — 请求体与 agent_bridge_once / lifers_gate 相同 JSON
- GET  /api/editor-settings — 来自仓库 tools/vscodium_editor_defaults.json 的主题映射

默认仅监听 127.0.0.1；与扩展一致可向子逻辑注入环境变量（每请求临时覆盖）。
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import subprocess
import sys
import tempfile
import time
import webbrowser
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List


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
            from lifers.silent_mode import is_silent
            if is_silent():
                return
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
            if path == "/api/monitor":
                mf = brain_root / "weights" / ".monitor_status.json"
                if mf.is_file():
                    try:
                        data = json.loads(mf.read_text(encoding="utf-8"))
                        _json(self, 200, {"ok": True, "monitor": data})
                    except Exception:
                        _json(self, 500, {"ok": False, "error": "parse error"})
                else:
                    _json(self, 200, {"ok": True, "monitor": None, "hint": "monitor not running"})
                return
            if path == "/api/files":
                self._handle_api_files()
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

        # ── GET /api/files?path=PATH ──────────────────────────
        def _handle_api_files(self) -> None:
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            req_path = qs.get("path", ["."])[0]
            if ".." in req_path or req_path.startswith("/"):
                _json(self, 403, {"ok": False, "error": "invalid path"})
                return
            target = (brain_root / req_path).resolve()
            try:
                target.relative_to(brain_root.resolve())
            except ValueError:
                _json(self, 403, {"ok": False, "error": "path outside root"})
                return
            if not target.exists():
                _json(self, 404, {"ok": False, "error": "not found"})
                return
            if target.is_file():
                try:
                    content = target.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    content = "[binary file]"
                _json(self, 200, {"ok": True, "type": "file", "content": content, "name": target.name})
                return
            if target.is_dir():
                tree: List[Dict[str, Any]] = []
                try:
                    for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
                        node: Dict[str, Any] = {
                            "name": child.name,
                            "type": "dir" if child.is_dir() else "file",
                            "path": str(child.relative_to(brain_root)).replace("\\", "/"),
                        }
                        if child.is_dir():
                            try:
                                node["children"] = [
                                    {
                                        "name": gc.name,
                                        "type": "dir" if gc.is_dir() else "file",
                                        "path": str(gc.relative_to(brain_root)).replace("\\", "/"),
                                    }
                                    for gc in sorted(child.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))[:100]
                                ]
                            except PermissionError:
                                node["children"] = []
                        tree.append(node)
                except PermissionError:
                    _json(self, 403, {"ok": False, "error": "permission denied"})
                    return
                _json(self, 200, {"ok": True, "type": "dir", "tree": tree, "name": target.name})
                return

        # ── POST /api/upload  /  POST /api/exec ───────────────
        def _handle_data_post(self, path: str) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                _json(self, 400, {"ok": False, "error": "invalid JSON"})
                return

            if path == "/api/upload":
                files = body.get("files") or body.get("items") or []
                if not isinstance(files, list):
                    _json(self, 400, {"ok": False, "error": "files must be a list"})
                    return
                upload_dir = brain_root / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                saved: List[str] = []
                for f in files:
                    name = f.get("name", "untitled")
                    safe_name = Path(name).name.replace("\\", "/").split("/")[-1] or "untitled"
                    if safe_name.startswith("."):
                        safe_name = "_" + safe_name
                    data_b64 = f.get("data_base64", f.get("data", ""))
                    try:
                        raw_bytes = base64.b64decode(data_b64)
                    except Exception:
                        continue
                    # 限制单文件 50MB
                    if len(raw_bytes) > 50 * 1024 * 1024:
                        continue
                    dest = upload_dir / safe_name
                    ts = str(int(time.time() * 1000))
                    if dest.exists():
                        dest = upload_dir / f"{dest.stem}_{ts}{dest.suffix}"
                    dest.write_bytes(raw_bytes)
                    rel = str(dest.relative_to(brain_root)).replace("\\", "/")
                    saved.append(rel)
                _json(self, 200, {"ok": True, "paths": saved, "count": len(saved)})
                return

            if path == "/api/exec":
                cmd = body.get("cmd", "").strip()
                shell_name = body.get("shell", "cmd")
                if not cmd:
                    _json(self, 400, {"ok": False, "error": "cmd is required"})
                    return
                if len(cmd) > 8000:
                    _json(self, 400, {"ok": False, "error": "cmd too long"})
                    return
                try:
                    if shell_name == "powershell":
                        full_cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd]
                    elif shell_name == "bash":
                        full_cmd = ["bash", "-c", cmd]
                    else:
                        full_cmd = ["cmd", "/c", cmd]
                    proc = subprocess.run(
                        full_cmd,
                        capture_output=True,
                        timeout=30,
                        cwd=str(brain_root),
                        env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    )
                    _json(self, 200, {
                        "ok": True,
                        "stdout": proc.stdout.decode("utf-8", errors="replace")[:50000],
                        "stderr": proc.stderr.decode("utf-8", errors="replace")[:50000],
                        "exit_code": proc.returncode,
                    })
                except subprocess.TimeoutExpired:
                    _json(self, 200, {"ok": True, "stdout": "", "stderr": "Command timed out after 30s", "exit_code": -1})
                except Exception as e:
                    _json(self, 500, {"ok": False, "error": str(e)})
                return

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0].rstrip("/")

            # 流式端点代理到 gate
            if path in ("/api/stream", "/v1/stream"):
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = 0
                raw = self.rfile.read(length).decode("utf-8", errors="replace")
                try:
                    import urllib.request
                    body = raw.encode("utf-8")
                    req = urllib.request.Request(
                        "http://127.0.0.1:55555/v1/stream",
                        data=body,
                        headers={"Content-Type": "application/json; charset=utf-8"},
                    )
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/plain; charset=utf-8")
                        self.send_header("Transfer-Encoding", "chunked")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        while True:
                            chunk = resp.read(4096)
                            if not chunk:
                                break
                            self.wfile.write(f"{len(chunk):x}\r\n".encode("ascii"))
                            self.wfile.write(chunk)
                            self.wfile.write(b"\r\n")
                        self.wfile.write(b"0\r\n\r\n")
                except Exception as e:
                    _json(self, 500, {"ok": False, "error": f"Stream proxy: {e}"})
                return

            if path in ("/api/upload", "/api/exec"):
                self._handle_data_post(path)
                return

            if path != "/api/bridge":
                _json(self, 404, {"ok": False, "error": f"not found: {path}"})
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
                "LIFERS_QUICK_TEMPLATE_SHORTCUTS",
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
            os.environ.setdefault("LIFERS_QUICK_TEMPLATE_SHORTCUTS", "1")
            os.environ.setdefault("LIFERS_MICRO_THINK_EVERY", "999")
            os.environ.setdefault("LIFERS_MAX_SPEED", "1")
            os.environ.setdefault("LIFERS_QUICK_WEB", "0")

            try:
                from lifers.bridge_turn import lifers_turn_from_json_body

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
    parser.add_argument("--brain-root", type=Path, default=brain_default, help="lifers 根（含 config/stack.json）")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--webview",
        action="store_true",
        help="使用 pywebview 内嵌窗口（需 pip install pywebview；无则退出码 3）",
    )
    parser.add_argument("--silent", action="store_true", help="静默模式")
    args = parser.parse_args()

    if args.silent:
        os.environ["LIFERS_SILENT"] = "1"
    from lifers.silent_mode import is_silent
    if is_silent():
        from lifers.silent_mode import setup_silent
        setup_silent("gui_host")

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
