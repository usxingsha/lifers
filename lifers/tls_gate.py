"""
Lifers TLS Gate v1 — HTTPS/TLS 安全网关
TLS + 认证中间件 + 速率限制 + 加密存储
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import ssl
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent.parent
CERT_DIR = ROOT / "config" / "certs"


class RateLimiter:
    """滑动窗口速率限制"""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._clients: Dict[str, list] = {}

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        if client_id not in self._clients:
            self._clients[client_id] = []
        self._clients[client_id] = [t for t in self._clients[client_id] if now - t < self.window]
        if len(self._clients[client_id]) >= self.max_requests:
            return False
        self._clients[client_id].append(now)
        return True

    def cleanup(self):
        now = time.time()
        expired = [k for k, v in self._clients.items() if not v or now - v[-1] > self.window * 2]
        for k in expired:
            del self._clients[k]


def generate_self_signed_cert(cert_dir: Path = None):
    """生成自签名证书 (OpenSSL 封装)"""
    if cert_dir is None:
        cert_dir = CERT_DIR
    cert_dir.mkdir(parents=True, exist_ok=True)

    cert_file = cert_dir / "lifers.crt"
    key_file = cert_dir / "lifers.key"

    if cert_file.exists() and key_file.exists():
        return cert_file, key_file

    import subprocess
    import sys

    # 生成私钥和证书
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:4096", "-nodes",
        "-keyout", str(key_file), "-out", str(cert_file),
        "-days", "3650",
        "-subj", "/C=CN/O=Lifers/CN=lifers.local",
        "-addext", "subjectAltName=DNS:localhost,DNS:lifers.local,IP:127.0.0.1"
    ], capture_output=True, check=False)

    return cert_file, key_file


def create_ssl_context(cert_file: Path = None, key_file: Path = None):
    """创建 TLS SSL 上下文"""
    if cert_file is None:
        cert_file, key_file = generate_self_signed_cert()

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(cert_file), str(key_file))
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM")
    ctx.options |= ssl.OP_NO_COMPRESSION | ssl.OP_SINGLE_ECDH_USE
    return ctx


class SecureHTTPRequestHandler(BaseHTTPRequestHandler):
    """安全的 HTTP 请求处理器 — 认证 + 速率限制 + 租户隔离"""

    routes: Dict[str, Callable] = {}
    rate_limiter = RateLimiter(max_requests=200, window_seconds=60)

    def log_message(self, format, *args):
        pass  # 静默，由监控模块处理

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, X-API-Key, Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            return {}

    def _get_auth(self) -> Optional[dict]:
        from lifers.auth import require_auth
        headers = {k: v for k, v in self.headers.items()}
        return require_auth(headers)

    def _rate_check(self, client_ip: str) -> bool:
        return self.rate_limiter.is_allowed(client_ip)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, X-API-Key, Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        client_ip = self.client_address[0]

        if not self._rate_check(client_ip):
            self._send_json({"error": "Rate limit exceeded"}, 429)
            return

        # 健康检查免认证
        if path == "/health":
            uptime = time.time() - self.server.start_time
            self._send_json({
                "status": "healthy",
                "version": "Lifers v1.0",
                "uptime_seconds": round(uptime, 1),
                "tls": True,
                "timestamp": time.time(),
            })
            return

        # 公开路由
        if path == "/.well-known/lifers.json":
            self._send_json({
                "name": "Lifers AI Brain",
                "version": "1.0",
                "auth_methods": ["bearer_jwt", "api_key", "session"],
                "endpoints": {
                    "chat": "/v1/chat",
                    "stream": "/v1/stream",
                    "health": "/health",
                    "robots": "/v1/robots",
                    "cluster": "/v1/cluster",
                }
            })
            return

        # 需要认证的路由
        auth_payload = self._get_auth()
        if not auth_payload:
            self._send_json({"error": "Unauthorized"}, 401)
            return

        if path in self.routes:
            result = self.routes[path]("GET", auth_payload, parse_qs(parsed.query))
            self._send_json(result)
            return

        self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        client_ip = self.client_address[0]

        if not self._rate_check(client_ip):
            self._send_json({"error": "Rate limit exceeded"}, 429)
            return

        body = self._read_body()

        # 认证端点
        if path == "/auth/login" or path == "/auth/register":
            from lifers.auth import auth_http_handler
            headers = {k: v for k, v in self.headers.items()}
            result = auth_http_handler("POST", path, headers, body)
            if isinstance(result, tuple):
                self._send_json(*result)
            else:
                self._send_json(result)
            return

        # 其他端点需要认证
        auth_payload = self._get_auth()
        if not auth_payload:
            self._send_json({"error": "Unauthorized"}, 401)
            return

        if path in self.routes:
            result = self.routes[path]("POST", auth_payload, body)
            if isinstance(result, tuple):
                self._send_json(*result)
            else:
                self._send_json(result)
            return

        self._send_json({"error": "Not found"}, 404)


class SecureGate:
    """TLS 安全网关"""

    def __init__(self, host: str = "127.0.0.1", port: int = 55555,
                 cert_file: Path = None, key_file: Path = None):
        self.host = host
        self.port = port
        self.cert_file = cert_file
        self.key_file = key_file
        self.server: Optional[HTTPServer] = None

    def register_route(self, path: str, handler: Callable):
        SecureHTTPRequestHandler.routes[path] = handler

    def start(self):
        if self.cert_file is None:
            self.cert_file, self.key_file = generate_self_signed_cert()

        ctx = create_ssl_context(self.cert_file, self.key_file)
        self.server = HTTPServer((self.host, self.port), SecureHTTPRequestHandler)
        self.server.start_time = time.time()
        self.server.socket = ctx.wrap_socket(self.server.socket, server_side=True)

        print(f"[Lifers TLS Gate] HTTPS 服务启动 https://{self.host}:{self.port}")
        print(f"[Lifers TLS Gate] 证书: {self.cert_file}")
        print(f"[Lifers TLS Gate] 密钥: {self.key_file}")

        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        if self.server:
            self.server.shutdown()
            print("[Lifers TLS Gate] 已关闭")


# ============================================================================
# 预定义路由处理器
# ============================================================================

def robot_list_handler(method: str, auth: dict, data: dict = None) -> dict:
    from lifers.robot_hal import get_hal
    hal = get_hal()
    return {
        "robots": [{
            "id": "hal_001",
            "type": "simulated",
            "sensors": list(hal.sensors.keys()) if hal.sensors else [],
            "actuators": list(hal.actuators.keys()) if hal.actuators else [],
            "status": "active",
        }]
    }


def cluster_status_handler(method: str, auth: dict, data: dict = None) -> dict:
    try:
        from lifers.cluster import get_cluster
        cluster = get_cluster()
        return cluster.status()
    except ImportError:
        return {"mode": "standalone", "nodes": 1}


def create_secure_gate(host: str = "127.0.0.1", port: int = 55555) -> SecureGate:
    gate = SecureGate(host, port)
    gate.register_route("/v1/robots", robot_list_handler)
    gate.register_route("/v1/cluster", cluster_status_handler)
    return gate


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Lifers TLS 安全网关")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=55555)
    parser.add_argument("--public", action="store_true", help="绑定 0.0.0.0 (公网)")
    parser.add_argument("--cert", type=Path, help="自定义证书路径")
    parser.add_argument("--key", type=Path, help="自定义密钥路径")
    args = parser.parse_args()

    if args.public:
        args.host = "0.0.0.0"
        print("[警告] 绑定公网地址 0.0.0.0，确保防火墙已配置")

    gate = create_secure_gate(args.host, args.port)
    gate.cert_file = args.cert
    gate.key_file = args.key
    gate.start()


if __name__ == "__main__":
    main()
