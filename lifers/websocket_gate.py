"""
Lifers WebSocket v1 — 双向实时通信
WebSocket 升级 + 发布订阅 + 流式事件推送
纯标准库实现 (基于 hashlib + struct + base64)
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, List, Callable, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
import socket

ROOT = Path(__file__).resolve().parent.parent

# ============================================================================
# WebSocket 帧协议 (RFC 6455)
# ============================================================================

OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA

WEBSOCKET_MAGIC = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def compute_accept_key(client_key: str) -> str:
    h = hashlib.sha1((client_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
    return base64.b64encode(h).decode()


def encode_frame(payload: bytes, opcode: int = OP_TEXT) -> bytes:
    length = len(payload)
    frame = bytearray([0x80 | opcode])

    if length < 126:
        frame.append(length)
    elif length < 65536:
        frame.append(126)
        frame.extend(struct.pack(">H", length))
    else:
        frame.append(127)
        frame.extend(struct.pack(">Q", length))

    frame.extend(payload)
    return bytes(frame)


def decode_frame(data: bytes) -> tuple:
    if len(data) < 2:
        return None, None, b""

    fin = (data[0] & 0x80) >> 7
    opcode = data[0] & 0x0F
    masked = (data[1] & 0x80) >> 7
    length = data[1] & 0x7F
    offset = 2

    if length == 126:
        if len(data) < 4:
            return None, None, data
        length = struct.unpack(">H", data[2:4])[0]
        offset = 4
    elif length == 127:
        if len(data) < 10:
            return None, None, data
        length = struct.unpack(">Q", data[2:10])[0]
        offset = 10

    if masked:
        if len(data) < offset + 4:
            return None, None, data
        mask = data[offset:offset + 4]
        offset += 4

    if len(data) < offset + length:
        return None, None, data

    payload = data[offset:offset + length]
    if masked:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

    return opcode, payload, data[offset + length:]


# ============================================================================
# WebSocket 连接
# ============================================================================

class WebSocketConnection:
    def __init__(self, sock: socket.socket, addr: tuple, authenticated: dict = None):
        self.sock = sock
        self.addr = addr
        self.authenticated = authenticated
        self.alive = True
        self._lock = threading.Lock()
        self._recv_buf = b""
        self.subscriptions: set = set()

    def send(self, message: str):
        with self._lock:
            try:
                self.sock.sendall(encode_frame(message.encode("utf-8"), OP_TEXT))
            except Exception:
                self.alive = False

    def send_json(self, data: dict):
        self.send(json.dumps(data, ensure_ascii=False))

    def recv(self, timeout: float = 0.1) -> Optional[str]:
        try:
            self.sock.settimeout(timeout)
            chunk = self.sock.recv(65536)
            if not chunk:
                self.alive = False
                return None
            self._recv_buf += chunk
        except socket.timeout:
            pass
        except Exception:
            self.alive = False
            return None

        opcode, payload, remaining = decode_frame(self._recv_buf)
        if opcode is None:
            return None

        self._recv_buf = remaining

        if opcode == OP_TEXT:
            return payload.decode("utf-8")
        elif opcode == OP_CLOSE:
            self.alive = False
        elif opcode == OP_PING:
            self.sock.sendall(encode_frame(payload or b"", OP_PONG))
        elif opcode == OP_PONG:
            pass

        return None

    def close(self):
        self.alive = False
        try:
            self.sock.sendall(encode_frame(b"", OP_CLOSE))
            self.sock.close()
        except Exception:
            pass


# ============================================================================
# WebSocket 服务器
# ============================================================================

class WebSocketServer:
    """WebSocket 服务器 — 发布订阅 + 事件广播"""

    def __init__(self, host: str = "127.0.0.1", port: int = 55557):
        self.host = host
        self.port = port
        self.connections: Dict[str, WebSocketConnection] = {}
        self.topics: Dict[str, set] = defaultdict(set)  # topic -> conn_ids
        self.message_handlers: Dict[str, Callable] = {}
        self._stop = threading.Event()
        self._conn_counter = 0

    def register_handler(self, event_type: str, handler: Callable):
        self.message_handlers[event_type] = handler

    def broadcast(self, topic: str, data: dict):
        data["_topic"] = topic
        data["_ts"] = time.time()
        for conn_id in self.topics.get(topic, set()):
            conn = self.connections.get(conn_id)
            if conn and conn.alive:
                conn.send_json(data)

    def send_to(self, conn_id: str, data: dict):
        conn = self.connections.get(conn_id)
        if conn and conn.alive:
            conn.send_json(data)

    def start(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(50)
        sock.settimeout(1.0)

        print(f"[WebSocket] 服务启动 ws://{self.host}:{self.port}")

        while not self._stop.is_set():
            try:
                conn, addr = sock.accept()
                threading.Thread(target=self._handle, args=(conn, addr), daemon=True).start()
            except socket.timeout:
                continue
            except Exception:
                break

        sock.close()

    def _handle(self, sock: socket.socket, addr: tuple):
        try:
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = sock.recv(4096)
                if not chunk:
                    return
                data += chunk

            request = data.decode("utf-8", errors="replace")
            headers = {}
            for line in request.split("\r\n")[1:]:
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip().lower()] = v.strip()

            key = headers.get("sec-websocket-key", "")
            if not key:
                sock.close()
                return

            accept = compute_accept_key(key)
            response = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n"
                "\r\n"
            )
            sock.sendall(response.encode())

            self._conn_counter += 1
            conn_id = f"ws_{self._conn_counter}_{int(time.time())}"
            ws = WebSocketConnection(sock, addr)
            self.connections[conn_id] = ws

            ws.send_json({"type": "connected", "conn_id": conn_id, "ts": time.time()})

            while ws.alive and not self._stop.is_set():
                msg = ws.recv(timeout=0.5)
                if msg is None:
                    if not ws.alive:
                        break
                    continue

                try:
                    parsed = json.loads(msg)
                    msg_type = parsed.get("type", "")

                    if msg_type == "subscribe":
                        topic = parsed.get("topic", "")
                        self.topics[topic].add(conn_id)
                        ws.subscriptions.add(topic)
                        ws.send_json({"type": "subscribed", "topic": topic})

                    elif msg_type == "unsubscribe":
                        topic = parsed.get("topic", "")
                        self.topics[topic].discard(conn_id)
                        ws.subscriptions.discard(topic)

                    elif msg_type == "ping":
                        ws.send_json({"type": "pong", "ts": time.time()})

                    elif msg_type in self.message_handlers:
                        result = self.message_handlers[msg_type](parsed, conn_id)
                        if result:
                            ws.send_json(result)

                    else:
                        ws.send_json({"type": "echo", "data": parsed})

                except json.JSONDecodeError:
                    ws.send_json({"type": "error", "message": "Invalid JSON"})

        except Exception:
            pass
        finally:
            for topic in ws.subscriptions:
                self.topics[topic].discard(conn_id)
            self.connections.pop(conn_id, None)
            try:
                sock.close()
            except Exception:
                pass

    def shutdown(self):
        self._stop.set()
        for conn in list(self.connections.values()):
            conn.close()


# ============================================================================
# HTTP → WebSocket 升级处理器
# ============================================================================

class WebSocketUpgradeHandler(BaseHTTPRequestHandler):
    ws_server: WebSocketServer = None

    def do_GET(self):
        if self.headers.get("Upgrade", "").lower() == "websocket":
            self._upgrade()
            return
        self.send_response(426)
        self.end_headers()

    def _upgrade(self):
        key = self.headers.get("Sec-WebSocket-Key", "")
        accept = compute_accept_key(key)

        self.send_response(101)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()

        # 提取认证
        auth = None
        from lifers.auth import require_auth
        headers = {k: v for k, v in self.headers.items()}
        auth = require_auth(headers)

        self.ws_server._conn_counter += 1
        conn_id = f"ws_{self.ws_server._conn_counter}_{int(time.time())}"
        ws = WebSocketConnection(self.request, self.client_address, auth)
        self.ws_server.connections[conn_id] = ws

        ws.send_json({"type": "connected", "conn_id": conn_id,
                      "authenticated": auth is not None})

        while ws.alive:
            msg = ws.recv(timeout=0.5)
            if msg is None:
                if not ws.alive:
                    break
                continue
            try:
                parsed = json.loads(msg)
                msg_type = parsed.get("type", "")
                if msg_type == "subscribe":
                    topic = parsed.get("topic", "")
                    self.ws_server.topics[topic].add(conn_id)
                    ws.subscriptions.add(topic)
                    ws.send_json({"type": "subscribed", "topic": topic})
                elif msg_type in self.ws_server.message_handlers:
                    result = self.ws_server.message_handlers[msg_type](parsed, conn_id)
                    if result:
                        ws.send_json(result)
                else:
                    ws.send_json({"type": "echo", "data": parsed})
            except json.JSONDecodeError:
                pass


# ============================================================================
# 实时事件流
# ============================================================================

class EventStream:
    """实时事件流 — 用于推送训练进度、机器人状态、系统告警"""

    def __init__(self, ws_server: WebSocketServer):
        self.ws = ws_server
        self._lock = threading.Lock()

    def push_training_progress(self, pillar: str, epoch: int, total: int, loss: float, acc: float):
        self.ws.broadcast("training", {
            "type": "training_progress",
            "pillar": pillar, "epoch": epoch, "total_epochs": total,
            "loss": round(loss, 6), "accuracy": round(acc, 4),
        })

    def push_robot_status(self, robot_id: str, status: dict):
        self.ws.broadcast(f"robot:{robot_id}", {
            "type": "robot_status", "robot_id": robot_id, **status
        })

    def push_system_alert(self, level: str, message: str, source: str = ""):
        self.ws.broadcast("alerts", {
            "type": "alert", "level": level, "message": message,
            "source": source, "ts": time.time(),
        })

    def push_cluster_event(self, event: dict):
        self.ws.broadcast("cluster", {"type": "cluster_event", **event})

    def push_health_snapshot(self, snapshot: dict):
        self.ws.broadcast("health", {"type": "health_snapshot", **snapshot})


# ============================================================================
# 全局实例
# ============================================================================

_ws_server: Optional[WebSocketServer] = None
_event_stream: Optional[EventStream] = None


def get_ws_server(host: str = "127.0.0.1", port: int = 55557) -> WebSocketServer:
    global _ws_server
    if _ws_server is None:
        _ws_server = WebSocketServer(host, port)
    return _ws_server


def get_event_stream() -> EventStream:
    global _event_stream
    if _event_stream is None:
        _event_stream = EventStream(get_ws_server())
    return _event_stream
