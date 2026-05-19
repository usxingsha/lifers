"""
Lifers Cluster v1 — 分布式集群架构
Raft 共识 + 主从选举 + 心跳 + 状态同步 + 脑裂防护
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler

ROOT = Path(__file__).resolve().parent.parent
CLUSTER_DIR = ROOT / "state" / "cluster"

# ============================================================================
# Raft 共识核心
# ============================================================================

@dataclass
class LogEntry:
    term: int
    index: int
    command: str
    data: dict
    timestamp: float = field(default_factory=time.time)


class RaftNode:
    """Raft 共识节点"""

    def __init__(self, node_id: str, peers: List[str], data_dir: Path = None):
        self.node_id = node_id
        self.peers = peers  # ["node1:55560", "node2:55560", ...]
        self.data_dir = data_dir or (CLUSTER_DIR / node_id)

        # 持久状态
        self.current_term: int = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []

        # 易失状态
        self.commit_index: int = -1
        self.last_applied: int = -1

        # 领导者状态
        self.next_index: Dict[str, int] = defaultdict(int)
        self.match_index: Dict[str, int] = defaultdict(lambda: -1)

        # 角色
        self.role: str = "follower"  # follower, candidate, leader
        self.leader_id: Optional[str] = None
        self.last_heartbeat: float = 0
        self.election_timeout: float = random.uniform(3.0, 6.0)
        self.election_timer: float = time.time()

        # 状态机
        self.state: Dict[str, Any] = {}
        self.state_handlers: Dict[str, Callable] = {}

        # 网络
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._peers_http: Dict[str, str] = {}  # node_id -> "host:port"

        self._load_state()
        self._parse_peer_addrs()

    def _parse_peer_addrs(self):
        for p in self.peers:
            parts = p.split("@")
            if len(parts) == 2:
                self._peers_http[parts[0]] = parts[1]
            else:
                self._peers_http[p] = p

    def _load_state(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        state_file = self.data_dir / "raft_state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                data = json.load(f)
                self.current_term = data.get("term", 0)
                self.voted_for = data.get("voted_for")
                self.log = [LogEntry(**e) for e in data.get("log", [])]
                self.commit_index = data.get("commit_index", -1)

    def _save_state(self):
        state_file = self.data_dir / "raft_state.json"
        with open(state_file, "w") as f:
            json.dump({
                "term": self.current_term,
                "voted_for": self.voted_for,
                "log": [{"term": e.term, "index": e.index, "command": e.command,
                         "data": e.data, "timestamp": e.timestamp} for e in self.log],
                "commit_index": self.commit_index,
            }, f, ensure_ascii=False, indent=2)

    def register_handler(self, command: str, handler: Callable):
        self.state_handlers[command] = handler

    def propose(self, command: str, data: dict) -> bool:
        if self.role != "leader":
            return False
        with self._lock:
            entry = LogEntry(
                term=self.current_term,
                index=len(self.log),
                command=command,
                data=data,
            )
            self.log.append(entry)
            self.match_index[self.node_id] = entry.index
            self.next_index[self.node_id] = entry.index + 1
            self._save_state()
            self._try_commit()
            return True

    def _try_commit(self):
        while self.commit_index + 1 < len(self.log):
            next_idx = self.commit_index + 1
            entry = self.log[next_idx]
            if entry.term != self.current_term:
                self.commit_index = next_idx
                continue
            votes = sum(1 for nid, mi in self.match_index.items() if mi >= next_idx)
            if votes > len(self.peers) // 2:
                self.commit_index = next_idx
                self._apply(entry)
            else:
                break
        self._save_state()

    def _apply(self, entry: LogEntry):
        if entry.command in self.state_handlers:
            self.state_handlers[entry.command](entry.data)
        self.state[entry.command] = entry.data
        self.last_applied = entry.index

    def request_vote(self, candidate_id: str, term: int,
                     last_log_index: int, last_log_term: int) -> dict:
        with self._lock:
            if term < self.current_term:
                return {"term": self.current_term, "vote_granted": False}

            if term > self.current_term:
                self.current_term = term
                self.role = "follower"
                self.voted_for = None

            can_vote = (self.voted_for is None or self.voted_for == candidate_id)
            log_ok = (last_log_term > self.log[-1].term if self.log else True) or \
                     (last_log_term == (self.log[-1].term if self.log else 0) and
                      last_log_index >= len(self.log) - 1)

            if can_vote and log_ok:
                self.voted_for = candidate_id
                self._save_state()
                return {"term": self.current_term, "vote_granted": True}

            return {"term": self.current_term, "vote_granted": False}

    def append_entries(self, leader_id: str, term: int, prev_log_index: int,
                       prev_log_term: int, entries: list, leader_commit: int) -> dict:
        with self._lock:
            if term < self.current_term:
                return {"term": self.current_term, "success": False}

            self.last_heartbeat = time.time()
            self.election_timer = time.time()

            if term > self.current_term:
                self.current_term = term
                self.role = "follower"

            self.leader_id = leader_id

            # 日志一致性检查
            if prev_log_index >= 0:
                if prev_log_index >= len(self.log) or \
                   self.log[prev_log_index].term != prev_log_term:
                    return {"term": self.current_term, "success": False}

            # 追加条目
            for i, entry_data in enumerate(entries):
                idx = prev_log_index + 1 + i
                entry = LogEntry(**entry_data) if isinstance(entry_data, dict) else entry_data
                if idx < len(self.log):
                    if self.log[idx].term != entry.term:
                        self.log = self.log[:idx]
                        self.log.append(entry)
                else:
                    self.log.append(entry)

            # 提交
            if leader_commit > self.commit_index:
                old_commit = self.commit_index
                self.commit_index = min(leader_commit, len(self.log) - 1)
                for i in range(old_commit + 1, self.commit_index + 1):
                    if i < len(self.log):
                        self._apply(self.log[i])

            self._save_state()
            return {"term": self.current_term, "success": True}

    def start_election(self):
        with self._lock:
            self.current_term += 1
            self.role = "candidate"
            self.voted_for = self.node_id
            self._save_state()

        votes = 1
        last_idx = len(self.log) - 1
        last_term = self.log[-1].term if self.log else 0

        for peer_id, addr in self._peers_http.items():
            try:
                result = self._rpc(addr, "request_vote", {
                    "candidate_id": self.node_id,
                    "term": self.current_term,
                    "last_log_index": last_idx,
                    "last_log_term": last_term,
                })
                if result and result.get("vote_granted"):
                    votes += 1
                    if result.get("term", 0) > self.current_term:
                        self.current_term = result["term"]
                        self.role = "follower"
                        return
            except Exception:
                pass

        if votes > len(self.peers) // 2 and self.role == "candidate":
            self._become_leader()

    def _become_leader(self):
        self.role = "leader"
        self.leader_id = self.node_id
        for peer in self.peers:
            self.next_index[peer] = len(self.log)
            self.match_index[peer] = -1
        print(f"[Cluster] {self.node_id} 成为 Leader (term={self.current_term})")

        # 发送心跳
        for peer_id, addr in self._peers_http.items():
            self._send_heartbeat(peer_id, addr)

    def _send_heartbeat(self, peer_id: str, addr: str):
        try:
            self._rpc(addr, "append_entries", {
                "leader_id": self.node_id,
                "term": self.current_term,
                "prev_log_index": len(self.log) - 1,
                "prev_log_term": self.log[-1].term if self.log else 0,
                "entries": [],
                "leader_commit": self.commit_index,
            })
        except Exception:
            pass

    def _rpc(self, addr: str, method: str, data: dict, timeout: float = 2.0) -> Optional[dict]:
        host, port_str = addr.split(":")
        port = int(port_str) + 1  # RPC port = HTTP port + 1
        try:
            sock = socket.create_connection((host, port), timeout=timeout)
            payload = json.dumps({"method": method, "data": data}).encode()
            sock.sendall(struct.pack(">I", len(payload)) + payload)
            length_data = sock.recv(4)
            if len(length_data) < 4:
                return None
            length = struct.unpack(">I", length_data)[0]
            response = b""
            while len(response) < length:
                response += sock.recv(length - len(response))
            sock.close()
            return json.loads(response)
        except Exception:
            return None

    def run(self):
        """主循环"""
        print(f"[Cluster] 节点 {self.node_id} 启动 (peers={len(self.peers)})")

        # 启动 RPC 服务
        rpc_thread = threading.Thread(target=self._rpc_server, daemon=True)
        rpc_thread.start()

        last_heartbeat_send = 0

        while not self._stop.is_set():
            now = time.time()

            if self.role == "leader":
                if now - last_heartbeat_send > 0.5:
                    for peer_id, addr in self._peers_http.items():
                        self._send_heartbeat(peer_id, addr)
                    last_heartbeat_send = now

            elif now - self.election_timer > self.election_timeout:
                self.election_timeout = random.uniform(3.0, 6.0)
                self.start_election()

            time.sleep(0.1)

    def _rpc_server(self):
        _, port_str = self._peers_http.get(self.node_id, f"127.0.0.1:55560").split(":")
        rpc_port = int(port_str) + 1

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", rpc_port))
        sock.listen(10)
        sock.settimeout(1.0)

        while not self._stop.is_set():
            try:
                conn, _ = sock.accept()
                threading.Thread(target=self._handle_rpc, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
            except Exception:
                break

    def _handle_rpc(self, conn):
        try:
            length_data = conn.recv(4)
            if len(length_data) < 4:
                return
            length = struct.unpack(">I", length_data)[0]
            data = b""
            while len(data) < length:
                chunk = conn.recv(length - len(data))
                if not chunk:
                    return
                data += chunk

            request = json.loads(data)
            method_name = request["method"]
            args = request["data"]

            method_map = {
                "request_vote": lambda: self.request_vote(
                    args["candidate_id"], args["term"],
                    args["last_log_index"], args["last_log_term"]),
                "append_entries": lambda: self.append_entries(
                    args["leader_id"], args["term"], args["prev_log_index"],
                    args["prev_log_term"], args["entries"], args["leader_commit"]),
            }

            result = method_map.get(method_name, lambda: {"error": "unknown method"})()
            response = json.dumps(result).encode()
            conn.sendall(struct.pack(">I", len(response)) + response)
        except Exception:
            pass
        finally:
            conn.close()

    def status(self) -> dict:
        return {
            "node_id": self.node_id,
            "role": self.role,
            "term": self.current_term,
            "leader": self.leader_id,
            "commit_index": self.commit_index,
            "log_length": len(self.log),
            "peers": self.peers,
            "state_keys": list(self.state.keys()),
        }

    def stop(self):
        self._stop.set()


# ============================================================================
# 集群管理器
# ============================================================================

class ClusterManager:
    """集群管理器 — 多节点协调"""

    def __init__(self, node_id: str = None, bind_host: str = "127.0.0.1",
                 bind_port: int = 55560, seed_peers: List[str] = None):
        self.node_id = node_id or f"lifers-{secrets.token_hex(6)}"
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.peers = seed_peers or [f"{self.node_id}@{bind_host}:{bind_port}"]
        self.raft: Optional[RaftNode] = None
        self._thread: Optional[threading.Thread] = None

    def start(self, join_existing: bool = False):
        self.raft = RaftNode(self.node_id, self.peers)
        self.raft.register_handler("set_state", lambda d: self.raft.state.update(d))
        self.raft.register_handler("sync_weights", self._on_sync_weights)
        self._thread = threading.Thread(target=self.raft.run, daemon=True)
        self._thread.start()
        return self

    def _on_sync_weights(self, data: dict):
        weights_dir = ROOT / "weights"
        for name, content_b64 in data.get("weights", {}).items():
            import base64
            (weights_dir / name).write_bytes(base64.b64decode(content_b64))

    def propose_state(self, key: str, value: Any):
        if self.raft and self.raft.role == "leader":
            self.raft.propose("set_state", {key: value})

    def get_state(self, key: str) -> Optional[Any]:
        if self.raft:
            return self.raft.state.get(key)
        return None

    def status(self) -> dict:
        if self.raft:
            return self.raft.status()
        return {"mode": "standalone", "node_id": self.node_id}

    def stop(self):
        if self.raft:
            self.raft.stop()


# ============================================================================
# 全局实例
# ============================================================================

import secrets

_cluster_instance: Optional[ClusterManager] = None


def get_cluster() -> ClusterManager:
    global _cluster_instance
    if _cluster_instance is None:
        _cluster_instance = ClusterManager()
    return _cluster_instance


def initialize_cluster(node_id: str, peers: List[str], bind_host: str = "127.0.0.1",
                       bind_port: int = 55560) -> ClusterManager:
    global _cluster_instance
    _cluster_instance = ClusterManager(node_id, bind_host, bind_port, peers)
    _cluster_instance.start()
    return _cluster_instance


# ============================================================================
# 脑裂防护
# ============================================================================

class FencingToken:
    """隔离令牌 — 防止脑裂时的陈旧写入"""

    def __init__(self):
        self._token = 0
        self._lock = threading.Lock()

    def increment(self) -> int:
        with self._lock:
            self._token += 1
            return self._token

    def validate(self, token: int) -> bool:
        with self._lock:
            return token == self._token


# ============================================================================
# 集群 HTTP 端点
# ============================================================================

def cluster_http_handler(method: str, path: str, auth: dict, body: dict = None) -> dict:
    cluster = get_cluster()

    if path == "/v1/cluster/status":
        return cluster.status()

    if path == "/v1/cluster/nodes":
        return {"nodes": [cluster.status()]}

    if path == "/v1/cluster/state":
        if method == "POST" and body:
            key = body.get("key")
            value = body.get("value")
            cluster.propose_state(key, value)
            return {"status": "ok"}
        key = body.get("key") if body else None
        if key:
            val = cluster.get_state(key)
            return {"key": key, "value": val}
        return {"state": cluster.raft.state if cluster.raft else {}}

    return {"error": "Not found"}
