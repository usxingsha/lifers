"""
Lifers Fleet v1 — 机器人舰队管理
发现/注册/心跳/OTA升级/任务分发/健康监控
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
FLEET_DB = ROOT / "memory" / "fleet.sqlite3"


def _init_fleet_db():
    FLEET_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(FLEET_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS robots (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            robot_type TEXT DEFAULT 'generic',
            model TEXT DEFAULT '',
            serial_number TEXT,
            firmware_version TEXT,
            ip_address TEXT,
            mac_address TEXT,
            protocol TEXT DEFAULT 'mqtt',
            tenant_id TEXT,
            status TEXT DEFAULT 'offline',
            battery_pct REAL DEFAULT 100.0,
            cpu_pct REAL DEFAULT 0,
            ram_pct REAL DEFAULT 0,
            location TEXT DEFAULT '',
            registered_at REAL NOT NULL,
            last_heartbeat REAL,
            metadata TEXT DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS robot_tasks (
            id TEXT PRIMARY KEY,
            robot_id TEXT NOT NULL,
            task_type TEXT NOT NULL,
            priority INTEGER DEFAULT 5,
            payload TEXT DEFAULT '{}',
            status TEXT DEFAULT 'pending',
            created_at REAL NOT NULL,
            assigned_at REAL,
            completed_at REAL,
            result TEXT,
            error_message TEXT,
            FOREIGN KEY (robot_id) REFERENCES robots(id)
        );
        CREATE TABLE IF NOT EXISTS ota_packages (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            version TEXT NOT NULL,
            target_type TEXT DEFAULT 'generic',
            file_path TEXT NOT NULL,
            checksum TEXT NOT NULL,
            size_bytes INTEGER DEFAULT 0,
            release_notes TEXT DEFAULT '',
            created_at REAL NOT NULL,
            deployed_count INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS fleet_heartbeats (
            robot_id TEXT NOT NULL,
            received_at REAL NOT NULL,
            status TEXT DEFAULT 'ok',
            metrics TEXT DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS fleet_events (
            id TEXT PRIMARY KEY,
            robot_id TEXT,
            event_type TEXT NOT NULL,
            severity TEXT DEFAULT 'info',
            message TEXT NOT NULL,
            recorded_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_robot_status ON robots(status);
        CREATE INDEX IF NOT EXISTS idx_robot_tenant ON robots(tenant_id);
        CREATE INDEX IF NOT EXISTS idx_tasks_robot ON robot_tasks(robot_id);
        CREATE INDEX IF NOT EXISTS idx_tasks_status ON robot_tasks(status);
        CREATE INDEX IF NOT EXISTS idx_heartbeats_robot ON fleet_heartbeats(robot_id);
    """)
    conn.commit()
    return conn


_init_fleet_db()


# ============================================================================
# 机器人注册与发现
# ============================================================================

@dataclass
class Robot:
    id: str
    name: str
    robot_type: str
    model: str
    serial_number: str
    firmware_version: str
    ip_address: str
    protocol: str
    tenant_id: str
    status: str
    battery_pct: float
    location: str
    registered_at: float
    last_heartbeat: Optional[float]


class FleetManager:
    """机器人舰队管理器"""

    def __init__(self):
        self._lock = threading.Lock()
        self._task_handlers: Dict[str, Callable] = {}
        self._discovery_callbacks: List[Callable] = []

    def _conn(self):
        conn = sqlite3.connect(str(FLEET_DB))
        conn.row_factory = sqlite3.Row
        return conn

    # ---- 注册与发现 ----

    def register_robot(self, name: str, robot_type: str = "generic",
                       model: str = "", serial: str = "", firmware: str = "",
                       ip: str = "", mac: str = "", protocol: str = "mqtt",
                       tenant_id: str = "", metadata: dict = None) -> Optional[str]:
        rid = f"rb_{secrets.token_hex(10)}"
        now = time.time()
        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT INTO robots(id, name, robot_type, model, serial_number, "
                    "firmware_version, ip_address, mac_address, protocol, tenant_id, "
                    "status, registered_at, metadata) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (rid, name, robot_type, model, serial, firmware, ip, mac,
                     protocol, tenant_id, "online", now, json.dumps(metadata or {})))
                conn.commit()
                self._event("robot_registered", rid, f"Robot {name} registered")
                return rid
            except sqlite3.IntegrityError:
                return None
            finally:
                conn.close()

    def discover_robots(self, tenant_id: str = "") -> List[Robot]:
        conn = self._conn()
        try:
            query = "SELECT * FROM robots WHERE 1=1"
            params = []
            if tenant_id:
                query += " AND tenant_id=?"
                params.append(tenant_id)
            rows = conn.execute(query, params).fetchall()
            return [Robot(
                id=r["id"], name=r["name"], robot_type=r["robot_type"],
                model=r["model"], serial_number=r["serial_number"] or "",
                firmware_version=r["firmware_version"] or "", ip_address=r["ip_address"] or "",
                protocol=r["protocol"], tenant_id=r["tenant_id"] or "",
                status=r["status"], battery_pct=r["battery_pct"] or 100.0,
                location=r["location"] or "", registered_at=r["registered_at"],
                last_heartbeat=r["last_heartbeat"],
            ) for r in rows]
        finally:
            conn.close()

    def get_robot(self, robot_id: str) -> Optional[Robot]:
        conn = self._conn()
        try:
            r = conn.execute("SELECT * FROM robots WHERE id=?", (robot_id,)).fetchone()
            if not r:
                return None
            return Robot(
                id=r["id"], name=r["name"], robot_type=r["robot_type"],
                model=r["model"], serial_number=r["serial_number"] or "",
                firmware_version=r["firmware_version"] or "", ip_address=r["ip_address"] or "",
                protocol=r["protocol"], tenant_id=r["tenant_id"] or "",
                status=r["status"], battery_pct=r["battery_pct"] or 100.0,
                location=r["location"] or "", registered_at=r["registered_at"],
                last_heartbeat=r["last_heartbeat"],
            )
        finally:
            conn.close()

    def on_discovery(self, callback: Callable):
        self._discovery_callbacks.append(callback)

    # ---- 心跳监控 ----

    def heartbeat(self, robot_id: str, status: str = "ok", metrics: dict = None) -> bool:
        now = time.time()
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO fleet_heartbeats(robot_id, received_at, status, metrics) "
                "VALUES(?,?,?,?)", (robot_id, now, status, json.dumps(metrics or {})))
            conn.execute(
                "UPDATE robots SET last_heartbeat=?, status='online', "
                "battery_pct=COALESCE(?, battery_pct), cpu_pct=COALESCE(?, cpu_pct), "
                "ram_pct=COALESCE(?, ram_pct) WHERE id=?",
                (now, metrics and metrics.get("battery"), metrics and metrics.get("cpu"),
                 metrics and metrics.get("ram"), robot_id))
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            conn.close()

    def check_offline_robots(self, timeout_seconds: int = 30) -> List[str]:
        threshold = time.time() - timeout_seconds
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT id, name FROM robots WHERE status='online' AND "
                "(last_heartbeat IS NULL OR last_heartbeat < ?)", (threshold,)
            ).fetchall()
            for r in rows:
                conn.execute("UPDATE robots SET status='offline' WHERE id=?", (r["id"],))
                self._event("robot_offline", r["id"], f"Robot {r['name']} went offline")
            conn.commit()
            return [r["id"] for r in rows]
        finally:
            conn.close()

    # ---- 任务分发 ----

    def create_task(self, robot_id: str, task_type: str, payload: dict = None,
                    priority: int = 5) -> Optional[str]:
        tid = f"tsk_{secrets.token_hex(8)}"
        now = time.time()
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO robot_tasks(id, robot_id, task_type, priority, payload, "
                "status, created_at) VALUES(?,?,?,?,?,?,?)",
                (tid, robot_id, task_type, priority, json.dumps(payload or {}),
                 "pending", now))
            conn.commit()
            return tid
        except Exception:
            return None
        finally:
            conn.close()

    def get_pending_tasks(self, robot_id: str, limit: int = 10) -> List[dict]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT * FROM robot_tasks WHERE robot_id=? AND status='pending' "
                "ORDER BY priority ASC, created_at ASC LIMIT ?",
                (robot_id, limit)).fetchall()
            now = time.time()
            for r in rows:
                conn.execute(
                    "UPDATE robot_tasks SET status='assigned', assigned_at=? WHERE id=?",
                    (now, r["id"]))
            conn.commit()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def complete_task(self, task_id: str, success: bool = True, result: dict = None,
                      error: str = ""):
        now = time.time()
        conn = self._conn()
        conn.execute(
            "UPDATE robot_tasks SET status=?, completed_at=?, result=?, error_message=? "
            "WHERE id=?",
            ("completed" if success else "failed", now, json.dumps(result or {}),
             error, task_id))
        conn.commit()
        conn.close()

    def get_robot_tasks(self, robot_id: str, status: str = None) -> List[dict]:
        conn = self._conn()
        try:
            query = "SELECT * FROM robot_tasks WHERE robot_id=?"
            params = [robot_id]
            if status:
                query += " AND status=?"
                params.append(status)
            query += " ORDER BY created_at DESC LIMIT 100"
            return [dict(r) for r in conn.execute(query, params).fetchall()]
        finally:
            conn.close()

    def cancel_task(self, task_id: str) -> bool:
        conn = self._conn()
        conn.execute(
            "UPDATE robot_tasks SET status='cancelled' WHERE id=? AND status='pending'",
            (task_id,))
        conn.commit()
        conn.close()
        return True

    # ---- OTA 升级 ----

    def create_ota_package(self, name: str, version: str, file_path: Path,
                           target_type: str = "generic", release_notes: str = "") -> Optional[str]:
        oid = f"ota_{secrets.token_hex(8)}"
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        checksum = h.hexdigest()
        size = file_path.stat().st_size
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO ota_packages(id, name, version, target_type, file_path, "
                "checksum, size_bytes, release_notes, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (oid, name, version, target_type, str(file_path), checksum, size,
                 release_notes, time.time()))
            conn.commit()
            return oid
        except Exception:
            return None
        finally:
            conn.close()

    def get_latest_ota(self, robot_type: str) -> Optional[dict]:
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM ota_packages WHERE target_type=? OR target_type='generic' "
            "ORDER BY created_at DESC LIMIT 1", (robot_type,)).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_ota_packages(self) -> List[dict]:
        conn = self._conn()
        rows = conn.execute("SELECT * FROM ota_packages ORDER BY created_at DESC").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def deploy_ota(self, ota_id: str, robot_ids: List[str]) -> int:
        conn = self._conn()
        ota = conn.execute("SELECT * FROM ota_packages WHERE id=?", (ota_id,)).fetchone()
        if not ota:
            conn.close()
            return 0
        count = 0
        for rid in robot_ids:
            task_id = f"tsk_{secrets.token_hex(8)}"
            conn.execute(
                "INSERT INTO robot_tasks(id, robot_id, task_type, priority, payload, "
                "status, created_at) VALUES(?,?,?,?,?,?,?)",
                (task_id, rid, "ota_update", 1,
                 json.dumps({"ota_id": ota_id, "version": ota["version"],
                            "file": ota["file_path"], "checksum": ota["checksum"]}),
                 "pending", time.time()))
            count += 1
        conn.execute("UPDATE ota_packages SET deployed_count=deployed_count+? WHERE id=?", (count, ota_id))
        conn.commit()
        conn.close()
        return count

    # ---- 健康监控 ----

    def fleet_health_summary(self, tenant_id: str = "") -> dict:
        conn = self._conn()
        try:
            query = "SELECT status, COUNT(*) as cnt FROM robots WHERE 1=1"
            params = []
            if tenant_id:
                query += " AND tenant_id=?"
                params.append(tenant_id)
            query += " GROUP BY status"
            rows = conn.execute(query, params).fetchall()

            total = sum(r["cnt"] for r in rows)
            online = sum(r["cnt"] for r in rows if r["status"] == "online")
            avg_battery = 0
            if tenant_id:
                b_rows = conn.execute(
                    "SELECT AVG(battery_pct) FROM robots WHERE tenant_id=? AND status='online'",
                    (tenant_id,)).fetchone()
            else:
                b_rows = conn.execute(
                    "SELECT AVG(battery_pct) FROM robots WHERE status='online'").fetchone()
            avg_battery = b_rows[0] if b_rows and b_rows[0] else 0

            pending_tasks = conn.execute(
                "SELECT COUNT(*) FROM robot_tasks WHERE status='pending'").fetchone()[0]

            return {
                "total": total, "online": online, "offline": total - online,
                "avg_battery_pct": round(avg_battery, 1),
                "pending_tasks": pending_tasks,
                "by_status": {r["status"]: r["cnt"] for r in rows},
            }
        finally:
            conn.close()

    def get_recent_events(self, limit: int = 50, severity: str = None) -> List[dict]:
        conn = self._conn()
        try:
            query = "SELECT * FROM fleet_events"
            params = []
            if severity:
                query += " WHERE severity=?"
                params.append(severity)
            query += " ORDER BY recorded_at DESC LIMIT ?"
            params.append(limit)
            return [dict(r) for r in conn.execute(query, params).fetchall()]
        finally:
            conn.close()

    def _event(self, event_type: str, robot_id: str, message: str, severity: str = "info"):
        conn = self._conn()
        conn.execute(
            "INSERT INTO fleet_events(id, robot_id, event_type, severity, message, recorded_at) "
            "VALUES(?,?,?,?,?,?)",
            (f"ev_{secrets.token_hex(8)}", robot_id, event_type, severity, message, time.time()))
        conn.commit()
        conn.close()

    # ---- 批量操作 ----

    def broadcast_command(self, command: dict, robot_type: str = None,
                          tenant_id: str = "") -> int:
        conn = self._conn()
        try:
            query = "SELECT id FROM robots WHERE status='online'"
            params = []
            if robot_type:
                query += " AND robot_type=?"
                params.append(robot_type)
            if tenant_id:
                query += " AND tenant_id=?"
                params.append(tenant_id)
            rows = conn.execute(query, params).fetchall()

            count = 0
            for r in rows:
                self.create_task(r["id"], "command", command, priority=8)
                count += 1
            return count
        finally:
            conn.close()

    def shutdown_robot(self, robot_id: str) -> bool:
        task_id = self.create_task(robot_id, "shutdown", priority=1)
        return task_id is not None

    def restart_robot(self, robot_id: str) -> bool:
        task_id = self.create_task(robot_id, "restart", priority=1)
        return task_id is not None

    def emergency_stop_all(self, tenant_id: str = "") -> int:
        return self.broadcast_command({"action": "emergency_stop"}, tenant_id=tenant_id)


# ============================================================================
# 全局实例
# ============================================================================

_fleet_instance: Optional[FleetManager] = None


def get_fleet() -> FleetManager:
    global _fleet_instance
    if _fleet_instance is None:
        _fleet_instance = FleetManager()
    return _fleet_instance


# ============================================================================
# 舰队 HTTP 端点
# ============================================================================

def fleet_http_handler(method: str, path: str, auth: dict, body: dict = None) -> dict:
    fleet = get_fleet()
    body = body or {}

    # GET /v1/fleet/robots
    if method == "GET" and path == "/v1/fleet/robots":
        tenant = auth.get("tenant_id", "")
        robots = fleet.discover_robots(tenant)
        return {"robots": [{
            "id": r.id, "name": r.name, "type": r.robot_type,
            "status": r.status, "battery": r.battery_pct,
            "location": r.location, "last_heartbeat": r.last_heartbeat,
        } for r in robots]}

    # GET /v1/fleet/robots/{id}
    if method == "GET" and path.startswith("/v1/fleet/robots/"):
        robot_id = path.split("/")[-1]
        robot = fleet.get_robot(robot_id)
        if not robot:
            return {"error": "Not found"}, 404
        return {
            "id": robot.id, "name": robot.name, "type": robot.robot_type,
            "model": robot.model, "firmware": robot.firmware_version,
            "status": robot.status, "battery": robot.battery_pct,
            "ip": robot.ip_address, "protocol": robot.protocol,
            "location": robot.location,
        }

    # POST /v1/fleet/robots
    if method == "POST" and path == "/v1/fleet/robots":
        robot_id = fleet.register_robot(
            name=body.get("name", ""), robot_type=body.get("type", "generic"),
            model=body.get("model", ""), serial=body.get("serial", ""),
            firmware=body.get("firmware", ""), ip=body.get("ip", ""),
            protocol=body.get("protocol", "mqtt"),
            tenant_id=auth.get("tenant_id", ""), metadata=body.get("metadata"))
        if robot_id:
            return {"status": "ok", "robot_id": robot_id}
        return {"status": "error", "message": "Registration failed"}, 400

    # POST /v1/fleet/heartbeat
    if method == "POST" and path == "/v1/fleet/heartbeat":
        robot_id = body.get("robot_id", "")
        metrics = body.get("metrics", {})
        ok = fleet.heartbeat(robot_id, body.get("status", "ok"), metrics)
        return {"status": "ok" if ok else "error"}

    # GET /v1/fleet/tasks
    if method == "GET" and path == "/v1/fleet/tasks":
        robot_id = body.get("robot_id", "")
        tasks = fleet.get_robot_tasks(robot_id)
        return {"tasks": tasks}

    # POST /v1/fleet/tasks
    if method == "POST" and path == "/v1/fleet/tasks":
        task_id = fleet.create_task(
            body.get("robot_id", ""), body.get("task_type", ""),
            body.get("payload", {}), body.get("priority", 5))
        if task_id:
            return {"status": "ok", "task_id": task_id}
        return {"status": "error"}, 400

    # POST /v1/fleet/tasks/{id}/complete
    if method == "POST" and path.startswith("/v1/fleet/tasks/") and path.endswith("/complete"):
        task_id = path.split("/")[-2]
        fleet.complete_task(task_id, body.get("success", True),
                           body.get("result"), body.get("error", ""))
        return {"status": "ok"}

    # GET /v1/fleet/health
    if method == "GET" and path == "/v1/fleet/health":
        tenant = auth.get("tenant_id", "")
        return fleet.fleet_health_summary(tenant)

    # GET /v1/fleet/ota
    if method == "GET" and path == "/v1/fleet/ota":
        return {"packages": fleet.get_ota_packages()}

    # POST /v1/fleet/ota
    if method == "POST" and path == "/v1/fleet/ota":
        ota_id = fleet.create_ota_package(
            body.get("name", ""), body.get("version", ""),
            Path(body.get("file_path", "")), body.get("target_type", "generic"),
            body.get("release_notes", ""))
        if ota_id:
            return {"status": "ok", "ota_id": ota_id}
        return {"status": "error"}, 400

    # POST /v1/fleet/emergency-stop
    if method == "POST" and path == "/v1/fleet/emergency-stop":
        count = fleet.emergency_stop_all(auth.get("tenant_id", ""))
        return {"status": "ok", "stopped": count}

    return {"error": "Not found"}, 404
