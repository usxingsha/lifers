"""
Lifers Tenant v1 — 多租户隔离系统
命名空间隔离、数据分区、配额管理、租户级加密
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

ROOT = Path(__file__).resolve().parent.parent
TENANT_DB = ROOT / "memory" / "tenants.sqlite3"


def _init_tenant_db():
    TENANT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(TENANT_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tenants (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            display_name TEXT,
            owner_id TEXT NOT NULL,
            encryption_key_hash TEXT,
            quota_disk_mb INTEGER DEFAULT 1024,
            quota_memory_entries INTEGER DEFAULT 100000,
            quota_robots INTEGER DEFAULT 10,
            quota_api_calls_per_min INTEGER DEFAULT 100,
            is_active INTEGER DEFAULT 1,
            created_at REAL NOT NULL,
            metadata TEXT DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS tenant_members (
            tenant_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            role TEXT DEFAULT 'member',
            joined_at REAL NOT NULL,
            PRIMARY KEY(tenant_id, user_id),
            FOREIGN KEY(tenant_id) REFERENCES tenants(id)
        );
        CREATE TABLE IF NOT EXISTS tenant_namespaces (
            tenant_id TEXT NOT NULL,
            namespace TEXT NOT NULL,
            data_path TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY(tenant_id, namespace),
            FOREIGN KEY(tenant_id) REFERENCES tenants(id)
        );
        CREATE TABLE IF NOT EXISTS usage_log (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            metric TEXT NOT NULL,
            value REAL NOT NULL,
            recorded_at REAL NOT NULL,
            FOREIGN KEY(tenant_id) REFERENCES tenants(id)
        );
        CREATE INDEX IF NOT EXISTS idx_usage_tenant ON usage_log(tenant_id, metric);
        CREATE INDEX IF NOT EXISTS idx_members_user ON tenant_members(user_id);
    """)
    conn.commit()
    return conn


_init_tenant_db()


class TenantManager:
    """多租户管理器"""

    def __init__(self):
        pass

    def _conn(self):
        conn = sqlite3.connect(str(TENANT_DB))
        conn.row_factory = sqlite3.Row
        return conn

    def create_tenant(self, name: str, owner_id: str, display_name: str = "",
                      quota_disk_mb: int = 1024, quota_robots: int = 10) -> Optional[str]:
        tid = f"tn_{secrets.token_hex(10)}"
        enc_key = secrets.token_hex(32)
        enc_key_hash = __import__('hashlib').sha256(enc_key.encode()).hexdigest()
        now = time.time()
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO tenants(id, name, display_name, owner_id, encryption_key_hash, "
                "quota_disk_mb, quota_robots, created_at) VALUES(?,?,?,?,?,?,?,?)",
                (tid, name.lower(), display_name or name, owner_id, enc_key_hash,
                 quota_disk_mb, quota_robots, now))
            conn.commit()
            self._create_tenant_storage(tid)
            return tid
        except sqlite3.IntegrityError:
            return None
        finally:
            conn.close()

    def _create_tenant_storage(self, tid: str):
        base = ROOT / "tenants" / tid
        for sub in ["memory", "sessions", "models", "logs", "data"]:
            (base / sub).mkdir(parents=True, exist_ok=True)
        conn = self._conn()
        for ns in ["memory", "sessions", "models", "logs", "data"]:
            conn.execute(
                "INSERT OR IGNORE INTO tenant_namespaces(tenant_id, namespace, data_path, created_at) "
                "VALUES(?,?,?,?)",
                (tid, ns, str(base / ns), time.time()))
        conn.commit()
        conn.close()

    def get_tenant(self, tid: str) -> Optional[dict]:
        conn = self._conn()
        row = conn.execute("SELECT * FROM tenants WHERE id=? AND is_active=1", (tid,)).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_tenant_by_name(self, name: str) -> Optional[dict]:
        conn = self._conn()
        row = conn.execute("SELECT * FROM tenants WHERE name=? AND is_active=1", (name.lower(),)).fetchone()
        conn.close()
        return dict(row) if row else None

    def add_member(self, tenant_id: str, user_id: str, role: str = "member") -> bool:
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO tenant_members(tenant_id, user_id, role, joined_at) VALUES(?,?,?,?)",
                (tenant_id, user_id, role, time.time()))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def get_user_tenants(self, user_id: str) -> list:
        conn = self._conn()
        rows = conn.execute(
            "SELECT t.*, tm.role as member_role FROM tenants t "
            "JOIN tenant_members tm ON t.id = tm.tenant_id "
            "WHERE tm.user_id=? AND t.is_active=1", (user_id,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_tenant_path(self, tenant_id: str, namespace: str) -> Optional[Path]:
        conn = self._conn()
        row = conn.execute(
            "SELECT data_path FROM tenant_namespaces WHERE tenant_id=? AND namespace=?",
            (tenant_id, namespace)).fetchone()
        conn.close()
        return Path(row["data_path"]) if row else None

    def check_quota(self, tenant_id: str, metric: str) -> dict:
        conn = self._conn()
        tenant = conn.execute("SELECT * FROM tenants WHERE id=?", (tenant_id,)).fetchone()
        if not tenant:
            conn.close()
            return {"allowed": False, "reason": "tenant not found"}

        # 统计最近1分钟 API 调用
        if metric == "api_calls":
            one_min_ago = time.time() - 60
            count = conn.execute(
                "SELECT COUNT(*) FROM usage_log WHERE tenant_id=? AND metric='api_call' "
                "AND recorded_at > ?", (tenant_id, one_min_ago)).fetchone()[0]
            limit = tenant["quota_api_calls_per_min"]
            conn.close()
            return {"allowed": count < limit, "current": count, "limit": limit}

        # 统计内存条目
        if metric == "memory_entries":
            tenant_path = ROOT / "tenants" / tenant_id / "memory"
            count = 0
            if tenant_path.exists():
                mem_db = tenant_path / "memory.sqlite3"
                if mem_db.exists():
                    mc = sqlite3.connect(str(mem_db))
                    count = mc.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                    mc.close()
            limit = tenant["quota_memory_entries"]
            conn.close()
            return {"allowed": count < limit, "current": count, "limit": limit}

        # 磁盘使用
        if metric == "disk":
            tenant_path = ROOT / "tenants" / tenant_id
            total_bytes = 0
            if tenant_path.exists():
                for f in tenant_path.rglob("*"):
                    if f.is_file():
                        total_bytes += f.stat().st_size
            limit_bytes = tenant["quota_disk_mb"] * 1024 * 1024
            conn.close()
            return {"allowed": total_bytes < limit_bytes,
                    "current_mb": total_bytes / (1024 * 1024),
                    "limit_mb": tenant["quota_disk_mb"]}

        conn.close()
        return {"allowed": True}

    def log_usage(self, tenant_id: str, metric: str, value: float = 1.0):
        conn = self._conn()
        conn.execute(
            "INSERT INTO usage_log(id, tenant_id, metric, value, recorded_at) VALUES(?,?,?,?,?)",
            (f"ul_{secrets.token_hex(8)}", tenant_id, metric, value, time.time()))
        conn.commit()
        conn.close()

    def deactivate_tenant(self, tenant_id: str):
        conn = self._conn()
        conn.execute("UPDATE tenants SET is_active=0 WHERE id=?", (tenant_id,))
        conn.commit()
        conn.close()

    def list_all_tenants(self) -> list:
        conn = self._conn()
        rows = conn.execute("SELECT * FROM tenants WHERE is_active=1").fetchall()
        conn.close()
        return [dict(r) for r in rows]


# ============================================================================
# 全局实例
# ============================================================================

_tenant_instance: Optional[TenantManager] = None


def get_tenant_manager() -> TenantManager:
    global _tenant_instance
    if _tenant_instance is None:
        _tenant_instance = TenantManager()
    return _tenant_instance


# ============================================================================
# 租户隔离的 HTTP 中间件
# ============================================================================

def tenant_middleware(auth_payload: dict) -> Optional[str]:
    """从认证载荷提取租户 ID。若无租户则返回空字符串（个人空间）"""
    return auth_payload.get("tenant_id", "")


def scoped_query(tenant_id: str, base_path: str) -> Path:
    """返回租户隔离的文件路径"""
    if tenant_id:
        return ROOT / "tenants" / tenant_id / base_path
    return ROOT / base_path


def isolate_memory_db(tenant_id: str) -> Path:
    if tenant_id:
        p = ROOT / "tenants" / tenant_id / "memory" / "memory.sqlite3"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    return ROOT / "memory" / "longterm.sqlite3"
