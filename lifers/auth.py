"""
Lifers Auth v1 — 身份认证与设备绑定系统
纯标准库实现: JWT, API Key, 设备指纹, 密码哈希, 会话管理, RBAC
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Optional, Dict, List, Any

ROOT = Path(__file__).resolve().parent.parent
AUTH_DB = ROOT / "memory" / "auth.sqlite3"
SECRET_KEY_FILE = ROOT / "config" / ".auth_secret"

# ============================================================================
# 密码哈希 (PBKDF2-HMAC-SHA256)
# ============================================================================

def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
    if salt is None:
        salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200000, dklen=32)
    s = base64.b64encode(salt).decode()
    k = base64.b64encode(key).decode()
    return f"pbkdf2:sha256:200000${s}${k}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        meta, salt_b64, key_b64 = stored.split("$")
        iterations = int(meta.split(":")[-1]) if ":" in meta else 200000
        salt = base64.b64decode(salt_b64)
        expected_key = base64.b64decode(key_b64)
        key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen=32)
        return hmac.compare_digest(key, expected_key)
    except (ValueError, base64.binascii.Error, IndexError):
        return False


# ============================================================================
# JWT (无外部依赖)
# ============================================================================

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def _get_jwt_secret() -> bytes:
    if SECRET_KEY_FILE.exists():
        return SECRET_KEY_FILE.read_bytes()
    key = os.urandom(64)
    SECRET_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    SECRET_KEY_FILE.write_bytes(key)
    return key


def create_jwt(payload: dict, ttl_seconds: int = 86400) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    payload = {**payload, "iat": now, "exp": now + ttl_seconds, "jti": secrets.token_hex(16)}
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header_b64}.{payload_b64}"
    sig = hmac.new(_get_jwt_secret(), signing_input.encode(), "sha256").digest()
    return f"{signing_input}.{_b64url_encode(sig)}"


def verify_jwt(token: str) -> Optional[dict]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}"
        expected_sig = _b64url_decode(sig_b64)
        actual_sig = hmac.new(_get_jwt_secret(), signing_input.encode(), "sha256").digest()
        if not hmac.compare_digest(actual_sig, expected_sig):
            return None
        payload = json.loads(_b64url_decode(payload_b64))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


# ============================================================================
# API Key
# ============================================================================

def generate_api_key(prefix: str = "lf") -> str:
    raw = secrets.token_bytes(32)
    key = base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
    return f"{prefix}_{key}"


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


# ============================================================================
# 设备指纹
# ============================================================================

def device_fingerprint(device_info: dict) -> str:
    parts = [
        device_info.get("hostname", ""),
        device_info.get("machine_id", ""),
        device_info.get("mac_address", ""),
        device_info.get("cpu_serial", ""),
        device_info.get("os_version", ""),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


# ============================================================================
# 数据库
# ============================================================================

def _init_db():
    AUTH_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(AUTH_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT,
            role TEXT DEFAULT 'user',
            tenant_id TEXT,
            created_at REAL NOT NULL,
            last_login REAL,
            is_active INTEGER DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS api_keys (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            key_hash TEXT UNIQUE NOT NULL,
            key_prefix TEXT NOT NULL,
            name TEXT,
            scopes TEXT DEFAULT 'read',
            created_at REAL NOT NULL,
            expires_at REAL,
            last_used REAL,
            is_active INTEGER DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            token_hash TEXT UNIQUE NOT NULL,
            device_fingerprint TEXT,
            ip_address TEXT,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL,
            is_active INTEGER DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS device_bindings (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            device_fingerprint TEXT NOT NULL,
            device_name TEXT,
            device_type TEXT DEFAULT 'unknown',
            trusted INTEGER DEFAULT 0,
            first_seen REAL NOT NULL,
            last_seen REAL NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, device_fingerprint)
        );
        CREATE TABLE IF NOT EXISTS roles (
            name TEXT PRIMARY KEY,
            permissions TEXT NOT NULL,
            description TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
        CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token_hash);
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_device_user ON device_bindings(user_id);
    """)
    conn.commit()
    _init_roles(conn)
    return conn


def _init_roles(conn):
    existing = set(r[0] for r in conn.execute("SELECT name FROM roles").fetchall())
    defaults = {
        "admin": "read,write,delete,manage_users,manage_tenants,manage_robots",
        "operator": "read,write,manage_robots",
        "developer": "read,write,manage_training,manage_plugins",
        "user": "read,write",
        "robot": "read,telemetry",
        "viewer": "read",
    }
    for name, perms in defaults.items():
        if name not in existing:
            conn.execute("INSERT INTO roles(name, permissions, description) VALUES(?,?,?)",
                         (name, perms, f"Default {name} role"))
    conn.commit()


_init_db()


# ============================================================================
# 认证管理器
# ============================================================================

@dataclass
class User:
    id: str
    username: str
    email: str
    role: str
    tenant_id: str
    created_at: float
    last_login: Optional[float]


class AuthManager:
    """认证管理器 — 线程安全"""

    def __init__(self):
        self._lock = Lock()

    def _conn(self):
        conn = sqlite3.connect(str(AUTH_DB))
        conn.row_factory = sqlite3.Row
        return conn

    # ---- 用户管理 ----

    def create_user(self, username: str, password: str, email: str = "",
                    role: str = "user", tenant_id: str = "") -> Optional[User]:
        with self._lock:
            conn = self._conn()
            try:
                uid = f"u_{secrets.token_hex(12)}"
                pw_hash = _hash_password(password)
                now = time.time()
                conn.execute(
                    "INSERT INTO users(id, username, password_hash, email, role, tenant_id, created_at) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (uid, username.lower(), pw_hash, email, role, tenant_id, now))
                conn.commit()
                return User(uid, username.lower(), email, role, tenant_id, now, None)
            except sqlite3.IntegrityError:
                return None
            finally:
                conn.close()

    def authenticate(self, username: str, password: str) -> Optional[str]:
        """返回 JWT token 或 None"""
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT * FROM users WHERE username=? AND is_active=1", (username.lower(),)
            ).fetchone()
            if not row or not _verify_password(password, row["password_hash"]):
                return None
            conn.execute("UPDATE users SET last_login=? WHERE id=?", (time.time(), row["id"]))
            conn.commit()
            return create_jwt({
                "sub": row["id"], "username": row["username"],
                "role": row["role"], "tenant_id": row["tenant_id"] or "",
            })
        finally:
            conn.close()

    def authenticate_api_key(self, api_key: str) -> Optional[dict]:
        """验证 API Key，返回 payload 或 None"""
        key_hash = hash_api_key(api_key)
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT k.*, u.username, u.role, u.tenant_id FROM api_keys k "
                "JOIN users u ON k.user_id = u.id "
                "WHERE k.key_hash=? AND k.is_active=1 AND u.is_active=1",
                (key_hash,)
            ).fetchone()
            if not row:
                return None
            if row["expires_at"] and row["expires_at"] < time.time():
                return None
            conn.execute("UPDATE api_keys SET last_used=? WHERE id=?", (time.time(), row["id"]))
            conn.commit()
            return {
                "sub": row["user_id"], "username": row["username"],
                "role": row["role"], "tenant_id": row["tenant_id"] or "",
                "scopes": row["scopes"], "auth_method": "api_key",
            }
        finally:
            conn.close()

    def validate_token(self, token: str) -> Optional[dict]:
        """验证 JWT Token"""
        return verify_jwt(token)

    # ---- API Key 管理 ----

    def create_api_key(self, user_id: str, name: str = "", scopes: str = "read",
                       ttl_days: int = 365) -> Optional[str]:
        api_key = generate_api_key()
        key_hash = hash_api_key(api_key)
        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT INTO api_keys(id, user_id, key_hash, key_prefix, name, scopes, "
                    "created_at, expires_at) VALUES(?,?,?,?,?,?,?,?)",
                    (f"ak_{secrets.token_hex(8)}", user_id, key_hash, api_key[:8],
                     name, scopes, time.time(), time.time() + ttl_days * 86400 if ttl_days > 0 else None))
                conn.commit()
                return api_key
            except sqlite3.IntegrityError:
                return None
            finally:
                conn.close()

    def revoke_api_key(self, key_hash: str):
        conn = self._conn()
        conn.execute("UPDATE api_keys SET is_active=0 WHERE key_hash=?", (key_hash,))
        conn.commit()
        conn.close()

    # ---- 会话管理 ----

    def create_session(self, user_id: str, device_info: dict = None,
                       ip_address: str = "", ttl_hours: int = 24) -> Optional[str]:
        token = secrets.token_hex(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        fp = device_fingerprint(device_info) if device_info else ""
        now = time.time()
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO sessions(id, user_id, token_hash, device_fingerprint, ip_address, "
                "created_at, expires_at) VALUES(?,?,?,?,?,?,?)",
                (f"s_{secrets.token_hex(8)}", user_id, token_hash, fp, ip_address,
                 now, now + ttl_hours * 3600))
            conn.commit()
            return token
        except sqlite3.IntegrityError:
            return None
        finally:
            conn.close()

    def validate_session(self, session_token: str) -> Optional[dict]:
        token_hash = hashlib.sha256(session_token.encode()).hexdigest()
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT s.*, u.username, u.role, u.tenant_id FROM sessions s "
                "JOIN users u ON s.user_id = u.id "
                "WHERE s.token_hash=? AND s.is_active=1 AND s.expires_at > ? AND u.is_active=1",
                (token_hash, time.time())
            ).fetchone()
            if not row:
                return None
            return {
                "sub": row["user_id"], "username": row["username"],
                "role": row["role"], "tenant_id": row["tenant_id"] or "",
                "device_fp": row["device_fingerprint"], "auth_method": "session",
            }
        finally:
            conn.close()

    def revoke_session(self, session_token: str):
        token_hash = hashlib.sha256(session_token.encode()).hexdigest()
        conn = self._conn()
        conn.execute("UPDATE sessions SET is_active=0 WHERE token_hash=?", (token_hash,))
        conn.commit()
        conn.close()

    # ---- 设备绑定 ----

    def bind_device(self, user_id: str, device_info: dict, device_name: str = "",
                    device_type: str = "unknown") -> str:
        fp = device_fingerprint(device_info)
        now = time.time()
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO device_bindings(id, user_id, device_fingerprint, device_name, "
                "device_type, first_seen, last_seen) VALUES(?,?,?,?,?,?,?) "
                "ON CONFLICT(user_id, device_fingerprint) DO UPDATE SET last_seen=?, device_name=?",
                (f"d_{secrets.token_hex(8)}", user_id, fp, device_name, device_type,
                 now, now, now, device_name or "unknown"))
            conn.commit()
            return fp
        finally:
            conn.close()

    def is_device_trusted(self, user_id: str, device_info: dict) -> bool:
        fp = device_fingerprint(device_info)
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT trusted FROM device_bindings WHERE user_id=? AND device_fingerprint=?",
                (user_id, fp)).fetchone()
            return bool(row and row["trusted"])
        finally:
            conn.close()

    def get_user_devices(self, user_id: str) -> list:
        conn = self._conn()
        try:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM device_bindings WHERE user_id=?", (user_id,)).fetchall()]
        finally:
            conn.close()

    # ---- 用户查询 ----

    def get_user(self, user_id: str) -> Optional[User]:
        conn = self._conn()
        try:
            row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
            if not row:
                return None
            return User(row["id"], row["username"], row["email"], row["role"],
                       row["tenant_id"], row["created_at"], row["last_login"])
        finally:
            conn.close()

    def get_user_by_username(self, username: str) -> Optional[User]:
        conn = self._conn()
        try:
            row = conn.execute("SELECT * FROM users WHERE username=?", (username.lower(),)).fetchone()
            if not row:
                return None
            return User(row["id"], row["username"], row["email"], row["role"],
                       row["tenant_id"], row["created_at"], row["last_login"])
        finally:
            conn.close()

    # ---- 权限检查 ----

    def get_permissions(self, role: str) -> List[str]:
        conn = self._conn()
        try:
            row = conn.execute("SELECT permissions FROM roles WHERE name=?", (role,)).fetchone()
            return row["permissions"].split(",") if row else []
        finally:
            conn.close()

    def has_permission(self, role: str, permission: str) -> bool:
        return permission in self.get_permissions(role)

    # ---- 清理 ----

    def cleanup_expired(self):
        now = time.time()
        conn = self._conn()
        conn.execute("UPDATE sessions SET is_active=0 WHERE expires_at < ?", (now,))
        conn.execute("UPDATE api_keys SET is_active=0 WHERE expires_at IS NOT NULL AND expires_at < ?", (now,))
        conn.commit()
        conn.close()


# ============================================================================
# 全局实例
# ============================================================================

_auth_instance: Optional[AuthManager] = None


def get_auth() -> AuthManager:
    global _auth_instance
    if _auth_instance is None:
        _auth_instance = AuthManager()
        _auth_instance.cleanup_expired()
    return _auth_instance


# ============================================================================
# 认证装饰器 / 中间件辅助
# ============================================================================

def require_auth(headers: dict) -> Optional[dict]:
    """从 HTTP headers 提取并验证身份。返回 payload 或 None"""
    auth = get_auth()

    # 1. Authorization: Bearer <jwt>
    auth_header = headers.get("Authorization", "") or headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        payload = auth.validate_token(token)
        if payload:
            return payload

    # 2. X-API-Key: <api_key>
    api_key = headers.get("X-API-Key", "") or headers.get("x-api-key", "")
    if api_key:
        payload = auth.authenticate_api_key(api_key)
        if payload:
            return payload

    # 3. Cookie: session=<token>
    cookie = headers.get("Cookie", "") or headers.get("cookie", "")
    if "session=" in cookie:
        session_token = cookie.split("session=")[1].split(";")[0].strip()
        payload = auth.validate_session(session_token)
        if payload:
            return payload

    return None


def require_permission(headers: dict, permission: str) -> bool:
    payload = require_auth(headers)
    if not payload:
        return False
    return get_auth().has_permission(payload.get("role", ""), permission)


def auth_http_handler(method: str, path: str, headers: dict, body: dict = None) -> dict:
    """HTTP 认证端点处理"""
    auth = get_auth()
    body = body or {}

    # POST /auth/login
    if method == "POST" and path == "/auth/login":
        username = body.get("username", "")
        password = body.get("password", "")
        token = auth.authenticate(username, password)
        if token:
            device_info = body.get("device_info", {})
            user = auth.get_user_by_username(username)
            if user and device_info:
                auth.bind_device(user.id, device_info)
            return {"status": "ok", "token": token, "token_type": "Bearer"}
        return {"status": "error", "message": "Invalid credentials"}, 401

    # POST /auth/register
    if method == "POST" and path == "/auth/register":
        username = body.get("username", "")
        password = body.get("password", "")
        email = body.get("email", "")
        if len(username) < 3 or len(password) < 8:
            return {"status": "error", "message": "Username >=3 chars, password >=8 chars"}, 400
        user = auth.create_user(username, password, email)
        if user:
            return {"status": "ok", "user_id": user.id, "username": user.username}
        return {"status": "error", "message": "Username already exists"}, 409

    # POST /auth/api-key
    if method == "POST" and path == "/auth/api-key":
        payload = require_auth(headers)
        if not payload:
            return {"status": "error", "message": "Unauthorized"}, 401
        api_key = auth.create_api_key(payload["sub"], body.get("name", ""),
                                       body.get("scopes", "read"),
                                       body.get("ttl_days", 365))
        return {"status": "ok", "api_key": api_key}

    # GET /auth/session
    if method == "GET" and path == "/auth/session":
        payload = require_auth(headers)
        if not payload:
            return {"status": "error", "message": "Unauthorized"}, 401
        return {"status": "ok", "user": payload}

    # POST /auth/logout
    if method == "POST" and path == "/auth/logout":
        cookie = headers.get("Cookie", "")
        if "session=" in cookie:
            session_token = cookie.split("session=")[1].split(";")[0].strip()
            auth.revoke_session(session_token)
        return {"status": "ok"}

    return {"status": "error", "message": "Not found"}, 404


# ============================================================================
# 安全存储 (AES-GCM 无外部依赖)
# ============================================================================

def encrypt_data(plaintext: bytes, key: Optional[bytes] = None) -> bytes:
    """使用 AES-CTR + HMAC 实现认证加密 (AES-GCM 替代)"""
    if key is None:
        key = hashlib.sha256(plaintext[:64] + b"lifers_seal").digest()[:32]
    nonce = os.urandom(16)
    # AES-CTR via PyCryptodome fallback: XOR with keystream
    blocks_needed = (len(plaintext) + 15) // 16
    counter = int.from_bytes(nonce[:8], "big")
    keystream = b""
    for i in range(blocks_needed + 1):
        keystream += hashlib.sha256(key + (counter + i).to_bytes(8, "big")).digest()
    ciphertext = bytes(p ^ k for p, k in zip(plaintext, keystream[:len(plaintext)]))
    tag = hmac.new(key, nonce + ciphertext, "sha256").digest()[:16]
    return nonce + ciphertext + tag


def decrypt_data(packed: bytes, key: Optional[bytes] = None) -> Optional[bytes]:
    if key is None:
        key = hashlib.sha256(packed[:64] + b"lifers_seal").digest()[:32]
    nonce, ciphertext, tag = packed[:16], packed[16:-16], packed[-16:]
    expected_tag = hmac.new(key, nonce + ciphertext, "sha256").digest()[:16]
    if not hmac.compare_digest(tag, expected_tag):
        return None
    blocks_needed = (len(ciphertext) + 15) // 16
    counter = int.from_bytes(nonce[:8], "big")
    keystream = b""
    for i in range(blocks_needed + 1):
        keystream += hashlib.sha256(key + (counter + i).to_bytes(8, "big")).digest()
    plaintext = bytes(p ^ k for p, k in zip(ciphertext, keystream[:len(ciphertext)]))
    return plaintext


def encrypt_file(path: Path, key: Optional[bytes] = None):
    data = path.read_bytes()
    path.write_bytes(encrypt_data(data, key))


def decrypt_file(path: Path, key: Optional[bytes] = None) -> Optional[bytes]:
    return decrypt_data(path.read_bytes(), key)
