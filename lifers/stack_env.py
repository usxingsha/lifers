"""Load config/stack.json and apply optional env bridges (SIM_EXEC_CMD, ROBOT_*)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def validate_stack_schema(data: Dict[str, Any]) -> List[str]:
    """
    Lightweight schema validation: check that known keys exist and are of the
    expected type.  Returns a list of warning strings (empty = valid).

    Does *not* raise — warnings are consumed by caller (e.g. logged to stderr).
    """
    from lifers.constants import STACK_SCHEMA

    warnings: List[str] = []

    def _get(d: Dict[str, Any], k: str) -> Any:
        parts = k.split(".")
        v: Any = d
        for p in parts:
            if not isinstance(v, dict):
                return None
            v = v.get(p)
        return v

    for key, expected in STACK_SCHEMA.items():
        val = _get(data, key)
        if val is not None:
            if not isinstance(val, expected):
                warnings.append(
                    f"stack.{key}: expected {expected.__name__}, got {type(val).__name__} = {val!r}"
                )
    return warnings


def load_stack(root: Path) -> Dict[str, Any]:
    """
    读取 ``<root>/config/stack.json``。

    若当前 ``root`` 下无文件（或 JSON 损坏），再尝试 ``$LIFERS_ROOT/config/stack.json``
    （与桥接/扩展注入时机对齐，避免 cwd 与 LIFERS_ROOT 短暂不一致时采样回落到代码硬默认）。
    """
    seen: set[str] = set()
    paths: list[Path] = []

    def _add(p: Path) -> None:
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        if key in seen:
            return
        seen.add(key)
        paths.append(p)

    try:
        _add(Path(root).resolve() / "config" / "stack.json")
    except OSError:
        _add(Path(root) / "config" / "stack.json")

    env_root = os.environ.get("LIFERS_ROOT", "").strip()
    if env_root:
        try:
            _add(Path(env_root).resolve() / "config" / "stack.json")
        except OSError:
            _add(Path(env_root) / "config" / "stack.json")

    for p in paths:
        if not p.is_file():
            continue
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
    return {}


def apply_secrets_file(root: Path) -> None:
    """
    Optional KEY=VALUE lines from config/secrets.env (gitignored).
    os.environ already set by the user / OS wins (setdefault only).
    """
    p = root / "config" / "secrets.env"
    if not p.is_file():
        return
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()
        if v[:1] in ("'", '"') and len(v) >= 2 and v[-1:] == v[:1]:
            v = v[1:-1]
        if k:
            os.environ.setdefault(k, v)


def apply_stack_env(root: Path) -> Dict[str, Any]:
    """
    Merge stack.json into process env when keys are not already set.
    Existing OS env / launcher wins (user override).
    """
    try:
        from lifers.self_heal import heal_stack_at_startup

        heal_stack_at_startup(root)
    except Exception as exc:
        sys.stderr.write(f"LIFERS_PROGRESS self_heal_error {exc}\n")
        sys.stderr.flush()

    os.environ.setdefault("LIFERS_ROOT", str(root.resolve()))
    apply_secrets_file(root)

    data = load_stack(root)
    warnings = validate_stack_schema(data)
    if warnings:
        for w in warnings:
            sys.stderr.write(f"LIFERS_PROGRESS stack_schema_warning {w}\n")
    if not data:
        try:
            from lifers.self_code_runner import process_self_code_queue

            process_self_code_queue(root)
        except Exception as exc:
            sys.stderr.write(f"LIFERS_PROGRESS self_code_queue_error {exc}\n")
            sys.stderr.flush()
        return {}

    brain = data.get("brain") or {}
    if not os.environ.get("MODEL") and brain.get("model"):
        os.environ["MODEL"] = str(brain["model"]).strip().lower()
    if not os.environ.get("SANDBOX") and "sandbox" in brain:
        os.environ["SANDBOX"] = "1" if brain.get("sandbox", True) else "0"

    robot = data.get("robot") or {}
    if not os.environ.get("SIM_EXEC_CMD", "").strip() and robot.get("sim_exec_cmd"):
        os.environ["SIM_EXEC_CMD"] = str(robot["sim_exec_cmd"]).strip()
    if not os.environ.get("ROBOT_SENSE_CMD", "").strip() and robot.get("sense_exec_cmd"):
        os.environ["ROBOT_SENSE_CMD"] = str(robot["sense_exec_cmd"]).strip()
    if not os.environ.get("ROBOT_ACT_CMD", "").strip() and robot.get("act_exec_cmd"):
        os.environ["ROBOT_ACT_CMD"] = str(robot["act_exec_cmd"]).strip()

    try:
        from lifers.self_code_runner import process_self_code_queue

        process_self_code_queue(root)
    except Exception as exc:
        sys.stderr.write(f"LIFERS_PROGRESS self_code_queue_error {exc}\n")
        sys.stderr.flush()

    from .runtime_mode import resolve_runtime

    resolved = resolve_runtime(root, data)
    os.environ.setdefault("LIFERS_RUNTIME", resolved)

    ri = data.get("remote_infer") or {}
    # Cloud chat/completions is opt-in via OS env LIFERS_ALLOW_REMOTE_INFER=1 even if stack.remote_infer.enabled=true.
    # Default: local/Kali-trained weights only (no NVIDIA/OpenAI key required).
    allow_remote = os.environ.get("LIFERS_ALLOW_REMOTE_INFER", "").strip().lower() in ("1", "true", "yes", "on")
    if isinstance(ri, dict) and ri.get("enabled") and allow_remote:
        os.environ.setdefault("LIFERS_REMOTE_CHAT", "1")
        cu = str(ri.get("chat_url") or "").strip()
        if cu:
            os.environ.setdefault("LIFERS_CHAT_URL", cu)
        md = str(ri.get("model") or "").strip()
        if md:
            os.environ.setdefault("LIFERS_CHAT_MODEL", md)
        mx = ri.get("max_tokens")
        if mx is not None:
            os.environ.setdefault("LIFERS_CHAT_MAX_TOKENS", str(mx))
        ev = str(ri.get("api_key_env") or "").strip()
        if ev:
            os.environ.setdefault("LIFERS_CHAT_API_KEY_ENV", ev)

    ll = data.get("local_llm") or {}
    # Local LLM (Ollama / llama.cpp / LM Studio / vLLM) — enabled without LIFERS_ALLOW_REMOTE_INFER.
    # Automatically sets LIFERS_REMOTE_CHAT=1 with localhost URL; no real API key needed.
    if isinstance(ll, dict) and ll.get("enabled"):
        os.environ.setdefault("LIFERS_REMOTE_CHAT", "1")
        cu = str(ll.get("chat_url") or "").strip()
        if cu:
            os.environ.setdefault("LIFERS_CHAT_URL", cu)
        md = str(ll.get("model") or "").strip()
        if md:
            os.environ.setdefault("LIFERS_CHAT_MODEL", md)
        mx = ll.get("max_tokens")
        if mx is not None:
            os.environ.setdefault("LIFERS_CHAT_MAX_TOKENS", str(mx))
        to = ll.get("timeout_sec")
        if to is not None:
            os.environ.setdefault("LIFERS_CHAT_TIMEOUT_SEC", str(to))
        ev = str(ll.get("api_key_env") or "").strip()
        if ev:
            os.environ.setdefault("LIFERS_CHAT_API_KEY_ENV", ev)

    from .openclaw_compat import resolve_openclaw_effective

    ocx = resolve_openclaw_effective(data.get("openclaw") or {})
    if ocx.get("enabled"):
        # 仅表示「已启用上游对照」；不表示安装或调用 OpenClaw
        os.environ.setdefault("LIFERS_OPENCLAW_UPSTREAM", "1")
    if ocx.get("use_external_openclaw_runtime"):
        os.environ.setdefault("LIFERS_OPENCLAW_BRIDGE", "1")
        wp = str(ocx.get("workspace_path", "")).strip()
        if wp and not os.environ.get("OPENCLAW_WORKSPACE", "").strip():
            os.environ["OPENCLAW_WORKSPACE"] = wp

    layout_p = root.parent / "config" / "integrated_layout.json"
    if layout_p.is_file():
        os.environ.setdefault("LIFERS_RS_INTEGRATED_LAYOUT", str(layout_p.resolve()))
        try:
            from .rs_integration import run_rs_integration_bootstrap

            run_rs_integration_bootstrap(root)
        except Exception as exc:
            sys.stderr.write(f"LIFERS_PROGRESS rs_integration_bootstrap_error {exc}\n")
            sys.stderr.flush()

    return data
