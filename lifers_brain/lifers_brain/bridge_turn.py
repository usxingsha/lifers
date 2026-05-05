"""
Shared single-turn agent invocation for stdin bridge and lifers_gate HTTP service.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def _safe_read_under_root(root: Path, rel: str, max_chars: int = 6000) -> str:
    target = (root / rel).resolve()
    try:
        root_res = root.resolve()
        if not str(target).startswith(str(root_res)):
            return ""
        if not target.is_file():
            return ""
        return target.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


def lifers_turn(root: Path, text: str, context_files: List[Any]) -> Dict[str, Any]:
    """
    Run one LifersAgent step. Returns {"ok": bool, "text": str, "error": optional str}.
    """
    from lifers_brain.stack_env import apply_stack_env

    apply_stack_env(root)
    # Bridge（扩展）默认注入 LIFERS_FORCE_LOCAL_ONLY：禁用云端 chat/completions，仅用本地权重 + 联网工具。
    if os.environ.get("LIFERS_FORCE_LOCAL_ONLY", "").strip().lower() in ("1", "true", "yes", "on"):
        os.environ.pop("LIFERS_REMOTE_CHAT", None)

    ctx_files = context_files if isinstance(context_files, list) else []
    max_ctx = int(os.environ.get("LIFERS_CONTEXT_MAX_FILES", "32").strip() or "32")
    if max_ctx < 1:
        max_ctx = 32

    prefix_parts: list[str] = []
    for rel in ctx_files[:max_ctx]:
        if not isinstance(rel, str) or not rel.strip():
            continue
        rel_n = rel.replace("\\", "/").lstrip("/")
        blob = _safe_read_under_root(root, rel_n)
        if blob:
            prefix_parts.append(f"--- context file: {rel_n} ---\n{blob}")

    if prefix_parts:
        full = "\n\n".join(prefix_parts) + "\n\n--- user message ---\n" + text
    else:
        full = text

    from lifers_brain.agent import AgentConfig, LifersAgent
    from lifers_brain.model_names import canonical_brain_model
    from lifers_brain.taskflow.orchestrator import run_lifers_turn

    model = canonical_brain_model(os.environ.get("MODEL", "transformer"))
    sandbox = os.environ.get("SANDBOX", "1") == "1"
    agent = LifersAgent(AgentConfig(root_dir=root, model=model, sandbox=sandbox))
    try:
        print("LIFERS_PROGRESS bridge LifersAgent.step 开始 …", file=sys.stderr, flush=True)
        use_tf = os.environ.get("LIFERS_TASKFLOW", "1").strip().lower() not in ("0", "false", "no", "off")
        reply = run_lifers_turn(agent, full) if use_tf else agent.step(full)
        return {"ok": True, "text": reply}
    except Exception as e:
        return {"ok": False, "text": "", "error": str(e)}


def lifers_turn_from_json_body(root: Path, raw_json: str) -> Dict[str, Any]:
    """Parse stdin-like JSON body and run lifers_turn."""
    if not raw_json.strip():
        return {"ok": False, "text": "", "error": "empty body"}
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        return {"ok": False, "text": "", "error": f"invalid json: {e}"}
    text = str(data.get("text", "")).strip()
    ctx = data.get("contextFiles") or []
    return lifers_turn(root, text, ctx)
