"""
Shared single-turn agent invocation for stdin bridge and lifers_gate HTTP service.

CHAT_QUICK（Agents UI）在 **MODEL=transformer** 且边缘 CPU 上时，agent 会默认收紧
`LIFERS_QUICK_SESSION_CONTEXT_CHARS` / `LIFERS_QUICK_PACK_MAX_CHARS`，减轻本地生成超时；
可用环境变量按需放宽。

Agents Chat 设 `LIFERS_AGENTS_UI_BRIDGE=1` 时，默认在快路径回复末追加 **【本轮·生成锚】**（`agent._append_quick_reply_time_footer`）；
可用 `LIFERS_QUICK_TIME_FOOTER=0` 关闭或 `=1` 强制开启（无 Bridge 时亦可在终端试验）。
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


def _strip_api_key_nag_when_local_only(reply: str) -> str:
    """Agents Chat 已强制本地时，去掉模型复述的「须配 NVIDIA 密钥」类段落。"""
    if os.environ.get("LIFERS_FORCE_LOCAL_ONLY", "").strip().lower() not in ("1", "true", "yes", "on"):
        return reply
    if not reply or not reply.strip():
        return reply
    needles = (
        "NVIDIA_API_KEY",
        "LIFERS_CHAT_API_KEY",
        "未检测到 API 密钥",
        "未检测到API密钥",
        "未检测到 NVIDIA",
    )
    lines = reply.split("\n")
    kept: list[str] = []
    for ln in lines:
        if any(n in ln for n in needles):
            continue
        if "远程推理" in ln and "密钥" in ln:
            continue
        if "云端" in ln and "密钥" in ln and ("须" in ln or "必须" in ln or "请设置" in ln):
            continue
        kept.append(ln)
    out = "\n".join(kept).strip()
    return out if out else reply


def _sanitize_bridge_input(text: str) -> str:
    """移除 NUL 等会破坏 JSON/日志的控制字符（运行时最小输入清洗，非 XSS 沙箱）。"""
    if not text:
        return text
    return text.replace("\x00", "").replace("\ufeff", "")


def lifers_turn(root: Path, text: str, context_files: List[Any]) -> Dict[str, Any]:
    """
    Run one LifersAgent step. Returns {"ok": bool, "text": str, "error": optional str}.
    """
    from lifers_brain.stack_env import apply_stack_env

    apply_stack_env(root)
    text = _sanitize_bridge_input(text)
    # Bridge（扩展）默认注入 LIFERS_FORCE_LOCAL_ONLY：禁用云端 chat/completions，仅用本地权重 + 联网工具。
    if os.environ.get("LIFERS_FORCE_LOCAL_ONLY", "").strip().lower() in ("1", "true", "yes", "on"):
        os.environ.pop("LIFERS_REMOTE_CHAT", None)
    # CHAT_QUICK 自动联网：非 Agents UI 发起的 bridge 默认关（避免宿主误 export LIFERS_QUICK_WEB=1）。
    # Agents Chat 会设 LIFERS_AGENTS_UI_BRIDGE=1，此时以扩展传入的 LIFERS_QUICK_WEB 为准。
    # 终端 / gate 若要沿用宿主 LIFERS_QUICK_WEB，设 LIFERS_QUICK_WEB_RESPECT_HOST=1。
    ui = os.environ.get("LIFERS_AGENTS_UI_BRIDGE", "").strip().lower() in ("1", "true", "yes", "on")
    respect = os.environ.get("LIFERS_QUICK_WEB_RESPECT_HOST", "").strip().lower() in ("1", "true", "yes", "on")
    if not ui and not respect:
        os.environ["LIFERS_QUICK_WEB"] = "0"

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
        reply = _strip_api_key_nag_when_local_only(reply)
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
