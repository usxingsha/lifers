"""
人体器官 / 系统 ↔ 智架子系统（类比说明），写入本地 LLM 的 SYSTEM 段。

完整映射表默认从 config/organ_capabilities.json 加载（相对 lifers 根）。
也可在 stack.organ_system.custom_for_llm 覆盖正文。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_BODY = """感官 ~ 感知：上下文路径、web、fs_read、Planner。
工作记忆 ~ SessionMemory + Scratchpad。
长期记忆 ~ LongTermMemory；意向到动作 ~ Planner+ToolCall。
执行 ~ cmd/fs/robot；自主神经 ~ instincts；免疫 ~ SANDBOX；内分泌 ~ physiology_sim（可选）。
人格 ~ human_sim。"""


def _safe_inside_root(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _render_capabilities(
    organs: List[Dict[str, Any]],
    disclaimer_zh: str,
    goal_zh: str = "",
) -> str:
    lines = []
    if goal_zh.strip():
        lines.append("【全能型对齐目标 Omnidirectional】\n" + goal_zh.strip())
    if disclaimer_zh.strip():
        lines.append(disclaimer_zh.strip())
    for o in organs:
        ops = "；".join(o.get("operations") or [])
        sk = "；".join(o.get("stack_keys") or [])
        tl = ",".join(o.get("tools") or [])
        envs = "；".join(o.get("env_or_commands") or [])
        instinct = str(o.get("instinct_link_zh") or "").strip()
        block = (
            f"【{o.get('name_zh', '')} · {o.get('id', '')}】\n"
            f"  人·能力: {o.get('human_capability_zh', '')}\n"
            f"  机·对齐: {o.get('llm_analog_zh', '')}\n"
            f"  操作: {ops}\n"
            f"  配置(stack/扩展): {sk}\n"
            f"  工具: {tl}\n"
            f"  命令/环境: {envs}"
        )
        if instinct:
            block += f"\n  本能协同(instincts): {instinct}"
        lines.append(block.strip())
    return "\n\n".join(lines)


def _body_from_capabilities_file(root: Path, rel: str) -> Optional[str]:
    rel = (rel or "").strip() or "config/organ_capabilities.json"
    p = (root / rel).resolve()
    if not _safe_inside_root(root, p) or not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    organs = data.get("organs")
    if not isinstance(organs, list):
        return None
    disc = str(data.get("disclaimer_zh") or "").strip()
    goal = str(data.get("goal_zh") or "").strip()
    return _render_capabilities(organs, disc, goal)


def format_organ_system_context(stack: Dict[str, Any], root: Optional[Path] = None) -> str:
    sec = stack.get("organ_system")
    if not isinstance(sec, dict):
        return ""
    if sec.get("enabled") is False:
        return ""

    custom = str(sec.get("custom_for_llm") or "").strip()
    if custom:
        body = custom
    else:
        body = None
        if root is not None:
            rel = str(sec.get("capabilities_file") or "config/organ_capabilities.json")
            body = _body_from_capabilities_file(root, rel)
        if not body:
            body = DEFAULT_BODY

    title = str(sec.get("title") or "").strip()
    prefix = (title + "\n\n") if title else "ORGAN_SYSTEM_ANALOG（器官-系统-LLM 对齐表）:\n\n"

    extra = str(sec.get("extra") or "").strip()
    if extra:
        body = body.rstrip() + "\n\n【附加说明】\n" + extra

    text = (prefix + body).strip() + "\n"
    try:
        cap = int(sec.get("max_inject_chars", 12000) or 12000)
    except (TypeError, ValueError):
        cap = 12000
    cap = max(400, min(cap, 32000))
    if len(text) > cap:
        return text[: cap - 24].rstrip() + "\n…(organ_system 已截断)\n"
    return text
