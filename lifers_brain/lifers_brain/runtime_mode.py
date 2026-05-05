"""
三合一智脑：解析当前宿主状态（本地智脑 / 仿真人 / 机器人）。

优先级（由高到低）：
1. 环境变量 LIFERS_RUNTIME = brain | robot | human_sim
2. config/stack.json → runtime.role（或兼容 runtime.profile）
3. role=auto 时：LIFERS_ON_ROBOT / ON_ROBOT、或 config/.host_robot 标记文件 → robot；
   可选 runtime.auto_infer_from_robot_cmds=true 且 stack 中配置了机器人命令 → robot
4. 默认 brain（电脑侧本地智脑）

机器人镜像建议在 stack 中写死 "role": "robot"；电脑侧写 "brain" 或依赖默认。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

VALID = frozenset({"brain", "robot", "human_sim"})


def _truthy_env(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _robot_cmds_in_stack(data: Dict[str, Any]) -> bool:
    robot = data.get("robot") or {}
    for k in ("sense_exec_cmd", "act_exec_cmd", "sim_exec_cmd"):
        if str(robot.get(k) or "").strip():
            return True
    return False


def resolve_runtime(root: Path, stack: Optional[Dict[str, Any]] = None) -> str:
    env_r = os.environ.get("LIFERS_RUNTIME", "").strip().lower()
    if env_r in VALID:
        return env_r

    from .stack_env import load_stack

    data = stack if stack is not None else load_stack(root)
    rt = data.get("runtime") or {}
    role = str(rt.get("role") or rt.get("profile") or "auto").strip().lower()
    if role in VALID:
        return role
    if role not in ("auto", ""):
        return "brain"

    if _truthy_env("LIFERS_ON_ROBOT") or _truthy_env("ON_ROBOT"):
        return "robot"
    marker = root / "config" / ".host_robot"
    if marker.is_file():
        return "robot"
    if rt.get("auto_infer_from_robot_cmds") and _robot_cmds_in_stack(data):
        return "robot"
    return "brain"


_RUNTIME_LINES = {
    "brain": "【三合一·当前状态】宿主=本地智脑（电脑/边缘）：优先推理、文件、知识库与工具链。",
    "robot": "【三合一·当前状态】宿主=机器人本体：优先 sense_snapshot、motion_execute、manipulate、safety_stop；勿默认桌面路径可用。",
    "human_sim": "【三合一·当前状态】宿主=仿真人模式：以人格与自然对话为先，工具为辅。",
}

_RUNTIME_LABEL_ZH = {"brain": "本地智脑", "robot": "机器人", "human_sim": "仿真人"}


def runtime_system_line(role: str) -> str:
    """SYSTEM 前缀（已由 resolve_runtime 得到 role）。"""
    return _RUNTIME_LINES.get(role, _RUNTIME_LINES["brain"])


def runtime_label_from_role(role: str) -> str:
    return _RUNTIME_LABEL_ZH.get(role, role)


def runtime_system_hint(root: Path, stack: Optional[Dict[str, Any]] = None) -> str:
    """Short Chinese line for SYSTEM context (三合一状态)."""
    return runtime_system_line(resolve_runtime(root, stack))


def runtime_label_zh(root: Path, stack: Optional[Dict[str, Any]] = None) -> str:
    return runtime_label_from_role(resolve_runtime(root, stack))
