"""类型 CMD_SHELL：cmd 前缀命令执行（沙箱内），由智脑 step 处理。"""

from __future__ import annotations

from lifers_brain.taskflow.context import TaskContext
from lifers_brain.taskflow.kinds import HandlerResult, TaskKind

from .common import run_agent_step

KIND = TaskKind.CMD_SHELL


def handle(ctx: TaskContext) -> HandlerResult:
    return run_agent_step(ctx)
