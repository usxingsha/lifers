"""类型 TOOL_PLAN：Planner 产出其它工具链（非上述显式口令），由智脑 step 执行。"""

from __future__ import annotations

from lifers.taskflow.context import TaskContext
from lifers.taskflow.kinds import HandlerResult, TaskKind

from .common import run_agent_step

KIND = TaskKind.TOOL_PLAN


def handle(ctx: TaskContext) -> HandlerResult:
    return run_agent_step(ctx)
