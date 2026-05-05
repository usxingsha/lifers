"""类型 PLAN_PREVIEW：方案/plan 仅预览工具链，由智脑 step 内逻辑处理。"""

from __future__ import annotations

from lifers_brain.taskflow.context import TaskContext
from lifers_brain.taskflow.kinds import HandlerResult, TaskKind

from .common import run_agent_step

KIND = TaskKind.PLAN_PREVIEW


def handle(ctx: TaskContext) -> HandlerResult:
    return run_agent_step(ctx)
