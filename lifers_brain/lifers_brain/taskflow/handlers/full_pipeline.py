"""类型 FULL_PIPELINE：含上下文文件前缀或其它兜底，整包交给 LifersAgent.step。"""

from __future__ import annotations

from lifers_brain.taskflow.context import TaskContext
from lifers_brain.taskflow.kinds import HandlerResult, TaskKind

from .common import run_agent_step

KIND = TaskKind.FULL_PIPELINE


def handle(ctx: TaskContext) -> HandlerResult:
    return run_agent_step(ctx)
