"""类型 WORKFLOW_DUAL：流程/workflow，KB→Web 固定两步，由智脑 step 处理。"""

from __future__ import annotations

from lifers.taskflow.context import TaskContext
from lifers.taskflow.kinds import HandlerResult, TaskKind

from .common import run_agent_step

KIND = TaskKind.WORKFLOW_DUAL


def handle(ctx: TaskContext) -> HandlerResult:
    return run_agent_step(ctx)
