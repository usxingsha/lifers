"""类型 SIM_RUN：sim_run 仿真任务。"""

from __future__ import annotations

from lifers.taskflow.context import TaskContext
from lifers.taskflow.kinds import HandlerResult, TaskKind

from .common import run_agent_step

KIND = TaskKind.SIM_RUN


def handle(ctx: TaskContext) -> HandlerResult:
    return run_agent_step(ctx)
