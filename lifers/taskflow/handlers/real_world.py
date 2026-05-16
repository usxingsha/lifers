"""类型 REAL_WORLD：仅本能层 real_world（时钟/天气/地图等），由智脑 step 处理。"""

from __future__ import annotations

from lifers.taskflow.context import TaskContext
from lifers.taskflow.kinds import HandlerResult, TaskKind

from .common import run_agent_step

KIND = TaskKind.REAL_WORLD


def handle(ctx: TaskContext) -> HandlerResult:
    return run_agent_step(ctx)
