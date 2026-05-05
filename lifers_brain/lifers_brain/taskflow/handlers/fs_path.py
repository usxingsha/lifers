"""类型 FS_PATH：检测到路径令牌，fs_read 由智脑 step 处理。"""

from __future__ import annotations

from lifers_brain.taskflow.context import TaskContext
from lifers_brain.taskflow.kinds import HandlerResult, TaskKind

from .common import run_agent_step

KIND = TaskKind.FS_PATH


def handle(ctx: TaskContext) -> HandlerResult:
    return run_agent_step(ctx)
