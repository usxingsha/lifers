"""类型 WEB_SEARCH：search 前缀联网搜索。"""

from __future__ import annotations

from lifers_brain.taskflow.context import TaskContext
from lifers_brain.taskflow.kinds import HandlerResult, TaskKind

from .common import run_agent_step

KIND = TaskKind.WEB_SEARCH


def handle(ctx: TaskContext) -> HandlerResult:
    return run_agent_step(ctx)
