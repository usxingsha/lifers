"""类型 SMART_SEARCH：smart / 中文「智搜」前缀，先 KB 再联网，由智脑 step 处理（与自动检索并列，非唯一入口）。"""

from __future__ import annotations

from lifers.taskflow.context import TaskContext
from lifers.taskflow.kinds import HandlerResult, TaskKind

from .common import run_agent_step

KIND = TaskKind.SMART_SEARCH


def handle(ctx: TaskContext) -> HandlerResult:
    return run_agent_step(ctx)
