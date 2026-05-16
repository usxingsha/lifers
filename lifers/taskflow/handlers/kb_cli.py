"""类型 KB_CLI：kb_search / kb_prune / kb_compact 等显式记忆命令。"""

from __future__ import annotations

from lifers.taskflow.context import TaskContext
from lifers.taskflow.kinds import HandlerResult, TaskKind

from .common import run_agent_step

KIND = TaskKind.KB_CLI


def handle(ctx: TaskContext) -> HandlerResult:
    return run_agent_step(ctx)
