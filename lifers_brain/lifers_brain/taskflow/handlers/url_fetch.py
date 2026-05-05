"""类型 URL_FETCH：输入中含 http(s) URL，抓取与证据提取由智脑 step 处理。"""

from __future__ import annotations

from lifers_brain.taskflow.context import TaskContext
from lifers_brain.taskflow.kinds import HandlerResult, TaskKind

from .common import run_agent_step

KIND = TaskKind.URL_FETCH


def handle(ctx: TaskContext) -> HandlerResult:
    return run_agent_step(ctx)
