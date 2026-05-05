"""类型分发：TaskKind -> 处理函数（各类型独立处理库）。"""

from __future__ import annotations

from typing import Callable, Dict

from .context import TaskContext
from .kinds import HandlerResult, TaskKind

HandlerFn = Callable[[TaskContext], HandlerResult]


class TaskDispatcher:
    def __init__(self, routes: Dict[TaskKind, HandlerFn]) -> None:
        self._routes = dict(routes)
        self._fallback = routes[TaskKind.FULL_PIPELINE]

    def dispatch(self, kind: TaskKind, ctx: TaskContext) -> HandlerResult:
        fn = self._routes.get(kind, self._fallback)
        return fn(ctx)
