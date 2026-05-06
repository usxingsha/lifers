"""类型 CHAT_QUICK：无工具链的短对话，由智脑 quick_chat（本地小模型 + 会话窗口）处理。"""

from __future__ import annotations

from lifers_brain.taskflow.context import TaskContext
from lifers_brain.taskflow.kinds import HandlerResult, TaskKind

KIND = TaskKind.CHAT_QUICK


def handle(ctx: TaskContext) -> HandlerResult:
    r = ctx.agent.quick_chat(ctx.user_text)
    meta = {
        "handler": "chat_quick",
        "kind": KIND.value,
        "dialogue_route_reason": getattr(ctx, "dialogue_route_reason", "") or "",
        "dialogue_route_notes_zh": getattr(ctx, "dialogue_route_notes_zh", "") or "",
    }
    return HandlerResult(handled=True, reply=r, meta=meta)
