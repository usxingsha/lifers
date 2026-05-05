"""智脑学习：将任务类型与对话摘要写入长期记忆，供后续检索与统计。"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

from lifers_brain.memory import MemoryItem

from .context import TaskContext
from .kinds import HandlerResult

if TYPE_CHECKING:
    from lifers_brain.agent import LifersAgent


def learn_from_turn(agent: LifersAgent, ctx: TaskContext, res: HandlerResult) -> None:
    if os.environ.get("LIFERS_TASKFLOW_LEARN", "1").strip().lower() in ("0", "false", "no", "off"):
        return
    if not res.reply or not res.reply.strip():
        return
    try:
        agent.longterm.add(
            MemoryItem(
                type="taskflow",
                content={
                    "kind": str(ctx.kind.value),
                    "user_excerpt": ctx.user_text[:480],
                    "reply_excerpt": res.reply[:960],
                    "meta": dict(res.meta),
                },
                importance=0.24,
                source="taskflow:learn",
                ts_ms=int(time.time() * 1000),
            )
        )
    except Exception:
        pass
