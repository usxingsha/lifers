"""共享执行：完整智脑单轮（含工具链、方案、智搜等）。"""

from __future__ import annotations

from lifers_brain.taskflow.context import TaskContext
from lifers_brain.taskflow.kinds import HandlerResult


def run_agent_step(ctx: TaskContext) -> HandlerResult:
    text = ctx.agent_input
    out = ctx.agent.step(text)
    return HandlerResult(handled=True, reply=out, meta={"handler": "agent.step", "kind": ctx.kind.value})
