"""
任务流编排：分类 → 分发 → 智脑执行 → 学习写入长期记忆。

桥接默认经 `run_lifers_turn`；可用环境变量 `LIFERS_TASKFLOW=0` 关闭（见 bridge_turn）。
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import os

from lifers.taskflow.classify import split_user_message
from lifers.taskflow.dialogue_router import infer_dialogue_route
from lifers.taskflow.context import TaskContext
from lifers.taskflow.handlers import build_default_dispatcher
from lifers.steward import steward_after_learn
from lifers.taskflow.learn import learn_from_turn
from lifers.taskflow.kinds import TaskKind
from lifers.inference_pipeline import log_inference

if TYPE_CHECKING:
    from lifers.agent import LifersAgent


def run_lifers_turn(agent: LifersAgent, agent_input: str) -> str:
    user_text, has_ctx = split_user_message(agent_input)
    route = infer_dialogue_route(user_text, has_ctx, emit=True, planner=agent.planner)
    kind = route.kind
    print(f"LIFERS_PROGRESS taskflow kind={kind.value}", file=sys.stderr, flush=True)
    log_inference(
        "taskflow_route",
        kind=kind.value,
        route_reason=route.reason,
        notes_zh=(route.notes_zh or "")[:240] or None,
        has_context_prefix=bool(has_ctx),
    )
    ctx = TaskContext(
        agent=agent,
        agent_input=agent_input,
        user_text=user_text,
        kind=kind,
        dialogue_route_reason=route.reason,
        dialogue_route_notes_zh=route.notes_zh,
    )
    dispatcher = build_default_dispatcher()
    res = dispatcher.dispatch(kind, ctx)
    # CHAT_QUICK 默认不写 longterm + 不跑 steward，避免 SQLite 膨胀后 count/prune 拖垮 Bridge（秒回路径）。
    if kind == TaskKind.CHAT_QUICK:
        if os.environ.get("LIFERS_QUICK_CHAT_LEARN", "0").strip().lower() in ("1", "true", "yes", "on"):
            learn_from_turn(agent, ctx, res)
            steward_after_learn(agent)
    else:
        learn_from_turn(agent, ctx, res)
        steward_after_learn(agent)
    return res.reply
