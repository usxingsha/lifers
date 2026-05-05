"""
任务流编排：分类 → 分发 → 智脑执行 → 学习写入长期记忆。

桥接默认经 `run_lifers_turn`；可用环境变量 `LIFERS_TASKFLOW=0` 关闭（见 bridge_turn）。
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import os

from lifers_brain.taskflow.classify import classify_task, split_user_message
from lifers_brain.taskflow.context import TaskContext
from lifers_brain.taskflow.handlers import build_default_dispatcher
from lifers_brain.steward import steward_after_learn
from lifers_brain.taskflow.learn import learn_from_turn
from lifers_brain.taskflow.kinds import TaskKind

if TYPE_CHECKING:
    from lifers_brain.agent import LifersAgent


def run_lifers_turn(agent: LifersAgent, agent_input: str) -> str:
    user_text, has_ctx = split_user_message(agent_input)
    kind = classify_task(user_text, has_ctx)
    print(f"LIFERS_PROGRESS taskflow kind={kind.value}", file=sys.stderr, flush=True)
    ctx = TaskContext(agent=agent, agent_input=agent_input, user_text=user_text, kind=kind)
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
