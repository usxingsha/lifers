"""
多类型任务流：分类器 → 分发器 → 各类型处理库 → 智脑执行 → 学习（长期记忆）。

详细流程见同目录 `FLOW.md`。
"""

from __future__ import annotations

from lifers.taskflow.kinds import HandlerResult, TaskKind
from lifers.taskflow.orchestrator import run_lifers_turn

__all__ = ["TaskKind", "HandlerResult", "run_lifers_turn"]
