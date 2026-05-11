"""规则分类器：仅做消息切分并委托 `dialogue_router` 产出 TaskKind。

新增路由条件请改 `dialogue_router.py` / `daily_intents.py`，勿在本文件复制 if/else（见 taskflow/FLOW.md「单一事实来源」）。
"""

from __future__ import annotations

from .dialogue_router import infer_dialogue_route
from .kinds import TaskKind

_USER_SEP = "\n--- user message ---\n"


def split_user_message(agent_input: str) -> tuple[str, bool]:
    """
    返回 (用户尾句, 是否带上下文前缀)。
    有前缀时一律走 FULL_PIPELINE，避免把文件里的路径误当成用户口令。
    """
    if _USER_SEP in agent_input:
        tail = agent_input.rsplit(_USER_SEP, 1)[-1].strip()
        return tail, True
    return agent_input.strip(), False


def classify_task(user_text: str, has_context_prefix: bool) -> TaskKind:
    """兼容入口：等价于 `infer_dialogue_route(...).kind`（含 stderr 路由进度）。"""
    return infer_dialogue_route(user_text, has_context_prefix, emit=True).kind
