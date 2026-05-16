"""分发上下文：智脑实例 + 原始桥接输入 + 用户尾句 + 分类结果 + 对话路由元数据。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .kinds import TaskKind

if TYPE_CHECKING:
    from lifers.agent import LifersAgent


@dataclass
class TaskContext:
    agent: LifersAgent
    """桥接传入的完整字符串（可含上下文文件前缀）。"""
    agent_input: str
    """仅用户一句（用于分类与闲聊快路径）。"""
    user_text: str
    kind: TaskKind
    """对话推理分发器给出的机器可读原因（如 daily_chat_quick）。"""
    dialogue_route_reason: str = ""
    """给人看的短说明，便于 handler / 日志扩展。"""
    dialogue_route_notes_zh: str = ""
