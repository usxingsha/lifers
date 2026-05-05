"""
与「主流 Chat API」对齐的最小消息结构（OpenAI chat/completions 风格 role + content）。

Lifers 本地推理仍由 LocalBrain / taskflow 驱动；本模块只提供 **数据结构互操作**，
便于与 openai_compat_chat、网关或外部脚本对接，**不**引入 PyTorch/Transformers 等重型依赖。
"""

from __future__ import annotations

from typing import Any, List, Literal, TypedDict


class ChatMessage(TypedDict, total=False):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str


def coerce_messages(obj: Any) -> List[ChatMessage]:
    """把任意 list[dict] 规范成 ChatMessage 列表，忽略缺字段项。"""
    if not isinstance(obj, list):
        return []
    out: list[ChatMessage] = []
    for it in obj:
        if not isinstance(it, dict):
            continue
        role = it.get("role")
        content = it.get("content")
        if role not in ("system", "user", "assistant", "tool"):
            continue
        if not isinstance(content, str):
            continue
        m: ChatMessage = {"role": role, "content": content}
        name = it.get("name")
        if isinstance(name, str) and name.strip():
            m["name"] = name.strip()
        out.append(m)
    return out


def transcript_to_messages(system: str, user: str, assistant: str) -> List[ChatMessage]:
    """把一轮对话压成三条标准消息（用于对照测试或导出）。"""
    return [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": assistant.strip()},
    ]
