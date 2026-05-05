"""日常用语与任务类型对齐（无 agent 依赖，供 classify / Planner 共用）。"""

from __future__ import annotations

from typing import Optional, Tuple


def cn_web_query_line(text: str) -> Optional[str]:
    """若整句为「中文显式检索」则返回查询串，否则 None（与 `search …` 语义对齐）。"""
    s = text.strip()
    if not s:
        return None
    for head in ("帮我搜索", "帮我搜一下", "帮我查查", "帮我查询"):
        if s.startswith(head):
            q = s[len(head) :].strip()
            return q or None
    for head in ("搜索", "搜一下", "查查", "查询"):
        if s.startswith(head):
            q = s[len(head) :].strip()
            if q:
                return q
    if s.startswith("搜") and len(s) >= 2 and s[1].isspace():
        q = s[1:].strip()
        return q or None
    return None


def looks_like_rewrite_or_longform(text: str) -> bool:
    """总结 / 续写 / 翻译等：宜走完整管道与工具链。"""
    s = text.strip()
    if not s:
        return False
    low = s.lower()
    if low.startswith("translate ") or s.startswith("翻译"):
        return len(s) > 3
    heads = ("总结", "续写", "改写", "润色", "扩写", "缩写", "概括", "提炼要点", "写一段", "写一篇")
    return any(s.startswith(h) for h in heads)


def looks_like_remember_or_todo_surface(text: str) -> bool:
    """备忘 / 待办类表面语句：走完整管道便于写入 KB 或工具链。"""
    s = text.strip()
    if not s:
        return False
    heads = ("备忘", "待办", "提醒我", "提醒我明天", "提醒我后天", "记事", "记录一下", "帮我记")
    return any(s.startswith(h) for h in heads)


def parse_workspace_write_message(text: str) -> Optional[Tuple[str, str]]:
    """
    首行 `rel_write|workspace_write|self_write <rel_path>`，自第二行起为完整文件正文。
    用于工具链直接改 LIFERS_ROOT 下任意相对路径（含自身源码）。
    """
    raw = text.strip()
    if "\n" not in raw:
        return None
    first, rest = raw.split("\n", 1)
    fl = first.strip()
    low = fl.lower()
    for prefix in ("rel_write ", "workspace_write ", "self_write "):
        if low.startswith(prefix):
            rel = fl[len(prefix) :].strip()
            if rel:
                return rel, rest
    return None
