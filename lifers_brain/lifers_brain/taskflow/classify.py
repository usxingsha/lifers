"""规则分类器：与 Planner / 本能层启发式对齐，产出 TaskKind。"""

from __future__ import annotations

from lifers_brain.agent import Planner
from lifers_brain.daily_intents import (
    cn_web_query_line,
    looks_like_rewrite_or_longform,
    looks_like_remember_or_todo_surface,
    parse_workspace_write_message,
)

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
    if has_context_prefix:
        return TaskKind.FULL_PIPELINE

    s = user_text.strip()
    low = s.lower()
    if s.startswith("方案") or low.startswith("plan "):
        return TaskKind.PLAN_PREVIEW
    if low.startswith("smart ") or s.startswith("智搜"):
        return TaskKind.SMART_SEARCH
    if (s.startswith("流程") and len(s) > 2) or low.startswith("workflow "):
        return TaskKind.WORKFLOW_DUAL
    if low.startswith("kb_search ") or low.startswith("kb_prune") or low.startswith("kb_compact "):
        return TaskKind.KB_CLI
    if low.startswith("sim_run "):
        return TaskKind.SIM_RUN
    if low.startswith("cmd "):
        return TaskKind.CMD_SHELL
    if parse_workspace_write_message(s):
        return TaskKind.TOOL_PLAN
    if "http://" in user_text or "https://" in user_text:
        return TaskKind.URL_FETCH
    if low.startswith("search ") or cn_web_query_line(s):
        return TaskKind.WEB_SEARCH
    if looks_like_rewrite_or_longform(s) or looks_like_remember_or_todo_surface(s):
        return TaskKind.FULL_PIPELINE

    p = Planner()
    rw = p.plan_real_world_instinct(s)
    inner = p.plan(user_text)

    # smart / 中文前缀 已在上方早返回。
    if not rw and not inner:
        return TaskKind.CHAT_QUICK
    if rw and not inner:
        return TaskKind.REAL_WORLD
    if ":" in s and ("\\" in s or "/" in s):
        for token in s.split():
            if (":\\" in token) or (token.startswith("/") and "/" in token):
                return TaskKind.FS_PATH
    if inner:
        return TaskKind.TOOL_PLAN
    return TaskKind.FULL_PIPELINE
