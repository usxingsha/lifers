"""
对话推理分发器：根据用户尾句与是否带「上下文文件前缀」推断 TaskKind，并给出可审计的路由原因。

- 供 `classify_task` / 编排器统一调用；stderr 打 `LIFERS_PROGRESS dialogue_route …` 便于 Agents Chat「执行过程」里看到走向。
- 规则与 `classify.py` 历史行为一致，集中在此便于日后加「多意图 / 置信度 / A-B 策略」而不散在 agent 里。
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Dict

from lifers_brain.agent import Planner, _META_CAP_SHORT, _META_SELF_RE
from lifers_brain.daily_intents import (
    cn_web_query_line,
    looks_like_build_or_code_request,
    looks_like_rewrite_or_longform,
    looks_like_remember_or_todo_surface,
    parse_workspace_write_message,
)

from .kinds import TaskKind


@dataclass
class DialogueRoute:
    """单轮路由结果（主类型 + 给人看的说明 + 可选调试载荷）。"""

    kind: TaskKind
    reason: str
    notes_zh: str = ""
    confidence: float = 1.0
    debug: Dict[str, Any] | None = None


def _emit_route_coarse(bucket: str, signals: list[str]) -> None:
    """正式 dialogue_route 分支前的粗粒度桶（审计用，不等价于最终 TaskKind）。"""
    line = "LIFERS_PROGRESS route_coarse " + json.dumps({"bucket": bucket, "signals": signals}, ensure_ascii=False)
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def _coarse_intent_hint(s: str, low: str) -> None:
    if low.startswith("search ") or cn_web_query_line(s):
        _emit_route_coarse("tool_or_web", ["search_or_cn_query"])
        return
    if low.startswith("cmd "):
        _emit_route_coarse("shell", ["explicit_cmd"])
        return
    if "http://" in s or "https://" in s:
        _emit_route_coarse("fetch", ["url_in_text"])
        return
    if parse_workspace_write_message(s):
        _emit_route_coarse("workspace_write", ["workspace_write_surface"])
        return
    if looks_like_build_or_code_request(s) or looks_like_rewrite_or_longform(s) or looks_like_remember_or_todo_surface(s):
        _emit_route_coarse("full_pipeline_likely", ["longform_or_build_or_remember"])
        return
    _emit_route_coarse("chat_or_planner", ["dialogue_router"])


def _emit_progress(route: DialogueRoute) -> None:
    payload = {
        "kind": route.kind.value,
        "reason": route.reason,
        "notes_zh": route.notes_zh,
        "confidence": route.confidence,
    }
    line = "LIFERS_PROGRESS dialogue_route " + json.dumps(payload, ensure_ascii=False)
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def infer_dialogue_route(user_text: str, has_context_prefix: bool, *, emit: bool = True) -> DialogueRoute:
    """
    根据用户输入推断应走的任务类型（与历史 classify_task 行为对齐）。

    :param emit: True 时写入 stderr（Bridge / gate 可见）。
    """
    if has_context_prefix:
        route = DialogueRoute(
            kind=TaskKind.FULL_PIPELINE,
            reason="context_prefix",
            notes_zh="消息含「上下文文件」前缀：走完整管线以免把文件内容误当口令。",
        )
        if emit:
            _emit_progress(route)
        return route

    s = user_text.strip()
    low = s.lower()
    _coarse_intent_hint(s, low)

    def ok(k: TaskKind, reason: str, notes_zh: str = "", **dbg: Any) -> DialogueRoute:
        r = DialogueRoute(kind=k, reason=reason, notes_zh=notes_zh, debug=dbg if dbg else None)
        if emit:
            _emit_progress(r)
        return r

    if s.startswith("方案") or low.startswith("plan "):
        return ok(TaskKind.PLAN_PREVIEW, "explicit_plan", "用户要「方案/plan」预览，不直接执行工具。")
    if low.startswith("smart ") or s.startswith("智搜"):
        return ok(TaskKind.SMART_SEARCH, "explicit_smart", "智搜 / smart：记忆 + 联网组合检索。")
    if (s.startswith("流程") and len(s) > 2) or low.startswith("workflow "):
        return ok(TaskKind.WORKFLOW_DUAL, "explicit_workflow", "流程 / workflow：KB 后再 web。")
    if low.startswith("kb_search ") or low.startswith("kb_prune") or low.startswith("kb_compact "):
        return ok(TaskKind.KB_CLI, "explicit_kb_cli", "长期记忆 CLI 子命令。")
    if low.startswith("sim_run "):
        return ok(TaskKind.SIM_RUN, "explicit_sim_run", "仿真运行。")
    if low.startswith("cmd "):
        return ok(TaskKind.CMD_SHELL, "explicit_cmd", "命令行执行请求。")
    if parse_workspace_write_message(s):
        return ok(TaskKind.TOOL_PLAN, "workspace_write_surface", "检测到工作区写入类结构化消息。")
    if "http://" in user_text or "https://" in user_text:
        return ok(TaskKind.URL_FETCH, "url_in_text", "文本中含 http(s) 链接，走 URL 抓取。")
    if low.startswith("search ") or cn_web_query_line(s):
        return ok(TaskKind.WEB_SEARCH, "web_or_cn_query", "显式 search 或中文检索型问句。")
    if looks_like_rewrite_or_longform(s) or looks_like_remember_or_todo_surface(s):
        return ok(
            TaskKind.FULL_PIPELINE,
            "rewrite_remember_or_longform",
            "长文改写 / 备忘待办表面 / 长形输入：走完整管线。",
        )
    if looks_like_build_or_code_request(s):
        return ok(
            TaskKind.FULL_PIPELINE,
            "build_or_code_project",
            "实现 / 开发 / 做游戏或项目类：走完整管线以启用工具链与多步规划。",
        )

    # 与 agent `_quick_chat_meta_capability_reply` 对齐：主流对话里常见的 intent≈assistant_meta。
    if _META_SELF_RE.search(s) or _META_CAP_SHORT.match(s):
        return ok(
            TaskKind.CHAT_QUICK,
            "assistant_meta_intent",
            "问助手能力或身份：仍走 CHAT_QUICK，由本地说明模板作答（不当作泛检索）。",
            assistant_meta=True,
        )

    p = Planner()
    rw = p.plan_real_world_instinct(s)
    inner = p.plan(user_text)

    if not rw and not inner:
        return ok(
            TaskKind.CHAT_QUICK,
            "daily_chat_quick",
            "日常对话：无工具链信号，走 CHAT_QUICK（本地快答）。",
            planner_rw=False,
            planner_inner=False,
        )
    if rw and not inner:
        return ok(TaskKind.REAL_WORLD, "real_world_instinct_only", "仅命中本能/现实侧链，无结构化工具计划。")
    if ":" in s and ("\\" in s or "/" in s):
        for token in s.split():
            if (":\\" in token) or (token.startswith("/") and "/" in token):
                return ok(TaskKind.FS_PATH, "path_token", "句中含路径形态 token，走文件系统路径处理。")
    if inner:
        return ok(
            TaskKind.TOOL_PLAN,
            "planner_tools",
            f"Planner 产出 {len(inner)} 步工具意图，走 TOOL_PLAN。",
            tool_count=len(inner),
        )
    return ok(TaskKind.FULL_PIPELINE, "planner_fallback", "Planner 有残余分支：走完整管线兜底。")


def route_kind(user_text: str, has_context_prefix: bool) -> TaskKind:
    """仅返回 TaskKind（兼容旧调用点测试）。"""
    return infer_dialogue_route(user_text, has_context_prefix, emit=True).kind
