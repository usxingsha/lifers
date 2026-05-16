"""
taskflow/handlers.py
─────────────────────
TaskKind → 处理函数注册表。从各 handler 模块组装 TaskDispatcher。
"""
from __future__ import annotations

from .dispatcher import TaskDispatcher, HandlerFn
from .kinds import TaskKind

from . import (
    chat_quick, full_pipeline, web_search, url_fetch,
    cmd_shell, fs_path, real_world, tool_plan,
    plan_preview, smart_search, workflow_dual,
    kb_cli, sim_run,
)


def build_default_dispatcher() -> TaskDispatcher:
    routes: dict[TaskKind, HandlerFn] = {
        TaskKind.CHAT_QUICK:    chat_quick.handle,
        TaskKind.FULL_PIPELINE: full_pipeline.handle,
        TaskKind.WEB_SEARCH:    web_search.handle,
        TaskKind.URL_FETCH:     url_fetch.handle,
        TaskKind.CMD_SHELL:     cmd_shell.handle,
        TaskKind.FS_PATH:       fs_path.handle,
        TaskKind.REAL_WORLD:    real_world.handle,
        TaskKind.TOOL_PLAN:     tool_plan.handle,
        TaskKind.PLAN_PREVIEW:  plan_preview.handle,
        TaskKind.SMART_SEARCH:  smart_search.handle,
        TaskKind.WORKFLOW_DUAL: workflow_dual.handle,
        TaskKind.KB_CLI:        kb_cli.handle,
        TaskKind.SIM_RUN:       sim_run.handle,
    }
    return TaskDispatcher(routes)
