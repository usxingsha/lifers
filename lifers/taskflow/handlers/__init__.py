"""各类型处理库注册表 → 构建默认 TaskDispatcher。"""

from __future__ import annotations

from lifers.taskflow.dispatcher import TaskDispatcher
from lifers.taskflow.kinds import TaskKind

from . import (
    chat_quick,
    cmd_shell,
    fs_path,
    full_pipeline,
    kb_cli,
    plan_preview,
    real_world,
    sim_run,
    smart_search,
    tool_plan,
    url_fetch,
    web_search,
    workflow_dual,
)


def build_default_dispatcher() -> TaskDispatcher:
    routes = {
        TaskKind.CHAT_QUICK: chat_quick.handle,
        TaskKind.PLAN_PREVIEW: plan_preview.handle,
        TaskKind.SMART_SEARCH: smart_search.handle,
        TaskKind.WORKFLOW_DUAL: workflow_dual.handle,
        TaskKind.KB_CLI: kb_cli.handle,
        TaskKind.CMD_SHELL: cmd_shell.handle,
        TaskKind.SIM_RUN: sim_run.handle,
        TaskKind.URL_FETCH: url_fetch.handle,
        TaskKind.WEB_SEARCH: web_search.handle,
        TaskKind.FS_PATH: fs_path.handle,
        TaskKind.REAL_WORLD: real_world.handle,
        TaskKind.TOOL_PLAN: tool_plan.handle,
        TaskKind.FULL_PIPELINE: full_pipeline.handle,
    }
    return TaskDispatcher(routes)
