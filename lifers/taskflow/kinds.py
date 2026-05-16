"""任务类型枚举：由分类器产出，由分发器路由到对应处理库。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class TaskKind(str, Enum):
    """单轮用户输入的语义/工具类型（可多标签逻辑，此处取主类型）。"""

    CHAT_QUICK = "chat_quick"
    PLAN_PREVIEW = "plan_preview"
    SMART_SEARCH = "smart_search"
    WORKFLOW_DUAL = "workflow_dual"
    KB_CLI = "kb_cli"
    CMD_SHELL = "cmd_shell"
    SIM_RUN = "sim_run"
    URL_FETCH = "url_fetch"
    WEB_SEARCH = "web_search"
    FS_PATH = "fs_path"
    REAL_WORLD = "real_world"
    TOOL_PLAN = "tool_plan"
    FULL_PIPELINE = "full_pipeline"


@dataclass
class HandlerResult:
    handled: bool
    reply: str
    meta: Dict[str, Any] = field(default_factory=dict)
