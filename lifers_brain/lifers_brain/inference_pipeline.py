"""
高层推理阶段划分（与常见对话产品类比）：意图/路由 → 上下文装配 → 生成/工具兜底。

stderr 写入 `LIFERS_PROGRESS inference …`（`log_inference`），便于 Agents Chat「执行过程」里对齐完整链路。
与 `dialogue_router` 的 `LIFERS_PROGRESS dialogue_route`、编排器 `LIFERS_PROGRESS taskflow kind=…` 互补，不重复写路由逻辑。

常见 stage：`taskflow_route`（orchestrator）；`agent` 内另有 transformer 起止等细粒度日志。
"""

from __future__ import annotations

import json
import sys
from typing import Any


def log_inference(stage: str, **fields: Any) -> None:
    payload: dict[str, Any] = {"stage": stage}
    for k, v in fields.items():
        if v is not None and v != "":
            payload[k] = v
    sys.stderr.write("LIFERS_PROGRESS inference " + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stderr.flush()
