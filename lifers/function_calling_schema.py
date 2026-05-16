"""轻量 JSON Schema 子集校验 + 工具规格 → OpenAI-style function 描述 + 从文本抽取 JSON 块。

不依赖 jsonschema 包；用于结构化工具参数校验与可选「函数式」回复解析。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from lifers.tools import ToolSpec


def validate_object(obj: Any, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    schema 子集：
    - type: "object"
    - properties: { key: { "type": "string|number|integer|boolean|object|array|any" } }
    - required: [keys]
    """
    errs: list[str] = []
    if schema.get("type") != "object" or not isinstance(obj, dict):
        return False, ["root must be object"]
    req = schema.get("required") or []
    if isinstance(req, list):
        for k in req:
            if k not in obj:
                errs.append(f"missing required: {k}")
    props = schema.get("properties") or {}
    if not isinstance(props, dict):
        return len(errs) == 0, errs
    for k, v in obj.items():
        spec = props.get(k)
        if spec is None:
            continue
        if not isinstance(spec, dict):
            continue
        want = str(spec.get("type", "any")).lower()
        if want == "any":
            continue
        if want == "string" and not isinstance(v, str):
            errs.append(f"{k}: expected string")
        elif want == "number" and not isinstance(v, (int, float)):
            errs.append(f"{k}: expected number")
        elif want == "integer" and not isinstance(v, int):
            errs.append(f"{k}: expected integer")
        elif want == "boolean" and not isinstance(v, bool):
            errs.append(f"{k}: expected boolean")
        elif want == "object" and not isinstance(v, dict):
            errs.append(f"{k}: expected object")
        elif want == "array" and not isinstance(v, list):
            errs.append(f"{k}: expected array")
    return len(errs) == 0, errs


def tool_spec_to_openai_function(spec: ToolSpec) -> Dict[str, Any]:
    """将内部 ToolSpec 转为 chat.completions tools[] 风格的 function 条目（仅元数据，不含执行器）。"""
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": f"Lifers tool `{spec.name}` risk={spec.risk_level}",
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "string", "description": str(v)} for k, v in (spec.args_schema or {}).items()},
                "required": [],
            },
        },
    }


_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def extract_json_values(text: str) -> List[Any]:
    """从模型或用户文本中尽量抽出 JSON 对象（代码块优先，其次花括号平衡片段）。"""
    out: list[Any] = []
    for m in _JSON_FENCE.finditer(text or ""):
        block = m.group(1).strip()
        if not block:
            continue
        try:
            out.append(json.loads(block))
        except json.JSONDecodeError:
            continue
    if out:
        return out
    s = (text or "").strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            out.append(json.loads(s))
        except json.JSONDecodeError:
            pass
    return out


def parse_tool_invocation(obj: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    识别 {"tool":"web_search","arguments":{...}} 或 {"name":"...","arguments":{}}。
    """
    if not isinstance(obj, dict):
        return None
    name = obj.get("tool") or obj.get("name") or obj.get("function")
    if not isinstance(name, str) or not name.strip():
        return None
    args = obj.get("arguments") or obj.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    return name.strip(), dict(args)
