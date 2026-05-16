"""从 config/stack.json#llm_ops 拼出写入本地 LLM 上下文的运维说明（日常 + 权重流水线 + 可选 playbook 文件）。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def format_llm_ops_context(stack: Dict[str, Any], root: Optional[Path] = None) -> str:
    sec = stack.get("llm_ops")
    if not isinstance(sec, dict):
        return ""
    if sec.get("enabled") is False:
        return ""

    chunks: list[str] = []
    title = str(sec.get("title") or "").strip()
    if title:
        chunks.append(title)

    daily = str(sec.get("daily") or "").strip()
    if daily:
        chunks.append("日常对话与工具（默认）:\n" + daily)

    wp = str(sec.get("weights_pipeline") or "").strip()
    if wp:
        chunks.append("权重与流水线（与日常并存；缺文件或要重训时跑脚本）:\n" + wp)

    rel_pb = str(sec.get("daily_ai_playbook_relpath") or "").strip()
    if rel_pb and root is not None:
        try:
            pb_path = (root / rel_pb.replace("\\", "/")).resolve()
            rroot = root.resolve()
            if pb_path.is_file() and str(pb_path).startswith(str(rroot) + os.sep):
                blob = pb_path.read_text(encoding="utf-8", errors="replace").strip()
                if blob:
                    chunks.append("【本地 AI 日常 playbook】\n" + blob)
        except OSError:
            pass

    extra = str(sec.get("extra_for_llm") or "").strip()
    if extra:
        chunks.append(extra)

    if root is not None:
        vm = root / "config" / "lifers_vendor_map.json"
        if vm.is_file():
            try:
                data = json.loads(vm.read_text(encoding="utf-8"))
                primary = str(data.get("primary_product") or "").strip()
                pol = str(data.get("policy_zh") or "").strip()
                tail = f"【vendor 对照 · 以 Lifers 为主】{primary}"
                if pol:
                    tail += f"。{pol}"
                tail += " 详见 config/lifers_vendor_map.json（OpenClaw 子模块 + claw-code/rust vendor）。"
                chunks.append(tail)
            except (OSError, json.JSONDecodeError, TypeError):
                pass

    if not chunks:
        return ""
    text = "\n\n".join(chunks).strip() + "\n"
    try:
        cap = int(sec.get("max_inject_chars", 4000) or 4000)
    except (TypeError, ValueError):
        cap = 4000
    cap = max(200, min(cap, 32000))
    if len(text) > cap:
        return text[: cap - 20].rstrip() + "\n…(llm_ops 已截断)\n"
    return text
