"""
Planner: dependency-free tool chain planner for LifersAgent.

Extracted from agent.py to keep the agent module focused on orchestration.
Rules:
- ``plan()`` — parse user input into a list of ToolCall steps
- ``plan_real_world_instinct()`` — clock / weather / map triggers at instinct layer
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from lifers.daily_intents import cn_web_query_line, parse_workspace_write_message
from lifers.stack_env import load_stack
from lifers.tools import ToolCall

_PLANNER_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")


def _planner_token_is_image_path(raw: str) -> bool:
    low = raw.strip().lower()
    return any(low.endswith(ext) for ext in _PLANNER_IMAGE_EXTS)


def _planner_rel_path_image_under_lifers_root(raw: str) -> str | None:
    """If the path points to an image under LIFERS_ROOT, return the relative posix path."""
    root = Path(os.environ.get("LIFERS_ROOT", ".")).resolve()
    raw = raw.strip("\"'")
    try:
        p = Path(raw).expanduser()
        p = p.resolve() if p.is_absolute() else (root / raw).resolve()
        rel = p.relative_to(root)
        if not _planner_token_is_image_path(str(p)):
            return None
        return str(rel).replace("\\", "/")
    except (ValueError, OSError):
        return None


class Planner:
    """
    Minimal planner (dependency-free):
    - If input looks like "search ..." or contains a URL -> plan web tools
    - If input mentions file path -> plan fs_read, or vision_digest for images under LIFERS_ROOT
    - Otherwise respond directly
    """

    def plan(self, user_input: str) -> List[ToolCall]:
        text = user_input.strip()
        calls: List[ToolCall] = []
        low = text.lower()
        # Short weather queries are handled by plan_real_world_instinct() — skip here
        # to avoid duplicate tool calls.
        if len(text) < 56 and any(k in text for k in ("天气", "气温", "下雨", "下雪", "温度")) and "search" not in low:
            return []

        # Fixed two-step: local KB then web (always runs both when using this prefix).
        if text.startswith("流程") and len(text) > 2:
            q = text[2:].strip()
            if q:
                calls.append(
                    ToolCall(name="kb_search", args={"query": q, "k": 6}, expected_effect="search long-term memory first", mode="execute")
                )
                calls.append(
                    ToolCall(name="web_search", args={"query": q, "limit": 5}, expected_effect="then search the web", mode="execute")
                )
                return calls
        if low.startswith("workflow "):
            q = text[len("workflow ") :].strip()
            if q:
                calls.append(
                    ToolCall(name="kb_search", args={"query": q, "k": 6}, expected_effect="search long-term memory first", mode="execute")
                )
                calls.append(
                    ToolCall(name="web_search", args={"query": q, "limit": 5}, expected_effect="then search the web", mode="execute")
                )
                return calls

        q_cn = cn_web_query_line(text)
        if q_cn:
            calls.append(
                ToolCall(
                    name="web_search",
                    args={"query": q_cn, "limit": 5},
                    expected_effect="search web (Chinese surface intent)",
                    mode="execute",
                )
            )
            return calls

        ws = parse_workspace_write_message(text)
        if ws:
            rel, body = ws
            calls.append(
                ToolCall(
                    name="lifers_workspace_write",
                    args={"rel_path": rel, "new_text": body},
                    expected_effect="write file under LIFERS_ROOT (self-code allowed)",
                    mode="execute",
                )
            )
            return calls

        if low.startswith("kb_search "):
            q = text[len("kb_search ") :].strip()
            calls.append(
                ToolCall(name="kb_search", args={"query": q, "k": 6}, expected_effect="search KB", mode="execute")
            )
            return calls

        if low.startswith("kb_prune"):
            parts = text.split()
            min_imp = float(parts[1]) if len(parts) >= 2 else 0.15
            days = int(parts[2]) if len(parts) >= 3 else 30
            calls.append(
                ToolCall(
                    name="kb_prune",
                    args={"min_importance": min_imp, "older_than_days": days, "limit": 500},
                    expected_effect="prune KB",
                    mode="execute",
                )
            )
            return calls

        if low.startswith("kb_compact "):
            url = text[len("kb_compact ") :].strip()
            calls.append(
                ToolCall(
                    name="kb_compact",
                    args={"url": url, "k": 6},
                    expected_effect="compact KB",
                    mode="execute",
                )
            )
            return calls

        if low.startswith("sim_run "):
            tid = text[len("sim_run ") :].strip()
            root = Path(os.environ.get("LIFERS_ROOT", ".")).resolve()
            dr = int((load_stack(root).get("robot") or {}).get("default_sim_runs", 10))
            calls.append(
                ToolCall(name="sim_run", args={"task_id": tid, "runs": dr}, expected_effect="run sim task", mode="execute")
            )
            return calls

        if low.startswith("cmd "):
            cmd = text[len("cmd ") :].strip()
            calls.append(ToolCall(name="cmd_run", args={"cmd": cmd}, expected_effect="run command", mode="execute"))
            return calls

        if "http://" in text or "https://" in text:
            url = None
            for token in text.split():
                if token.startswith("http://") or token.startswith("https://"):
                    url = token
                    break
            if url:
                calls.append(ToolCall(name="web_fetch", args={"url": url}, expected_effect="fetch web page", mode="execute"))
                calls.append(ToolCall(name="extract_evidence", args={"text": ""}, expected_effect="extract evidence", mode="execute"))
        if text.lower().startswith("search "):
            q = text[7:].strip()
            calls.append(ToolCall(name="web_search", args={"query": q, "limit": 5}, expected_effect="search web", mode="execute"))
        path_trigger = (":" in text and ("\\" in text or "/" in text)) or any(
            ("/" in t or "\\" in t) and _planner_token_is_image_path(t.strip("\"'")) for t in text.split()
        )
        if path_trigger:
            for token in text.split():
                raw = token.strip("\"'")
                win_abs = ":\\" in raw
                unix_abs = raw.startswith("/") and "/" in raw
                rel_img = ("/" in raw or "\\" in raw) and _planner_token_is_image_path(raw)
                if not (win_abs or unix_abs or rel_img):
                    continue
                if _planner_token_is_image_path(raw):
                    dig = _planner_rel_path_image_under_lifers_root(raw)
                    if dig:
                        calls.append(
                            ToolCall(
                                name="vision_digest",
                                args={"rel_path": dig},
                                expected_effect="image digest under LIFERS_ROOT",
                                mode="execute",
                            )
                        )
                        break
                if win_abs or unix_abs:
                    calls.append(ToolCall(name="fs_read", args={"path": raw}, expected_effect="read file/dir", mode="execute"))
                    break
        return calls

    def plan_real_world_instinct(self, user_input: str) -> List[ToolCall]:
        """
        Instinct layer: time / weather / map — no memorised keywords needed,
        automatically dispatch real_world tool for live data.
        """
        text = user_input.strip()
        if not text:
            return []
        low = text.lower()
        out: List[ToolCall] = []

        if any(k in text for k in ("地图", "导航", "路线", "经纬", "坐标", "定位", "geocode")):
            q = text
            for p in ("搜地图", "查地图", "地图", "导航到", "打开地图", "定位"):
                if text.startswith(p):
                    q = text[len(p) :].strip()
                    break
            if len(q) >= 2:
                out.append(
                    ToolCall(
                        name="real_world",
                        args={"action": "map", "query": q[:400]},
                        expected_effect="地图 / 地点（OpenStreetMap）",
                        mode="execute",
                    )
                )

        if any(k in text for k in ("天气", "气温", "下雨", "下雪", "温度")) or "weather" in low:
            loc = ""
            toks = text.replace("，", " ").replace("。", " ").split()
            for i, w in enumerate(toks):
                if "天气" in w or w in ("气温", "温度"):
                    if i > 0 and 2 <= len(toks[i - 1]) <= 14:
                        loc = toks[i - 1]
                    break
            out.append(
                ToolCall(
                    name="real_world",
                    args={"action": "weather", "location": loc, "query": text[:280]},
                    expected_effect="天气（wttr.in）",
                    mode="execute",
                )
            )

        clk = any(
            k in text
            for k in (
                "几点", "几号", "星期几", "周几", "日期",
                "时区", "当前时间", "现在几点", "现在时间", "什么时间",
            )
        ) or low in ("what time", "current time", "today date", "date today")
        if clk:
            out.insert(
                0,
                ToolCall(name="real_world", args={"action": "clock"}, expected_effect="本机实时时钟", mode="execute"),
            )

        seen: set = set()
        deduped: List[ToolCall] = []
        for c in out:
            key = (c.name, str(c.args.get("action")))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)
        return deduped[:4]
