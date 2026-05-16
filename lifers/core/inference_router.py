"""
lifers/core/inference_router.py
──────────────────────────────────────
Routes cleaned input to: LOCAL / REMOTE / TOOL / NPC
Intent classification uses CJK-safe regex (no \b on CJK chars).
"""
from __future__ import annotations
import json, logging, os, re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class Route(str, Enum):
    LOCAL  = "local"
    REMOTE = "remote"
    TOOL   = "tool"
    NPC    = "npc"


class Intent(str, Enum):
    CHAT     = "chat"
    CODE     = "code"
    TOOL_USE = "tool_use"
    NPC      = "npc"
    TRAIN    = "train"
    META     = "meta"
    VOICE    = "voice"
    SKILL    = "skill"


# CJK-safe: no \b around CJK tokens
_INTENT_RULES: list[tuple[Intent, re.Pattern]] = [
    (Intent.CODE,     re.compile(r"(\b(code|script|debug|python|rust|def |class )\b|函数|写一个|写个|编写)", re.I)),
    (Intent.TOOL_USE, re.compile(r"(\b(search|find|open|run|execute|ls |cat )\b|搜索|查找|列出|打开|执行)", re.I)),
    (Intent.VOICE,    re.compile(r"(\b(voice|speak|tts|say|read aloud)\b|语音|朗读|播报|声音)", re.I)),
    (Intent.SKILL,    re.compile(r"(\b(skill|ability|use skill|cast)\b|技能|使用技能|施放)", re.I)),
    (Intent.NPC,      re.compile(r"(\b(npc|character|speak to|talk to)\b|角色|对话|扮演)", re.I)),
    (Intent.TRAIN,    re.compile(r"(\b(train|fine.?tune|escalate)\b|训练|更新权重|微调)", re.I)),
    (Intent.META,     re.compile(r"(\b(what are you|how do you work)\b|你是谁|系统|stack|配置)", re.I)),
]


def classify_intent(text: str) -> Intent:
    for intent, pattern in _INTENT_RULES:
        if pattern.search(text):
            return intent
    return Intent.CHAT


@dataclass
class RouteDecision:
    route:   Route
    intent:  Intent
    backend: str
    meta:    dict


class InferenceRouter:
    def __init__(self, stack_path: Optional[Path] = None) -> None:
        if stack_path is None:
            stack_path = Path(__file__).parent.parent / "config" / "stack.json"
        self._stack = self._load(stack_path)

    def route(self, text: str, context: Optional[dict] = None) -> RouteDecision:
        ctx    = context or {}
        intent = classify_intent(text)

        if ctx.get("active_npc") and intent not in (Intent.TOOL_USE, Intent.VOICE, Intent.SKILL):
            return RouteDecision(Route.NPC, intent,
                                 f"npc:{ctx['active_npc']}",
                                 {"npc_name": ctx["active_npc"]})

        if intent == Intent.NPC and self._stack.get("npc", {}).get("enabled"):
            return RouteDecision(Route.NPC, intent, "npc:auto", {})

        if intent in (Intent.TOOL_USE, Intent.VOICE, Intent.SKILL):
            return RouteDecision(Route.TOOL, intent, f"agent_tools:{intent.value}", {})

        r = self._stack.get("remote_infer", {})
        if r.get("enabled") and os.getenv(r.get("env_key", "LIFERS_API_KEY")):
            return RouteDecision(Route.REMOTE, intent,
                                 r.get("provider", "remote"),
                                 {"endpoint": r.get("endpoint")})

        m = self._stack.get("model", {})
        return RouteDecision(
            Route.LOCAL, intent,
            m.get("name", "lifers_transformer"),
            {"weights_path": m.get("weights_path"),
             "temperature":  m.get("temperature", 0.7),
             "top_p":        m.get("top_p", 0.9),
             "max_tokens":   m.get("max_tokens", 2048),
             "stream":       m.get("stream", True)},
        )

    @staticmethod
    def _load(path: Path) -> dict:
        if not path.exists():
            log.warning("stack.json not found at %s", path)
            return {}
        with path.open(encoding="utf-8") as f:
            return json.load(f)
