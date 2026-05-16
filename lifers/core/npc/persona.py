"""
lifers/core/npc/persona.py
─────────────────────────────────
NPC character profile: loads from JSON, builds system prompt, tracks emotion.
"""
from __future__ import annotations
import json, logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class Emotion:
    """2D emotion model: valence (negative↔positive) + arousal (calm↔excited)."""
    valence: float = 0.5
    arousal: float = 0.4
    decay:   float = 0.85

    def update(self, dv: float = 0.0, da: float = 0.0) -> None:
        self.valence = max(0.0, min(1.0,
            self.valence * self.decay + 0.5 * (1 - self.decay) + dv))
        self.arousal = max(0.0, min(1.0,
            self.arousal * self.decay + 0.4 * (1 - self.decay) + da))

    @property
    def label(self) -> str:
        if self.valence > 0.65:
            return "happy" if self.arousal > 0.5 else "content"
        if self.valence < 0.35:
            return "angry" if self.arousal > 0.5 else "sad"
        return "neutral"

    def to_dict(self) -> dict:
        return {"valence": round(self.valence, 3),
                "arousal": round(self.arousal, 3),
                "label":   self.label}


@dataclass
class Persona:
    name:         str
    role:         str        = "assistant"
    language:     str        = "zh"
    backstory:    str        = ""
    personality:  list[str]  = field(default_factory=list)
    speech_style: str        = ""
    emotion:      Emotion    = field(default_factory=Emotion)
    memory_key:   str        = ""

    @classmethod
    def from_file(cls, path: Path) -> "Persona":
        with path.open(encoding="utf-8") as f:
            d = json.load(f)
        emo = d.get("initial_emotion", {})
        return cls(
            name         = d["name"],
            role         = d.get("role", "assistant"),
            language     = d.get("language", "zh"),
            backstory    = d.get("backstory", ""),
            personality  = d.get("personality", []),
            speech_style = d.get("speech_style", ""),
            emotion      = Emotion(valence=emo.get("valence", 0.5),
                                   arousal=emo.get("arousal", 0.4)),
            memory_key   = d.get("memory_key", d["name"].lower()),
        )

    def system_prompt(self, extra_context: str = "") -> str:
        traits = "、".join(self.personality) or "neutral"
        emo    = f"当前情绪：{self.emotion.label}"
        lines  = [
            f"你正在扮演 [{self.name}]，身份是 {self.role}。",
            f"性格：{traits}。",
            f"说话风格：{self.speech_style}。" if self.speech_style else "",
            f"背景：{self.backstory}"          if self.backstory    else "",
            emo,
            "请始终保持角色一致，不要破坏第四面墙。",
            extra_context,
        ]
        return "\n".join(l for l in lines if l)

    def to_dict(self) -> dict:
        return {"name": self.name, "role": self.role,
                "emotion": self.emotion.to_dict()}


class PersonaRegistry:
    def __init__(self, persona_dir: Path) -> None:
        self._dir   = persona_dir
        self._cache: dict[str, Persona] = {}
        self._scan()

    def _scan(self) -> None:
        if not self._dir.exists():
            log.warning("Persona dir not found: %s", self._dir)
            return
        for p in self._dir.glob("*.json"):
            try:
                persona = Persona.from_file(p)
                self._cache[persona.name.lower()] = persona
                log.info("Loaded persona: %s", persona.name)
            except Exception as e:
                log.error("Failed to load %s: %s", p, e)

    def get(self, name: str) -> Optional[Persona]:
        return self._cache.get(name.lower())

    def list_names(self) -> list[str]:
        return sorted(self._cache)

    def reload(self) -> None:
        self._cache.clear()
        self._scan()
