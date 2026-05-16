"""
lifers/core/npc/state_machine.py
────────────────────────────────────────
Per-NPC dialogue state machine.

States:  IDLE → GREETING → ENGAGED ↔ RECALL / EMOTIONAL → CLOSING → IDLE
                                                         ↕ SUSPENDED
"""
from __future__ import annotations
import re, time, logging
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger(__name__)


class NPCState(str, Enum):
    IDLE      = "idle"
    GREETING  = "greeting"
    ENGAGED   = "engaged"
    RECALL    = "recall"
    EMOTIONAL = "emotional"
    CLOSING   = "closing"
    SUSPENDED = "suspended"


_FAREWELL = re.compile(r"\b(再见|bye|goodbye|结束对话|离开|later)\b",         re.I)
_MEMORY   = re.compile(r"\b(记得|remember|之前|以前|上次|last time|recall)\b", re.I)
_ANGER    = re.compile(r"\b(笨蛋|stupid|idiot|shut up|闭嘴|废物|useless)\b",  re.I)


@dataclass
class Transition:
    from_state: NPCState
    to_state:   NPCState
    trigger:    str
    ts:         float = field(default_factory=time.time)


class NPCStateMachine:
    """
    Usage
    -----
    sm = NPCStateMachine("Aria")
    state, hints = sm.advance(user_text, emotion_valence=0.3)
    """

    def __init__(self, npc_name: str) -> None:
        self.npc_name   = npc_name
        self.state      = NPCState.IDLE
        self.turn_count = 0
        self.history: list[Transition] = []

    def advance(self, user_text: str, emotion_valence: float = 0.5) -> tuple[NPCState, dict]:
        self.turn_count += 1
        hints: dict = {"prompt_prefix": "", "retrieve_memory": False, "emotion_delta": (0.0, 0.0)}

        # IDLE → GREETING
        if self.state == NPCState.IDLE:
            self._go(NPCState.GREETING, "first_message")
            hints["prompt_prefix"] = f"[{self.npc_name} 初次见面，进行自我介绍]"
            return self.state, hints

        # GREETING → ENGAGED
        if self.state == NPCState.GREETING:
            self._go(NPCState.ENGAGED, "greeted")
            return self.state, hints

        # SUSPENDED → ENGAGED
        if self.state == NPCState.SUSPENDED:
            self._go(NPCState.ENGAGED, "resumed")
            hints["prompt_prefix"] = f"[{self.npc_name} 从暂停状态恢复]"
            return self.state, hints

        # EMOTIONAL cool-down
        if self.state == NPCState.EMOTIONAL:
            self._go(NPCState.ENGAGED, "cooldown")

        # ENGAGED / RECALL triggers
        if self.state in (NPCState.ENGAGED, NPCState.RECALL):
            if _FAREWELL.search(user_text):
                self._go(NPCState.CLOSING, "farewell")
                hints["prompt_prefix"] = f"[{self.npc_name} 正在告别]"
                return self.state, hints
            if _MEMORY.search(user_text):
                self._go(NPCState.RECALL, "memory_trigger")
                hints["retrieve_memory"] = True
                hints["prompt_prefix"]   = f"[{self.npc_name} 正在回想…]"
                return self.state, hints
            if emotion_valence < 0.25 or _ANGER.search(user_text):
                self._go(NPCState.EMOTIONAL, "anger")
                hints["prompt_prefix"] = f"[{self.npc_name} 情绪激动]"
                hints["emotion_delta"] = (-0.25, 0.35)
                return self.state, hints
            if self.state == NPCState.RECALL:
                self._go(NPCState.ENGAGED, "recall_done")

        # CLOSING → IDLE
        if self.state == NPCState.CLOSING:
            self._go(NPCState.IDLE, "conversation_ended")

        return self.state, hints

    def suspend(self) -> None:
        if self.state != NPCState.IDLE:
            self._go(NPCState.SUSPENDED, "explicit_suspend")

    def reset(self) -> None:
        self.state      = NPCState.IDLE
        self.turn_count = 0
        self.history.clear()

    def _go(self, new: NPCState, trigger: str) -> None:
        log.debug("[%s] %s→%s (%s)", self.npc_name, self.state.value, new.value, trigger)
        self.history.append(Transition(self.state, new, trigger))
        self.state = new

    def summary(self) -> dict:
        return {"npc": self.npc_name, "state": self.state.value,
                "turns": self.turn_count,
                "history": [(t.from_state.value, t.to_state.value, t.trigger)
                            for t in self.history[-5:]]}
