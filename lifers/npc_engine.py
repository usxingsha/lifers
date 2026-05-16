"""
NPC (Non-Player Character) support for Lifers.

Provides:
- ``NpcProfile`` — character definition (name, persona, backstory, voice)
- ``NpcState`` — per-session emotional state & relationship tracking
- ``DialogueNode`` — simple dialogue tree for scripted branches
- ``NpcEngine`` — combine profile + state + tree into ``quick_chat()`` compatible context

Integration with LifersAgent:
    stack.embodied_world.dynamic_npc → NpcProfile[] loaded at startup.
    Each turn, the active NPC's state is injected into INSTINCT_AUTONOMIC
    so the local LM can react in-character.

Designed for edge deployment: no external dependencies, pure data-driven.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Emotion model ────────────────────────────────────────────────────────────

@dataclass
class NpcEmotion:
    """Simple 2D emotion model: valence (−1..+1) × arousal (0..+1)."""
    valence: float = 0.0   # −1 (sad/hostile)  →  +1 (happy/friendly)
    arousal: float = 0.3   #  0 (calm/sleepy)   →  +1 (excited/angry)

    def decay(self, factor: float = 0.95) -> None:
        self.valence *= factor
        self.arousal = max(0.0, self.arousal * factor)

    def mood_label(self) -> str:
        if self.arousal < 0.3:
            return "平静" if self.valence >= 0 else "低落"
        if self.valence > 0.3:
            return "愉快" if self.arousal > 0.5 else "放松"
        if self.valence < -0.3:
            return "恼怒" if self.arousal > 0.5 else "不悦"
        return "中性"


# ── Character profile ────────────────────────────────────────────────────────

@dataclass
class NpcProfile:
    """Static character definition (loaded from stack.json or NPC config file)."""
    name: str
    persona: str              # e.g. "友好的书店老板"
    backstory: str = ""       # short biographic context
    voice: str = ""           # speaking style hint, e.g. "轻声细语，爱用古诗词"
    greeting: str = ""        # first-encounter dialog line
    portrait_emoji: str = ""  # optional visual marker
    dialogue_root: Optional[DialogueNode] = None  # optional dialogue tree root
    _greeted: bool = False    # whether the greeting has been delivered this session

    def to_dict(self) -> Dict[str, Any]:
        """Serialize profile (without dialogue tree) for persistence."""
        d: Dict[str, Any] = {
            "name": self.name,
            "persona": self.persona,
            "backstory": self.backstory,
            "voice": self.voice,
            "greeting": self.greeting,
            "portrait_emoji": self.portrait_emoji,
        }
        if self.dialogue_root:
            d["dialogue_root"] = self.dialogue_root.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NpcProfile:
        """Deserialize profile, optionally restoring the dialogue tree."""
        root = None
        if "dialogue_root" in data and isinstance(data["dialogue_root"], dict):
            root = DialogueNode.from_dict(data["dialogue_root"])
        return cls(
            name=str(data["name"]),
            persona=str(data.get("persona", "")),
            backstory=str(data.get("backstory", "")),
            voice=str(data.get("voice", "")),
            greeting=str(data.get("greeting", "")),
            portrait_emoji=str(data.get("portrait_emoji", "")),
            dialogue_root=root,
        )


# ── Dialogue tree ────────────────────────────────────────────────────────────

@dataclass
class DialogueNode:
    """A single node in the NPC's dialogue tree."""
    id: str
    text: str                # NPC says this
    keywords: List[str] = field(default_factory=list)  # user must match one
    children: List[DialogueNode] = field(default_factory=list)
    is_fallback: bool = False       # catch-all when no keyword matches
    min_relationship: float = -1.0  # minimum relationship required (−1..+1)
    max_relationship: float = 1.0   # maximum relationship allowed (−1..+1)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this node and its subtree to a JSON-safe dict."""
        return {
            "id": self.id,
            "text": self.text,
            "keywords": list(self.keywords),
            "is_fallback": self.is_fallback,
            "min_relationship": self.min_relationship,
            "max_relationship": self.max_relationship,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DialogueNode:
        """Deserialize a dialogue tree from a dict (produced by ``to_dict``)."""
        children = [cls.from_dict(c) for c in data.get("children", [])]
        return cls(
            id=str(data["id"]),
            text=str(data.get("text", "")),
            keywords=list(data.get("keywords", [])),
            children=children,
            is_fallback=bool(data.get("is_fallback", False)),
            min_relationship=float(data.get("min_relationship", -1.0)),
            max_relationship=float(data.get("max_relationship", 1.0)),
        )


def _relationship_in_range(node: DialogueNode, relationship: float) -> bool:
    """Check whether *relationship* falls within the node's required range."""
    return node.min_relationship <= relationship <= node.max_relationship


def match_dialogue_tree(
    root: DialogueNode,
    user_input: str,
    relationship: float = 0.0,
    prefer_branch: str | None = None,
) -> DialogueNode | None:
    """Walk dialogue tree by keyword matching.

    Parameters
    ----------
    root : DialogueNode
        Root node of the dialogue tree.
    user_input : str
        The user's utterance.
    relationship : float, optional
        Current relationship value (−1..+1) used to gate relationship-gated nodes.
    prefer_branch : str | None, optional
        Node ID of the previously visited node.  When set, the walker checks
        children of *prefer_branch* first, enabling multi-turn continuations.

    Returns the matching leaf, a fallback node, or ``None``.
    """
    low = user_input.lower()

    # Phase 1: if we have a preferred branch, try its children first
    if prefer_branch:
        preferred = _find_by_id(root, prefer_branch)
        if preferred and preferred.children:
            match = _walk_children(preferred, low, relationship)
            if match:
                return match

    # Phase 2: normal walk from root
    match = _walk_children(root, low, relationship)
    if match:
        return match

    # Phase 3: try fallback (respecting relationship gating)
    fallbacks = [c for c in root.children if c.is_fallback and _relationship_in_range(c, relationship)]
    if fallbacks:
        return fallbacks[0]
    return None


def _walk_children(parent: DialogueNode, low: str, relationship: float) -> DialogueNode | None:
    """Walk the children of *parent* looking for a keyword match."""
    for child in parent.children:
        if child.is_fallback or not _relationship_in_range(child, relationship):
            continue
        if any(kw in low for kw in child.keywords):
            if child.children:
                deeper = _walk_children(child, low, relationship)
                if deeper is not None:
                    return deeper
            return child
    return None


def _find_by_id(root: DialogueNode, node_id: str) -> DialogueNode | None:
    """DFS search for a node by ID."""
    if root.id == node_id:
        return root
    for child in root.children:
        found = _find_by_id(child, node_id)
        if found:
            return found
    return None


# ── State engine ─────────────────────────────────────────────────────────────

@dataclass
class NpcState:
    """Mutable per-session NPC state, persisted to ``state/npc_{name}.json``."""
    profile: NpcProfile
    emotion: NpcEmotion = field(default_factory=NpcEmotion)
    turn_count: int = 0
    last_interaction_ts: float = 0.0
    relationship: float = 0.0  # −1 (hostile) → +1 (close)
    dialogue_history: List[str] = field(default_factory=list)
    last_node_id: Optional[str] = None  # last matched dialogue node (multi-turn)

    def context_line(self) -> str:
        mood = self.emotion.mood_label()
        parts = [
            f"【NPC 状态】{self.profile.name}（{self.profile.persona}）",
            f"情绪: {mood}（valence={self.emotion.valence:.2f}, arousal={self.emotion.arousal:.2f}）",
            f"好感度: {self.relationship:.2f}",
        ]
        if self.profile.voice:
            parts.append(f"语气风格: {self.profile.voice}")
        if self.dialogue_history:
            recent = self.dialogue_history[-3:]
            parts.append("最近对话:" + " ".join(recent))
        if self.last_node_id:
            parts.append(f"上次对话节点: {self.last_node_id}")
        return " | ".join(parts)

    def react(self, user_text: str, tool_result_ok: bool = True) -> None:
        """Update emotion based on interaction outcome."""
        self.turn_count += 1
        self.last_interaction_ts = time.time()
        # Simple heuristic: successful interaction → positive valence
        if tool_result_ok:
            self.emotion.valence = min(1.0, self.emotion.valence + 0.05)
            self.relationship = min(1.0, self.relationship + 0.03)
        else:
            self.emotion.valence = max(-1.0, self.emotion.valence - 0.03)
            self.relationship = max(-1.0, self.relationship - 0.02)
        self.emotion.decay(0.98)


# ── Engine ───────────────────────────────────────────────────────────────────

@dataclass
class NpcEngine:
    """Aggregate NPC system: load profiles, manage states, inject context."""

    states: Dict[str, NpcState] = field(default_factory=dict)

    # ── Factory / loading ─────────────────────────────────────────────────

    @classmethod
    def from_stack(cls, stack: Dict[str, Any], root: Path) -> NpcEngine:
        """Load NPC profiles from ``stack.embodied_world.dynamic_npc`` and ``config/npcs/*.json``."""
        eng = cls()
        raw = (
            (stack.get("embodied_world") or {}).get("dynamic_npc") or []
        )
        if not isinstance(raw, list):
            raw = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            eng._load_single_profile(item, root)
        # Also load from config/npcs/*.json (overrides stack entries by name)
        eng._load_config_dir(root)
        return eng

    @classmethod
    def load_npc_configs(cls, root: Path) -> Dict[str, Dict[str, Any]]:
        """Load NPC definitions from ``config/npcs/*.json`` files.

        Returns a dict keyed by NPC name with merged config values.
        Each file may define one NPC (top-level object) or multiple
        (top-level array of objects).  Later files override earlier ones.
        """
        configs: Dict[str, Dict[str, Any]] = {}
        npc_dir = root / "config" / "npcs"
        if not npc_dir.is_dir():
            return configs
        for fp in sorted(npc_dir.iterdir()):
            if fp.suffix.lower() != ".json":
                continue
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(data, dict):
                name = data.get("name")
                if name:
                    configs[str(name)] = data
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        name = item.get("name")
                        if name:
                            configs[str(name)] = item
        return configs

    def _load_single_profile(self, item: Dict[str, Any], root: Path) -> None:
        """Create or restore a single NPC state from a config dict."""
        profile = NpcProfile(
            name=str(item.get("name", "NPC")),
            persona=str(item.get("persona", "")),
            backstory=str(item.get("backstory", "")),
            voice=str(item.get("voice", "")),
            greeting=str(item.get("greeting", "")),
            portrait_emoji=str(item.get("portrait_emoji", "")),
        )
        # Restore dialogue_root from item if present
        if "dialogue_root" in item and isinstance(item["dialogue_root"], dict):
            profile.dialogue_root = DialogueNode.from_dict(item["dialogue_root"])

        state_path = root / "state" / f"npc_{profile.name}.json"
        tree_path = root / "state" / f"npc_{profile.name}_tree.json"
        if state_path.is_file():
            try:
                data = json.loads(state_path.read_text(encoding="utf-8"))
                emo = data.get("emotion", {})
                last_node = data.get("last_node_id")
                state = NpcState(
                    profile=profile,
                    emotion=NpcEmotion(valence=emo.get("v", 0.0), arousal=emo.get("a", 0.3)),
                    turn_count=int(data.get("turn_count", 0)),
                    last_interaction_ts=float(data.get("last_interaction_ts", 0.0)),
                    relationship=float(data.get("relationship", 0.0)),
                    dialogue_history=list(data.get("dialogue_history", [])),
                    last_node_id=str(last_node) if last_node else None,
                )
                # Restore dialogue tree from separate file if available
                if tree_path.is_file():
                    try:
                        tree_data = json.loads(tree_path.read_text(encoding="utf-8"))
                        state.profile.dialogue_root = DialogueNode.from_dict(tree_data)
                    except (json.JSONDecodeError, OSError):
                        pass
                self.states[profile.name] = state
                return
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        self.states[profile.name] = NpcState(profile=profile)

    def _load_config_dir(self, root: Path) -> None:
        """Merge NPC definitions from ``config/npcs/*.json`` into existing states."""
        configs = self.load_npc_configs(root)
        for name, item in configs.items():
            if name in self.states:
                # Update existing state's profile fields from config
                st = self.states[name]
                st.profile.persona = str(item.get("persona", st.profile.persona))
                st.profile.backstory = str(item.get("backstory", st.profile.backstory))
                st.profile.voice = str(item.get("voice", st.profile.voice))
                st.profile.greeting = str(item.get("greeting", st.profile.greeting))
                st.profile.portrait_emoji = str(item.get("portrait_emoji", st.profile.portrait_emoji))
                if "dialogue_root" in item and isinstance(item["dialogue_root"], dict):
                    st.profile.dialogue_root = DialogueNode.from_dict(item["dialogue_root"])
            else:
                self._load_single_profile(item, root)

    # ── Context / inference helpers ───────────────────────────────────────

    def active_context_lines(self, active_name: str | None = None) -> List[str]:
        """Return INSTINCT_AUTONOMIC lines for all (or one) NPC states."""
        if active_name:
            st = self.states.get(active_name)
            return [st.context_line()] if st else []
        return [s.context_line() for s in self.states.values()]

    def detect_active_npc(self, user_text: str) -> Optional[str]:
        """Return the name of the NPC being addressed, or None if unclear."""
        if not self.states:
            return None
        low = user_text.lower()
        for name in self.states:
            if name.lower() in low:
                return name
        return None

    def dialogue_match(
        self,
        npc_name: str,
        user_text: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Walk the NPC's dialogue tree with multi-turn and relationship support.

        Returns ``(node_id, npc_reply)`` or ``(None, None)``.
        Updates ``last_node_id`` on the NPC state so the next call
        can prefer continuing the current branch.
        """
        st = self.states.get(npc_name)
        if not st or not st.profile.dialogue_root:
            return None, None
        node = match_dialogue_tree(
            st.profile.dialogue_root,
            user_text,
            relationship=st.relationship,
            prefer_branch=st.last_node_id,
        )
        if node:
            st.last_node_id = node.id
            return node.id, node.text
        return None, None

    def greeting_for(self, npc_name: str) -> Optional[str]:
        """Return the NPC's greeting text if they haven't been greeted yet.

        Marks the greeting as delivered so it won't repeat.
        """
        st = self.states.get(npc_name)
        if not st or not st.profile.greeting:
            return None
        if st.profile._greeted:
            return None
        st.profile._greeted = True
        return st.profile.greeting

    def all_npc_names(self) -> List[str]:
        """Return sorted list of all NPC names."""
        return sorted(self.states.keys())

    # ── Persistence ───────────────────────────────────────────────────────

    def save_all(self, root: Path) -> None:
        """Persist all NPC states (emotion, relationship, history, last_node) to ``state/npc_*.json``."""
        for name, st in self.states.items():
            state_path = root / "state" / f"npc_{name}.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            payload: Dict[str, Any] = {
                "name": name,
                "persona": st.profile.persona,
                "emotion": {"v": st.emotion.valence, "a": st.emotion.arousal},
                "turn_count": st.turn_count,
                "last_interaction_ts": st.last_interaction_ts,
                "relationship": st.relationship,
                "dialogue_history": st.dialogue_history[-16:],
            }
            if st.last_node_id is not None:
                payload["last_node_id"] = st.last_node_id
            state_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            # Persist dialogue tree separately
            if st.profile.dialogue_root:
                tree_path = root / "state" / f"npc_{name}_tree.json"
                tree_path.write_text(
                    json.dumps(st.profile.dialogue_root.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    def save_dialogue_trees(self, root: Path) -> None:
        """Save all dialogue trees to ``config/npcs/*.json`` for sharing/editing."""
        for name, st in self.states.items():
            if not st.profile.dialogue_root:
                continue
            npc_dir = root / "config" / "npcs"
            npc_dir.mkdir(parents=True, exist_ok=True)
            path = npc_dir / f"{name}.json"
            profile_dict = st.profile.to_dict()
            path.write_text(
                json.dumps(profile_dict, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def load_dialogue_trees(self, root: Path) -> int:
        """Load dialogue trees from ``config/npcs/*.json`` into existing states.

        Returns the number of trees loaded.
        """
        count = 0
        configs = self.load_npc_configs(root)
        for name, item in configs.items():
            st = self.states.get(name)
            if st is None:
                continue
            if "dialogue_root" in item and isinstance(item["dialogue_root"], dict):
                st.profile.dialogue_root = DialogueNode.from_dict(item["dialogue_root"])
                count += 1
        return count
