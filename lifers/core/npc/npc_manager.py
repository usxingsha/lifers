"""
lifers/core/npc/npc_manager.py
──────────────────────────────────────
Orchestrates Persona + StateMachine + SQLite memory for each active NPC.
"""
from __future__ import annotations
import logging, sqlite3, time
from pathlib import Path
from typing import Optional

from .persona       import Persona, PersonaRegistry
from .state_machine import NPCStateMachine

log = logging.getLogger(__name__)


class NPCSession:
    def __init__(self, persona: Persona, db: sqlite3.Connection) -> None:
        self.persona  = persona
        self.sm       = NPCStateMachine(persona.name)
        self._db      = db
        self._history: list[dict] = []

    def build_prompt(self, user_text: str) -> tuple[str, dict]:
        state, hints = self.sm.advance(user_text, self.persona.emotion.valence)
        dv, da = hints.get("emotion_delta", (0.0, 0.0))
        self.persona.emotion.update(dv, da)

        memories = self._retrieve(user_text) if hints.get("retrieve_memory") else ""
        system   = self.persona.system_prompt(extra_context=memories)
        history  = self._fmt_history()
        prefix   = hints.get("prompt_prefix", "")

        prompt = (f"<|system|>\n{system}\n"
                  f"{history}"
                  f"<|user|>\n{prefix}{user_text}\n"
                  f"<|assistant|>\n")

        meta = {"npc_name":  self.persona.name,
                "npc_state": state.value,
                "emotion":   self.persona.emotion.to_dict(),
                "turn":      self.sm.turn_count}
        return prompt, meta

    def record(self, user_text: str, npc_reply: str) -> None:
        self._history += [{"role": "user", "text": user_text},
                          {"role": "npc",  "text": npc_reply}]
        try:
            self._db.execute(
                "INSERT INTO npc_memory(npc_name,user_text,npc_text,ts) VALUES(?,?,?,?)",
                (self.persona.memory_key, user_text, npc_reply, time.time()))
            self._db.commit()
        except Exception as e:
            log.warning("Memory persist failed: %s", e)

    def _fmt_history(self, max_turns: int = 10) -> str:
        turns = self._history[-(max_turns * 2):]
        parts = []
        for t in turns:
            tag = "<|user|>" if t["role"] == "user" else "<|npc|>"
            parts.append(f"{tag}\n{t['text']}")
        return "\n".join(parts) + "\n" if parts else ""

    def _retrieve(self, query: str, k: int = 3) -> str:
        try:
            rows = self._db.execute(
                "SELECT user_text,npc_text FROM npc_memory "
                "WHERE npc_name=? ORDER BY ts DESC LIMIT ?",
                (self.persona.memory_key, k * 3)).fetchall()
            if not rows:
                return ""
            q = set(query.lower().split())
            scored = sorted(rows, key=lambda r: len(q & set(r[0].lower().split())), reverse=True)
            return "\n".join(
                f'[记忆] 你曾说：\u201c{u}\u201d \u2192 我回答：\u201c{n}\u201d'
                for u, n in scored[:k]
            )
        except Exception as e:
            log.warning("Memory retrieval failed: %s", e)
            return ""


class NPCManager:
    """
    Usage
    -----
    mgr  = NPCManager(stack_cfg)
    sess = mgr.get_or_create("Aria")
    prompt, meta = sess.build_prompt(user_text)
    # ... inference ...
    sess.record(user_text, npc_reply)
    """

    def __init__(self, stack_cfg: dict) -> None:
        npc_cfg    = stack_cfg.get("npc", {})
        root       = Path(__file__).parent.parent.parent
        pdir       = root / npc_cfg.get("persona_dir", "config/personas")
        db_path    = root / stack_cfg.get("memory", {}).get("db_path", "memory/lifers.sqlite3")
        self._max  = npc_cfg.get("max_active_npcs", 8)
        self._reg  = PersonaRegistry(pdir)
        self._db   = self._init_db(db_path)
        self._sess: dict[str, NPCSession] = {}

    def get_or_create(self, name: str) -> Optional[NPCSession]:
        key = name.lower()
        if key in self._sess:
            return self._sess[key]
        persona = self._reg.get(key)
        if persona is None:
            log.error("NPC not found: %s", name)
            return None
        if len(self._sess) >= self._max:
            oldest = next(iter(self._sess))
            self._sess[oldest].sm.suspend()
            del self._sess[oldest]
        self._sess[key] = NPCSession(persona, self._db)
        return self._sess[key]

    def suspend(self, name: str) -> None:
        s = self._sess.get(name.lower())
        if s:
            s.sm.suspend()

    def list_active(self) -> list[str]:
        return list(self._sess)

    @staticmethod
    def _init_db(path: Path) -> sqlite3.Connection:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.execute("""CREATE TABLE IF NOT EXISTS npc_memory(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            npc_name TEXT NOT NULL,
            user_text TEXT NOT NULL,
            npc_text  TEXT NOT NULL,
            ts        REAL NOT NULL)""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_npc ON npc_memory(npc_name,ts)")
        conn.commit()
        return conn
