"""
Centralized default paths and constants for the Lifers project.

Single source of truth for file paths, config keys, and limits.
All modules should import from here rather than hardcoding strings.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ── Relative paths (under LIFERS_ROOT) ───────────────────────────────────────

CONFIG_STACK: str = "config/stack.json"
CONFIG_SECRETS: str = "config/secrets.env"
MEMORY_DB: str = "memory/longterm.sqlite3"
AUDIT_LOG: str = "logs/audit.jsonl"

WEIGHTS_MARKOV: str = "weights/lifers_markov.json"
WEIGHTS_TRANSFORMER: str = "weights/lifers_transformer.json"
WEIGHTS_MARKOV_FALLBACK: str = "weights/markov_v001.json"
WEIGHTS_LORA: str = "weights/lifers_lora.json"

STATE_SELF_CODE_QUEUE: str = "state/self_code_queue"

# ── Input safety limits ──────────────────────────────────────────────────────

MAX_INPUT_CHARS: int = 80_000
"""Hard upper bound for user text before routing.  Exceeding returns a message."""
MAX_INPUT_CHARS_WARN: int = 40_000
"""Threshold above which a warning is logged (but still processed)."""

# ── Output safety limits ─────────────────────────────────────────────────────

MAX_OUTPUT_CHARS: int = 16_384
"""Absolute ceiling on generated reply characters."""
MIN_OUTPUT_CHARS: int = 2
"""If generated output is shorter, retry with different temperature."""
OUTPUT_RETRY_TEMPERATURE: float = 0.5
"""Temperature bump for retry on empty/short generation (default +0.2)."""
REPEAT_PENALTY_NGRAM: int = 4
"""N-gram length for detecting repeated phrases in output."""
STATE_SELF_CODE_DONE: str = "state/self_code_done"
STATE_SELF_CODE_ERROR: str = "state/self_code_error"

# ── Default per-backend output limits ────────────────────────────────────────

MAX_OUT_CHARS: Dict[str, int] = {
    "transformer": 4800,
    "markov": 14_000,
}

# ── Stack schema spec (keys whose values must be of a given type) ────────────

STACK_SCHEMA: Dict[str, type] = {
    "version": int,
    "runtime.role": str,
    "brain.model": str,
    "brain.sandbox": (bool, int),
    "brain.max_tool_steps": int,
    "brain.session_max_turns": int,
    "brain.llm_identity_short": str,
    "brain.memory_db": str,
    "remote_infer.enabled": (bool, int),
    "robot.sim_exec_cmd": str,
    "robot.sense_exec_cmd": str,
    "robot.act_exec_cmd": str,
    "embodied_world.dynamic_npc": list,
}

STACK_WEIGHT_PATHS: List[str] = [WEIGHTS_MARKOV, WEIGHTS_TRANSFORMER]

# ── Default weight path lookup order ─────────────────────────────────────────

def default_weight_paths(canonical: str) -> Tuple[str, ...]:
    """Return weight file candidates for a given backend, in priority order."""
    if canonical == "transformer":
        return (WEIGHTS_TRANSFORMER,)
    return (WEIGHTS_MARKOV, WEIGHTS_MARKOV_FALLBACK)
