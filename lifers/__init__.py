from __future__ import annotations
from pathlib import Path

# 确保 .py 模块优先于同名子目录（tools.py > tools/, memory.py > memory/）
_ROOT = Path(__file__).resolve().parent
__path__ = [str(_ROOT)] + [p for p in __path__ if p != str(_ROOT)]

from .llm_ops_context import format_llm_ops_context
from .tools import ToolCall, ToolResult, ToolRegistry, build_default_registry
from .markov_lm import MarkovWeights, generate, train_from_text
from .transformer_lm import TinyTransformerWeights, generate_text, train_sgd_minimal
from .local_brain import AgentConfig, LocalBrain
from .planner import Planner
from .agent import LifersAgent
from .taskflow import run_lifers_turn
from .bridge_turn import iter_stream_simple_chars, lifers_turn
from .memory import LongTermMemory, MemoryItem, Scratchpad, SessionMemory, MemoryType
from .audit import audit_log
from .health import check_health, emit_health_report, HealthIssue
from .embodied import EmbodiedCoordinator, PhysBody, PhysWorld, run_embodied_tick
from .npc_engine import (
    DialogueNode,
    NpcEmotion,
    NpcEngine,
    NpcProfile,
    NpcState,
    match_dialogue_tree,
)

__all__ = [
    "format_llm_ops_context",
    "ToolCall",
    "ToolResult",
    "ToolRegistry",
    "build_default_registry",
    "MarkovWeights",
    "train_from_text",
    "generate",
    "TinyTransformerWeights",
    "train_sgd_minimal",
    "generate_text",
    "AgentConfig",
    "LocalBrain",
    "Planner",
    "LifersAgent",
    "run_lifers_turn",
    "LongTermMemory",
    "MemoryItem",
    "Scratchpad",
    "SessionMemory",
    "audit_log",
    "EmbodiedCoordinator",
    "PhysBody",
    "PhysWorld",
    "run_embodied_tick",
    "iter_stream_simple_chars",
    "lifers_turn",
    "DialogueNode",
    "NpcEmotion",
    "NpcEngine",
    "NpcProfile",
    "NpcState",
    "match_dialogue_tree",
    "MemoryType",
    "check_health",
    "emit_health_report",
    "HealthIssue",
]

