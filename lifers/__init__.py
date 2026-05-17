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

# ── Lifers 10-Pillar Architecture ──────────────────────────────────────────
# Knowledge Graph
from .knowledge_graph import KnowledgeGraph, Entity, Relation, build_world_ontology
# Deep Planner
from .deep_planner import HTNPlanner, MCTS, MCTSNode, ReflectionEngine, Plan, Task
# Audio / Voice
from .audio import LifersASR, LifersTTS, WakeWordDetector, VoicePipeline
# Safety / Guard
from .safety import LifersGuard, ContentFilter, InjectionDetector, Sandbox, AlignmentChecker, LIFERS_CONSTITUTION
# Multi-Agent Swarm
from .multi_agent import Swarm, SwarmAgent, Mailbox, Message, ROLE_SPECS
# Reinforcement Learning
from .rl import PPOTrainer, ContinualLearner, ReplayBuffer, CuriosityModule, ToolEnv
# Robot HAL
from .robot_hal import RobotHAL, Camera, Motor, Lidar, Servo
# Simulation World
from .sim.scenarios import WorldBuilder, ScenarioLibrary, BenchmarkRunner, Scenario
# Telemetry
from .telemetry import LifersPulse, MetricsRegistry, Counter, Gauge, Histogram, Tracer
# Dashboard
from .dashboard import DashboardServer
# Perception
from .perception import (
    PerceptionEngine, VisualAnalyzer, AudioAnalyzer,
    SituationModel, SituationSnapshot, PerceptEvent,
    SimCamera, SimMicrophone,
)
# Proactive
from .proactive import (
    ProactiveAgent, DriveSystem, ThoughtGenerator,
    InterruptionPolicy, IntentQueue, Thought, Intention,
    SimPerception,
)
# Social
from .social import (
    RelationshipLevel, Identity, PersonProfile,
    RelationshipModel, SocialContext, AttachmentSystem,
    SocialBehavior, SocialLearning, SocialBrain,
)

__all__ = [
    # Core (existing)
    "format_llm_ops_context",
    "ToolCall", "ToolResult", "ToolRegistry", "build_default_registry",
    "MarkovWeights", "train_from_text", "generate",
    "TinyTransformerWeights", "train_sgd_minimal", "generate_text",
    "AgentConfig", "LocalBrain",
    "Planner", "LifersAgent", "run_lifers_turn",
    "LongTermMemory", "MemoryItem", "Scratchpad", "SessionMemory",
    "audit_log",
    "EmbodiedCoordinator", "PhysBody", "PhysWorld", "run_embodied_tick",
    "iter_stream_simple_chars", "lifers_turn",
    "DialogueNode", "NpcEmotion", "NpcEngine", "NpcProfile", "NpcState", "match_dialogue_tree",
    "MemoryType",
    "check_health", "emit_health_report", "HealthIssue",
    # Knowledge Graph
    "KnowledgeGraph", "Entity", "Relation", "build_world_ontology",
    # Deep Planner
    "HTNPlanner", "MCTS", "MCTSNode", "ReflectionEngine", "Plan", "Task",
    # Audio / Voice
    "LifersASR", "LifersTTS", "WakeWordDetector", "VoicePipeline",
    # Safety / Guard
    "LifersGuard", "ContentFilter", "InjectionDetector", "Sandbox", "AlignmentChecker", "LIFERS_CONSTITUTION",
    # Multi-Agent
    "Swarm", "SwarmAgent", "Mailbox", "Message", "ROLE_SPECS",
    # RL
    "PPOTrainer", "ContinualLearner", "ReplayBuffer", "CuriosityModule", "ToolEnv",
    # Robot HAL
    "RobotHAL", "Camera", "Motor", "Lidar", "Servo",
    # Simulation
    "WorldBuilder", "ScenarioLibrary", "BenchmarkRunner", "Scenario",
    # Telemetry
    "LifersPulse", "MetricsRegistry", "Counter", "Gauge", "Histogram", "Tracer",
    # Dashboard
    "DashboardServer",
    # Perception
    "PerceptionEngine", "VisualAnalyzer", "AudioAnalyzer",
    "SituationModel", "SituationSnapshot", "PerceptEvent",
    "SimCamera", "SimMicrophone",
    # Proactive
    "ProactiveAgent", "DriveSystem", "ThoughtGenerator",
    "InterruptionPolicy", "IntentQueue", "Thought", "Intention",
    "SimPerception",
    # Social
    "RelationshipLevel", "Identity", "PersonProfile",
    "RelationshipModel", "SocialContext", "AttachmentSystem",
    "SocialBehavior", "SocialLearning", "SocialBrain",
]

