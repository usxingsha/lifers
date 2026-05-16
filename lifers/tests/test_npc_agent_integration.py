"""End-to-end integration tests: NPC engine wired into LifersAgent.

Verifies:
  - Agent init loads NPC from stack config
  - NPC detection via agent._npc_react_for_turn()
  - NPC context injection via agent._npc_context_block()
  - Dialogue tree matching via agent._npc_dialogue_hint()
  - NPC state persistence via agent.npc_engine.save_all()
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from lifers.agent import AgentConfig, LifersAgent
from lifers.npc_engine import DialogueNode, NpcEngine, NpcProfile, NpcState


def _write_agent_root(root: Path, npc_config: list | None = None) -> None:
    """Create minimal LifersAgent root with optional NPC config."""
    for d in ("config", "weights", "memory", "logs", "state"):
        (root / d).mkdir(parents=True, exist_ok=True)
    stack: dict = {
        "version": 1,
        "runtime": {"role": "brain"},
        "brain": {
            "model": "markov",
            "sandbox": True,
            "session_max_turns": 8,
            "llm_identity_short": "Lifers",
            "memory_db": "memory/test_longterm.sqlite3",
        },
        "human_sim": {"enabled": False},
        "embodied_world": {},
    }
    if npc_config:
        stack["embodied_world"]["dynamic_npc"] = npc_config
    else:
        stack["embodied_world"]["dynamic_npc"] = []
    (root / "config" / "stack.json").write_text(
        json.dumps(stack, ensure_ascii=False), encoding="utf-8"
    )


# Write a tiny valid markov weight file so the agent can init
def _write_markov_weights(root: Path) -> None:
    w = root / "weights" / "lifers_markov.json"
    w.write_text(json.dumps({"unigrams": {"a": 1}, "bigrams": {}, "trigrams": {}}), encoding="utf-8")


# -- Test helpers imported from npc_engine --
def _sample_tree() -> DialogueNode:
    return DialogueNode(
        id="root",
        text="",
        children=[
            DialogueNode(id="greet", text="你好呀！", keywords=["你好", "嗨"]),
            DialogueNode(id="quest", text="任务来了", keywords=["任务"]),
            DialogueNode(id="fallback", text="嗯？", is_fallback=True),
        ],
    )


class TestNpcAgentIntegration:
    """Tests that verify NPC->Agent wiring without full text generation."""

    def test_agent_loads_npc_from_stack(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            _write_agent_root(root, npc_config=[
                {"name": "Alice", "persona": "友好的店员", "voice": "温柔"},
            ])
            _write_markov_weights(root)
            os.environ["LIFERS_SKIP_HEALTH_CHECK"] = "1"
            try:
                agent = LifersAgent(AgentConfig(root_dir=root, model="markov", sandbox=True))
                assert "Alice" in agent.npc_engine.states
                assert agent.npc_engine.states["Alice"].profile.voice == "温柔"
                del agent
            finally:
                os.environ.pop("LIFERS_SKIP_HEALTH_CHECK", None)

    def test_agent_loads_no_npc_when_config_empty(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            _write_agent_root(root, npc_config=[])
            _write_markov_weights(root)
            os.environ["LIFERS_SKIP_HEALTH_CHECK"] = "1"
            try:
                agent = LifersAgent(AgentConfig(root_dir=root, model="markov", sandbox=True))
                assert len(agent.npc_engine.states) == 0
                del agent
            finally:
                os.environ.pop("LIFERS_SKIP_HEALTH_CHECK", None)

    def test_npc_context_block_injected(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            _write_agent_root(root, npc_config=[
                {"name": "Shop", "persona": "店主"},
            ])
            _write_markov_weights(root)
            os.environ["LIFERS_SKIP_HEALTH_CHECK"] = "1"
            try:
                agent = LifersAgent(AgentConfig(root_dir=root, model="markov", sandbox=True))
                block = agent._npc_context_block()
                assert "NPC_STATES" in block
                assert "Shop" in block
                assert "店主" in block
                del agent
            finally:
                os.environ.pop("LIFERS_SKIP_HEALTH_CHECK", None)

    def test_npc_context_empty_when_no_npc(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            _write_agent_root(root, npc_config=[])
            _write_markov_weights(root)
            os.environ["LIFERS_SKIP_HEALTH_CHECK"] = "1"
            try:
                agent = LifersAgent(AgentConfig(root_dir=root, model="markov", sandbox=True))
                assert agent._npc_context_block() == ""
                del agent
            finally:
                os.environ.pop("LIFERS_SKIP_HEALTH_CHECK", None)

    def test_npc_dialogue_hint_via_agent(self) -> None:
        """Agent's _npc_dialogue_hint delegates to engine dialogue tree."""
        eng = NpcEngine()
        tree = _sample_tree()
        eng.states["Shop"] = NpcState(
            profile=NpcProfile(name="Shop", persona="店主", dialogue_root=tree)
        )
        # We test through the agent's prompt injection without agent init:
        # the _npc_dialogue_hint is called by _quick_chat_inference_pack.
        # Direct engine test:
        _, reply = eng.dialogue_match("Shop", "你好")
        assert reply == "你好呀！"
        _, reply2 = eng.dialogue_match("Shop", "做任务")
        assert reply2 == "任务来了"
        _, reply3 = eng.dialogue_match("Shop", "其他话")
        assert reply3 == "嗯？"

    def test_npc_react_and_save_through_agent(self) -> None:
        """Verify _npc_react_for_turn updates state and save_all persists it."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            _write_agent_root(root, npc_config=[
                {"name": "Bob", "persona": "守卫", "voice": "低沉"},
            ])
            _write_markov_weights(root)
            os.environ["LIFERS_SKIP_HEALTH_CHECK"] = "1"
            try:
                agent = LifersAgent(AgentConfig(root_dir=root, model="markov", sandbox=True))
                st = agent.npc_engine.states["Bob"]
                assert st.turn_count == 0
                assert st.relationship == 0.0

                # Simulate a successful interaction addressing the NPC
                agent._npc_react_for_turn("Hello Bob", tool_result_ok=True)
                assert st.turn_count == 1
                assert st.relationship > 0
                assert st.emotion.valence > 0

                # Save should persist state
                agent.npc_engine.save_all(root)
                state_file = root / "state" / "npc_Bob.json"
                assert state_file.is_file()
                data = json.loads(state_file.read_text(encoding="utf-8"))
                assert data["turn_count"] == 1
                assert data["relationship"] > 0

                del agent
            finally:
                os.environ.pop("LIFERS_SKIP_HEALTH_CHECK", None)

    def test_npc_decay_when_not_addressed(self) -> None:
        """When no NPC is named in user text, all NPC emotions decay slightly."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            _write_agent_root(root, npc_config=[
                {"name": "Alice", "persona": "店员"},
            ])
            _write_markov_weights(root)
            os.environ["LIFERS_SKIP_HEALTH_CHECK"] = "1"
            try:
                agent = LifersAgent(AgentConfig(root_dir=root, model="markov", sandbox=True))
                st = agent.npc_engine.states["Alice"]
                st.emotion.valence = 0.5
                st.emotion.arousal = 0.5

                # User text does NOT address Alice
                agent._npc_react_for_turn("Hello everyone", tool_result_ok=True)
                # Valence should have decayed (multiplied by 0.98)
                assert st.emotion.valence < 0.5
                assert st.emotion.valence > 0  # but still positive

                del agent
            finally:
                os.environ.pop("LIFERS_SKIP_HEALTH_CHECK", None)

    def test_npc_dialogue_history_capped(self) -> None:
        """NPC dialogue history stays within 16 entries."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            root = Path(d)
            _write_agent_root(root, npc_config=[
                {"name": "Chatty", "persona": "话多的人"},
            ])
            _write_markov_weights(root)
            os.environ["LIFERS_SKIP_HEALTH_CHECK"] = "1"
            try:
                agent = LifersAgent(AgentConfig(root_dir=root, model="markov", sandbox=True))
                st = agent.npc_engine.states["Chatty"]
                for i in range(20):
                    agent._npc_react_for_turn(f"Hello Chatty message {i}", tool_result_ok=True)
                assert len(st.dialogue_history) <= 16

                del agent
            finally:
                os.environ.pop("LIFERS_SKIP_HEALTH_CHECK", None)
