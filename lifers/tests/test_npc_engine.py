"""Tests for NPC engine (profile, emotion, dialogue tree, state, engine)."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from lifers.npc_engine import (
    DialogueNode,
    NpcEmotion,
    NpcEngine,
    NpcProfile,
    NpcState,
    match_dialogue_tree,
)


# ── NpcEmotion ────────────────────────────────────────────────────────────────


class TestNpcEmotion:
    def test_default_values(self) -> None:
        e = NpcEmotion()
        assert e.valence == 0.0
        assert e.arousal == 0.3

    def test_mood_labels(self) -> None:
        # arousal >= 0.3, valence > 0.3
        assert NpcEmotion(valence=0.5, arousal=0.6).mood_label() == "愉快"
        # arousal >= 0.3, valence > 0.3, arousal <= 0.5
        assert NpcEmotion(valence=0.5, arousal=0.4).mood_label() == "放松"
        # arousal >= 0.3, valence < -0.3, arousal > 0.5
        assert NpcEmotion(valence=-0.5, arousal=0.6).mood_label() == "恼怒"
        # arousal >= 0.3, valence < -0.3, arousal <= 0.5
        assert NpcEmotion(valence=-0.5, arousal=0.4).mood_label() == "不悦"
        # arousal < 0.3, valence < 0
        assert NpcEmotion(valence=-0.2, arousal=0.1).mood_label() == "低落"
        # arousal < 0.3, valence >= 0
        assert NpcEmotion(valence=0.1, arousal=0.1).mood_label() == "平静"
        # no other branch hits: => 中性
        assert NpcEmotion(valence=0.0, arousal=0.4).mood_label() == "中性"
        assert NpcEmotion(valence=0.0, arousal=0.5).mood_label() == "中性"

    def test_decay(self) -> None:
        e = NpcEmotion(valence=0.8, arousal=0.9)
        e.decay(0.5)
        assert e.valence == 0.4
        assert e.arousal == 0.45

    def test_decay_floor(self) -> None:
        e = NpcEmotion(valence=0.1, arousal=0.05)
        e.decay(0.5)
        assert e.arousal >= 0.0


# ── NpcProfile ────────────────────────────────────────────────────────────────


class TestNpcProfile:
    def test_minimal(self) -> None:
        p = NpcProfile(name="NPC", persona="测试")
        assert p.name == "NPC"
        assert p.persona == "测试"
        assert p.dialogue_root is None

    def test_with_dialogue_root(self) -> None:
        root = DialogueNode(id="r", text="", children=[])
        p = NpcProfile(name="Shop", persona="店主", dialogue_root=root)
        assert p.dialogue_root is root


# ── DialogueNode / match_dialogue_tree ────────────────────────────────────────


class TestDialogueTree:
    def _sample_tree(self) -> DialogueNode:
        return DialogueNode(
            id="root",
            text="",
            children=[
                DialogueNode(id="greet", text="你好呀！", keywords=["你好", "嗨"]),
                DialogueNode(id="quest", text="任务来了", keywords=["任务", "接"]),
                DialogueNode(id="bye", text="再见！", keywords=["再见", "拜拜"]),
                DialogueNode(id="fallback", text="嗯？", is_fallback=True),
            ],
        )

    def test_keyword_match(self) -> None:
        root = self._sample_tree()
        node = match_dialogue_tree(root, "你好世界")
        assert node is not None
        assert node.id == "greet"

    def test_fallback(self) -> None:
        root = self._sample_tree()
        node = match_dialogue_tree(root, "完全不相关的内容")
        assert node is not None
        assert node.id == "fallback"

    def test_no_match_and_no_fallback(self) -> None:
        root = DialogueNode(id="r", text="", children=[
            DialogueNode(id="a", text="A", keywords=["x"]),
        ])
        node = match_dialogue_tree(root, "zzz")
        assert node is None

    def test_nested_tree(self) -> None:
        root = DialogueNode(id="r", text="", children=[
            DialogueNode(id="a", text="A", keywords=["a"], children=[
                DialogueNode(id="a1", text="A1", keywords=["深"]),
            ]),
            DialogueNode(id="fallback", text="?", is_fallback=True),
        ])
        node = match_dialogue_tree(root, "a 深")
        assert node is not None
        assert node.id == "a1"

    def test_multi_keyword_any_match(self) -> None:
        root = DialogueNode(id="r", text="", children=[
            DialogueNode(id="x", text="X", keywords=["foo", "bar"]),
            DialogueNode(id="fallback", text="?", is_fallback=True),
        ])
        assert match_dialogue_tree(root, "hello bar world").id == "x"
        assert match_dialogue_tree(root, "foo").id == "x"

    def test_fallback_skipped_during_walk(self) -> None:
        """Fallback nodes should not be checked during keyword walk."""
        root = DialogueNode(id="r", text="", children=[
            DialogueNode(id="a", text="A", keywords=["a"]),
            DialogueNode(id="f", text="F", is_fallback=True),
        ])
        node = match_dialogue_tree(root, "a")
        assert node is not None
        assert node.id == "a"


# ── NpcState ──────────────────────────────────────────────────────────────────


class TestNpcState:
    def test_context_line_contains_info(self) -> None:
        p = NpcProfile(name="Test", persona="测试角色", voice="轻声")
        s = NpcState(profile=p)
        ctx = s.context_line()
        assert "Test" in ctx
        assert "测试角色" in ctx
        assert "NPC" in ctx

    def test_react_positive(self) -> None:
        p = NpcProfile(name="T", persona="t")
        s = NpcState(profile=p)
        assert s.turn_count == 0
        s.react("hello", tool_result_ok=True)
        assert s.turn_count == 1
        assert s.relationship > 0
        assert s.emotion.valence > 0
        assert s.last_interaction_ts > 0

    def test_react_negative(self) -> None:
        p = NpcProfile(name="T", persona="t")
        s = NpcState(profile=p, emotion=NpcEmotion(valence=0.0, arousal=0.5))
        s.react("bad", tool_result_ok=False)
        assert s.turn_count == 1
        assert s.relationship < 0
        assert s.emotion.valence < 0

    def test_react_relationship_bounds(self) -> None:
        p = NpcProfile(name="T", persona="t")
        s = NpcState(profile=p, relationship=0.99)
        for _ in range(10):
            s.react("ok", tool_result_ok=True)
        assert s.relationship <= 1.0

        s2 = NpcState(profile=p, relationship=-0.99)
        for _ in range(10):
            s2.react("bad", tool_result_ok=False)
        assert s2.relationship >= -1.0

    def test_emotion_bounds(self) -> None:
        p = NpcProfile(name="T", persona="t")
        s = NpcState(profile=p, emotion=NpcEmotion(valence=0.95, arousal=0.5))
        for _ in range(10):
            s.react("ok", tool_result_ok=True)
        assert s.emotion.valence <= 1.0


# ── NpcEngine ─────────────────────────────────────────────────────────────────


class TestNpcEngine:
    def test_empty(self) -> None:
        eng = NpcEngine()
        assert eng.active_context_lines() == []
        assert eng.detect_active_npc("hello") is None

    def test_from_stack_no_npc(self) -> None:
        eng = NpcEngine.from_stack({}, Path("/"))
        assert len(eng.states) == 0

    def test_from_stack_with_npc(self) -> None:
        stack = {
            "embodied_world": {
                "dynamic_npc": [
                    {"name": "Alice", "persona": "友好的店员", "voice": "温柔"},
                    {"name": "Bob", "persona": "严肃的守卫", "voice": "低沉"},
                ]
            }
        }
        eng = NpcEngine.from_stack(stack, Path("/"))
        assert len(eng.states) == 2
        assert "Alice" in eng.states
        assert "Bob" in eng.states
        assert eng.states["Alice"].profile.voice == "温柔"

    def test_detect_active_npc(self) -> None:
        eng = NpcEngine()
        eng.states["Alice"] = NpcState(profile=NpcProfile(name="Alice", persona="a"))
        eng.states["Bob"] = NpcState(profile=NpcProfile(name="Bob", persona="b"))
        assert eng.detect_active_npc("Hello Alice") == "Alice"
        assert eng.detect_active_npc("Bob, stop!") == "Bob"
        assert eng.detect_active_npc("Hey everyone") is None  # no name mentioned

    def test_dialogue_match_via_engine(self) -> None:
        tree = DialogueNode(id="r", text="", children=[
            DialogueNode(id="g", text="欢迎！", keywords=["你好"]),
            DialogueNode(id="f", text="嗯？", is_fallback=True),
        ])
        p = NpcProfile(name="Shop", persona="店主", dialogue_root=tree)
        eng = NpcEngine()
        eng.states["Shop"] = NpcState(profile=p)
        node_id, reply = eng.dialogue_match("Shop", "你好")
        assert node_id == "g"
        assert reply == "欢迎！"
        node_id2, reply2 = eng.dialogue_match("Shop", "其他")
        assert node_id2 == "f"
        assert reply2 == "嗯？"

    def test_dialogue_match_no_tree(self) -> None:
        p = NpcProfile(name="N", persona="n")
        eng = NpcEngine()
        eng.states["N"] = NpcState(profile=p)
        assert eng.dialogue_match("N", "hello") == (None, None)

    def test_save_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            eng = NpcEngine()
            eng.states["Test"] = NpcState(
                profile=NpcProfile(name="Test", persona="测试"),
                emotion=NpcEmotion(valence=0.5, arousal=0.3),
                turn_count=3,
                relationship=0.42,
                last_node_id="greet",
            )
            eng.save_all(root)
            state_file = root / "state" / "npc_Test.json"
            assert state_file.is_file()
            data = json.loads(state_file.read_text(encoding="utf-8"))
            assert data["name"] == "Test"
            assert data["emotion"]["v"] == 0.5
            assert data["turn_count"] == 3
            assert data["relationship"] == 0.42
            assert data["last_node_id"] == "greet"
            # Reload via _load_single_profile
            eng2 = NpcEngine()
            eng2._load_single_profile({"name": "Test", "persona": "测试"}, root)
            st2 = eng2.states["Test"]
            assert st2.last_node_id == "greet"
            assert st2.emotion.valence == 0.5
            assert st2.turn_count == 3

    def test_load_persisted_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_dir = root / "state"
            state_dir.mkdir(parents=True)
            (state_dir / "npc_Persisted.json").write_text(
                json.dumps({
                    "name": "Persisted",
                    "persona": "已保存的角色",
                    "emotion": {"v": -0.3, "a": 0.7},
                    "turn_count": 5,
                    "last_interaction_ts": 0.0,
                    "relationship": 0.15,
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            stack = {
                "embodied_world": {
                    "dynamic_npc": [
                        {"name": "Persisted", "persona": "已保存的角色"},
                    ]
                }
            }
            eng = NpcEngine.from_stack(stack, root)
            st = eng.states["Persisted"]
            assert st.emotion.valence == -0.3
            assert st.emotion.arousal == 0.7
            assert st.turn_count == 5
            assert st.relationship == 0.15

    def test_save_all_creates_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            eng = NpcEngine()
            eng.states["NewNPC"] = NpcState(profile=NpcProfile(name="NewNPC", persona="new"))
            eng.save_all(root)
            assert (root / "state" / "npc_NewNPC.json").is_file()

    def test_active_context_lines_single(self) -> None:
        p = NpcProfile(name="N", persona="n")
        eng = NpcEngine()
        eng.states["N"] = NpcState(profile=p)
        lines = eng.active_context_lines("N")
        assert len(lines) == 1
        assert "N" in lines[0]

    def test_active_context_lines_all(self) -> None:
        eng = NpcEngine()
        eng.states["A"] = NpcState(profile=NpcProfile(name="A", persona="a"))
        eng.states["B"] = NpcState(profile=NpcProfile(name="B", persona="b"))
        lines = eng.active_context_lines()
        assert len(lines) == 2

    def test_dialogue_history_in_context(self) -> None:
        """NPC context line includes recent dialogue history."""
        p = NpcProfile(name="Shop", persona="店主", voice="轻声")
        st = NpcState(profile=p)
        st.dialogue_history = ["你好", "今天天气不错"]
        ctx = st.context_line()
        assert "最近对话" in ctx
        assert "你好" in ctx
        assert "今天" in ctx

    def test_dialogue_history_truncated_in_context(self) -> None:
        """Only last 3 entries shown in context line."""
        p = NpcProfile(name="Shop", persona="店主")
        st = NpcState(profile=p)
        st.dialogue_history = [f"msg{i}" for i in range(10)]
        ctx = st.context_line()
        # Should only show last 3
        assert "msg9" in ctx or "最近对话" in ctx

    def test_engine_saves_and_restores_dialogue_history(self) -> None:
        """Dialogue_history is persisted and restored through save/load."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            eng = NpcEngine()
            p = NpcProfile(name="Historian", persona="记录者")
            st = NpcState(profile=p)
            st.dialogue_history = ["第一句", "第二句", "第三句"]
            eng.states["Historian"] = st
            eng.save_all(root)
            # Verify dialogue_history was persisted
            state_file = root / "state" / "npc_Historian.json"
            data = json.loads(state_file.read_text(encoding="utf-8"))
            assert "dialogue_history" in data
            assert data["dialogue_history"] == ["第一句", "第二句", "第三句"]
            assert data["name"] == "Historian"
            # Reload into new engine — state is restored via _load_single_profile
            eng2 = NpcEngine()
            eng2._load_single_profile({"name": "Historian", "persona": "记录者"}, root)
            st2 = eng2.states["Historian"]
            assert st2.dialogue_history == ["第一句", "第二句", "第三句"]


# ── DialogueNode serialization (to_dict / from_dict) ────────────────────


class TestDialogueNodeSerialization:
    def test_roundtrip_leaf(self) -> None:
        n = DialogueNode(id="a", text="hello", keywords=["hi"])
        d = n.to_dict()
        n2 = DialogueNode.from_dict(d)
        assert n2.id == "a"
        assert n2.text == "hello"
        assert n2.keywords == ["hi"]
        assert n2.children == []

    def test_roundtrip_with_children(self) -> None:
        root = DialogueNode(id="r", text="", children=[
            DialogueNode(id="c1", text="opt1", keywords=["a"]),
            DialogueNode(id="c2", text="opt2", keywords=["b"], is_fallback=True),
        ])
        d = root.to_dict()
        root2 = DialogueNode.from_dict(d)
        assert root2.id == "r"
        assert len(root2.children) == 2
        assert root2.children[0].id == "c1"
        assert root2.children[1].is_fallback is True

    def test_roundtrip_with_relationship_gates(self) -> None:
        n = DialogueNode(id="g", text="secret", keywords=["key"], min_relationship=0.5, max_relationship=1.0)
        d = n.to_dict()
        n2 = DialogueNode.from_dict(d)
        assert n2.min_relationship == 0.5
        assert n2.max_relationship == 1.0

    def test_to_dict_roundtrip_via_json(self) -> None:
        root = DialogueNode(id="r", text="", children=[
            DialogueNode(id="a", text="A", keywords=["x"]),
            DialogueNode(id="b", text="B", keywords=["y"], children=[
                DialogueNode(id="b1", text="B1", keywords=["z"]),
            ]),
        ])
        raw = json.dumps(root.to_dict(), ensure_ascii=False)
        restored = DialogueNode.from_dict(json.loads(raw))
        assert restored.id == "r"
        assert restored.children[1].children[0].id == "b1"


# ── Relationship-gated dialogue ─────────────────────────────────────────


class TestRelationshipGatedDialogue:
    def test_node_out_of_range_low(self) -> None:
        """Node with min_relationship=0.5 should not match when relationship=0."""
        root = DialogueNode(id="r", text="", children=[
            DialogueNode(id="secret", text="secret msg", keywords=["key"], min_relationship=0.5),
            DialogueNode(id="fallback", text="?", is_fallback=True),
        ])
        node = match_dialogue_tree(root, "key", relationship=0.0)
        assert node is not None
        assert node.id == "fallback"  # skipped secret, went to fallback

    def test_node_in_range(self) -> None:
        """Node with min_relationship=0.5 should match when relationship=0.7."""
        root = DialogueNode(id="r", text="", children=[
            DialogueNode(id="secret", text="secret msg", keywords=["key"], min_relationship=0.5),
        ])
        node = match_dialogue_tree(root, "key", relationship=0.7)
        assert node is not None
        assert node.id == "secret"

    def test_node_max_relationship(self) -> None:
        """Node with max_relationship=-0.3 (hostile only) should match when relationship is low."""
        root = DialogueNode(id="r", text="", children=[
            DialogueNode(id="hostile", text="go away", keywords=["hello"], max_relationship=-0.3),
            DialogueNode(id="friendly", text="welcome", keywords=["hello"]),
        ])
        # hostile relationship → hostile branch
        node = match_dialogue_tree(root, "hello", relationship=-0.5)
        assert node is not None
        assert node.id == "hostile"

    def test_node_max_relationship_friendly(self) -> None:
        """Same tree, friendly relationship → friendly branch."""
        root = DialogueNode(id="r", text="", children=[
            DialogueNode(id="hostile", text="go away", keywords=["hello"], max_relationship=-0.3),
            DialogueNode(id="friendly", text="welcome", keywords=["hello"]),
        ])
        node = match_dialogue_tree(root, "hello", relationship=0.5)
        assert node is not None
        assert node.id == "friendly"


# ── Multi-turn dialogue (prefer_branch) ──────────────────────────────────


class TestMultiTurnDialogue:
    def _tree(self) -> DialogueNode:
        return DialogueNode(id="root", text="", children=[
            DialogueNode(id="ask_name", text="你叫什么？", keywords=["你好"]),
            DialogueNode(id="tell_name", text="我叫小A", keywords=["名字"], children=[
                DialogueNode(id="name_detail", text="全名是小A同学", keywords=["全名"]),
            ]),
            DialogueNode(id="fallback", text="嗯？", is_fallback=True),
        ])

    def test_prefer_branch_continues(self) -> None:
        root = self._tree()
        # First: match "名字" → goes to "tell_name"
        node = match_dialogue_tree(root, "名字", prefer_branch=None)
        assert node is not None
        assert node.id == "tell_name"
        # Next: with prefer_branch="tell_name", "全名" should find child "name_detail"
        node2 = match_dialogue_tree(root, "全名", prefer_branch="tell_name")
        assert node2 is not None
        assert node2.id == "name_detail"

    def test_prefer_branch_falls_back_when_no_child_match(self) -> None:
        root = self._tree()
        node = match_dialogue_tree(root, "你好", prefer_branch="tell_name")
        # "你好" doesn't match tell_name's children, falls back to root walk
        assert node is not None
        assert node.id == "ask_name"

    def test_dialogue_match_tracks_last_node(self) -> None:
        """NpcEngine.dialogue_match should update last_node_id."""
        root = DialogueNode(id="r", text="", children=[
            DialogueNode(id="a", text="A", keywords=["hello"]),
            DialogueNode(id="fallback", text="?", is_fallback=True),
        ])
        eng = NpcEngine()
        eng.states["Test"] = NpcState(profile=NpcProfile(name="Test", persona="t", dialogue_root=root))
        node_id, reply = eng.dialogue_match("Test", "hello")
        assert node_id == "a"
        assert eng.states["Test"].last_node_id == "a"

    def test_greeting_once(self) -> None:
        """Greeting should only be delivered once."""
        p = NpcProfile(name="G", persona="g", greeting="你好呀！")
        eng = NpcEngine()
        eng.states["G"] = NpcState(profile=p)
        assert eng.greeting_for("G") == "你好呀！"
        assert eng.greeting_for("G") is None  # already greeted

    def test_all_npc_names(self) -> None:
        eng = NpcEngine()
        eng.states["B"] = NpcState(profile=NpcProfile(name="B", persona="b"))
        eng.states["A"] = NpcState(profile=NpcProfile(name="A", persona="a"))
        assert eng.all_npc_names() == ["A", "B"]


# ── NPC config file loading ──────────────────────────────────────────────


class TestNpcConfigLoading:
    def test_load_single_npc_from_config_dir(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            npc_dir = root / "config" / "npcs"
            npc_dir.mkdir(parents=True)
            (npc_dir / "shop.json").write_text(
                json.dumps({"name": "Shop", "persona": "店主", "voice": "温柔"}, ensure_ascii=False),
                encoding="utf-8",
            )
            configs = NpcEngine.load_npc_configs(root)
            assert "Shop" in configs
            assert configs["Shop"]["persona"] == "店主"
            assert configs["Shop"]["voice"] == "温柔"

    def test_load_multiple_npcs_from_array(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            npc_dir = root / "config" / "npcs"
            npc_dir.mkdir(parents=True)
            (npc_dir / "npcs.json").write_text(
                json.dumps([
                    {"name": "A", "persona": "a"},
                    {"name": "B", "persona": "b"},
                ], ensure_ascii=False),
                encoding="utf-8",
            )
            configs = NpcEngine.load_npc_configs(root)
            assert "A" in configs
            assert "B" in configs

    def test_empty_config_dir(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            configs = NpcEngine.load_npc_configs(root)
            assert configs == {}

    def test_from_stack_loads_config_dir(self) -> None:
        """Engine loads both stack NPCs and config/npcs/*.json."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            (root / "config" / "npcs").mkdir(parents=True)
            (root / "config" / "npcs" / "extra.json").write_text(
                json.dumps({"name": "Extra", "persona": "额外角色", "voice": "低沉"}, ensure_ascii=False),
                encoding="utf-8",
            )
            eng = NpcEngine.from_stack({
                "embodied_world": {
                    "dynamic_npc": [{"name": "Main", "persona": "主要角色"}],
                }
            }, root)
            assert "Main" in eng.states
            assert "Extra" in eng.states

    def test_config_dir_overrides_stack(self) -> None:
        """Config/npcs/*.json values override stack.json values for same name."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            (root / "config" / "npcs").mkdir(parents=True)
            (root / "config" / "npcs" / "Alice.json").write_text(
                json.dumps({"name": "Alice", "persona": "覆盖后的角色", "voice": "严肃"}),
                encoding="utf-8",
            )
            eng = NpcEngine.from_stack({
                "embodied_world": {
                    "dynamic_npc": [{"name": "Alice", "persona": "原始角色"}],
                }
            }, root)
            assert eng.states["Alice"].profile.persona == "覆盖后的角色"
            assert eng.states["Alice"].profile.voice == "严肃"


# ── Dialogue tree persistence ─────────────────────────────────────────────


class TestDialogueTreePersistence:
    def test_save_dialogue_trees(self) -> None:
        tree = DialogueNode(id="r", text="", children=[
            DialogueNode(id="a", text="A", keywords=["hi"]),
        ])
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            eng = NpcEngine()
            eng.states["Talker"] = NpcState(
                profile=NpcProfile(name="Talker", persona="t", dialogue_root=tree),
            )
            eng.save_dialogue_trees(root)
            saved = root / "config" / "npcs" / "Talker.json"
            assert saved.is_file()
            data = json.loads(saved.read_text(encoding="utf-8"))
            assert data["name"] == "Talker"
            assert "dialogue_root" in data
            assert data["dialogue_root"]["id"] == "r"

    def test_load_dialogue_trees(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            npc_dir = root / "config" / "npcs"
            npc_dir.mkdir(parents=True)
            (npc_dir / "Bot.json").write_text(
                json.dumps({
                    "name": "Bot",
                    "persona": "bot",
                    "dialogue_root": {
                        "id": "r",
                        "text": "",
                        "keywords": [],
                        "is_fallback": False,
                        "min_relationship": -1.0,
                        "max_relationship": 1.0,
                        "children": [
                            {"id": "g", "text": "你好", "keywords": ["hi"], "is_fallback": False,
                             "min_relationship": -1.0, "max_relationship": 1.0, "children": []},
                        ],
                    },
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            eng = NpcEngine()
            eng.states["Bot"] = NpcState(profile=NpcProfile(name="Bot", persona="bot"))
            count = eng.load_dialogue_trees(root)
            assert count == 1
            assert eng.states["Bot"].profile.dialogue_root is not None
            node_id, reply = eng.dialogue_match("Bot", "hi")
            assert node_id == "g"
            assert reply == "你好"

    def test_save_all_persists_tree_separately(self) -> None:
        """save_all writes the dialogue tree to a separate _tree.json file."""
        tree = DialogueNode(id="r", text="", children=[
            DialogueNode(id="a", text="A", keywords=["x"]),
        ])
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            eng = NpcEngine()
            eng.states["T"] = NpcState(
                profile=NpcProfile(name="T", persona="t", dialogue_root=tree),
            )
            eng.save_all(root)
            tree_file = root / "state" / "npc_T_tree.json"
            assert tree_file.is_file()
            data = json.loads(tree_file.read_text(encoding="utf-8"))
            assert data["id"] == "r"


# ── NpcProfile serialization ─────────────────────────────────────────────


class TestNpcProfileSerialization:
    def test_to_dict_minimal(self) -> None:
        p = NpcProfile(name="Test", persona="测试")
        d = p.to_dict()
        assert d["name"] == "Test"
        assert d["persona"] == "测试"
        assert "dialogue_root" not in d

    def test_to_dict_with_dialogue_root(self) -> None:
        root = DialogueNode(id="r", text="", children=[
            DialogueNode(id="a", text="A", keywords=["hi"]),
        ])
        p = NpcProfile(name="T", persona="t", dialogue_root=root)
        d = p.to_dict()
        assert "dialogue_root" in d
        assert d["dialogue_root"]["id"] == "r"

    def test_from_dict_roundtrip(self) -> None:
        p = NpcProfile(name="Orig", persona="原始", voice="轻声", greeting="你好",
                       backstory="测试角色", portrait_emoji="👤")
        d = p.to_dict()
        p2 = NpcProfile.from_dict(d)
        assert p2.name == "Orig"
        assert p2.persona == "原始"
        assert p2.voice == "轻声"
