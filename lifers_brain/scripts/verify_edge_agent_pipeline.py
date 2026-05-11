#!/usr/bin/env python3
"""
一键核对：边缘对话链路（路由 → CHAT_QUICK 装配 → prompt 体量）+ stack 中 embodied 配置 + 训练控制文件。

用法（在 lifers_brain 根）:
  python3 scripts/verify_edge_agent_pipeline.py
  LIFERS_BRAIN_ROOT=/path/to/lifers_brain python3 scripts/verify_edge_agent_pipeline.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if (_ROOT / "lifers_brain").is_dir():
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    root = Path(os.environ.get("LIFERS_BRAIN_ROOT", str(_ROOT))).resolve()
    print("[verify-edge] root", root)

    ctl = root / "weights" / ".train_control"
    if ctl.is_file():
        print("[verify-edge] train_control", ctl.read_text(encoding="utf-8").strip().splitlines()[0])
    else:
        print("[verify-edge] train_control (missing)", ctl)

    stack_path = root / "config" / "stack.json"
    if stack_path.is_file():
        stack = json.loads(stack_path.read_text(encoding="utf-8"))
        emb = stack.get("embodied_world") or {}
        print("[verify-edge] embodied_world.enabled", emb.get("enabled"))
        print("[verify-edge] embodied_world.state_relpath", emb.get("state_relpath"))
        dn = emb.get("dynamic_npc") if isinstance(emb.get("dynamic_npc"), dict) else {}
        print("[verify-edge] embodied_world.dynamic_npc.enabled", dn.get("enabled"))
    else:
        print("[verify-edge] stack.json missing", stack_path)

    from lifers_brain.taskflow.dialogue_router import infer_dialogue_route

    for text, has_ctx in (
        ("收到", False),
        ("能做什么", False),
        ("search python asyncio", False),
        ("做一个飞机大战游戏", False),
        ("make a small game", False),
    ):
        r = infer_dialogue_route(text, has_ctx, emit=False)
        print("[verify-edge] route", repr(text), "->", r.kind.value, r.reason)

    from lifers_brain.agent import AgentConfig, LifersAgent
    from lifers_brain.markov_lm import train_from_text

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        td = Path(d)
        for sub in ("config", "weights", "memory", "logs", "state"):
            (td / sub).mkdir(parents=True, exist_ok=True)
        stack = {
            "version": 1,
            "runtime": {"role": "brain"},
            "brain": {
                "model": "markov",
                "sandbox": True,
                "session_max_turns": 8,
                "memory_db": "memory/t.sqlite3",
                "weights": {"markov": "weights/m.json", "transformer": "weights/t.json"},
                "deep_steward": {"enabled": False},
            },
            "human_sim": {"enabled": False},
            "instincts": {"enabled": False},
            "openclaw": {"enabled": False},
            "organ_system": {"enabled": False},
            "physiology_sim": {"enabled": False},
            "llm_ops": {"enabled": False},
        }
        (td / "config" / "stack.json").write_text(json.dumps(stack, ensure_ascii=False, indent=2), encoding="utf-8")
        train_from_text("测" * 200).save(td / "weights" / "m.json")
        agent = LifersAgent(AgentConfig(root_dir=td, model="markov", sandbox=True))
        agent.brain.model = "transformer"
        agent._stack_context_body = lambda: "STACK" * 3000
        for _ in range(30):
            agent.session.add_turn("user", "U" * 200)
            agent.session.add_turn("assistant", "A" * 200)
        pack = agent._quick_chat_inference_pack("hi", [])
        print("[verify-edge] quick_pack_chars(transformer caps)", len(pack))
        if len(pack) > 30_000:
            print("[verify-edge] FAIL pack too large", len(pack), file=sys.stderr)
            return 1

    print("[verify-edge] CHAT_QUICK: LIFERS_AGENTS_UI_BRIDGE=1 → 默认追加【本轮·生成锚】；LIFERS_QUICK_TIME_FOOTER=0 关闭")
    print("[verify-edge] ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
