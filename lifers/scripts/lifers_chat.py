"""
Lifers Chat CLI — 本地 AI 终端助手，完整接入 Agent / Memory / 推理管线。

用法:
  lifers           交互对话（默认）
  lifers chat      交互对话
  lifers ask ...   单次问答
  lifers code ...  生成代码
  lifers think ... 深度推理
  lifers stats     查看模型状态
  lifers test      运行基础能力测试

管线: 用户输入 → 意图分类 → 知识检索 → 推理生成 → 输出后处理 → 用户
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 输入验证与清洗
# ---------------------------------------------------------------------------

def validate_input(text: str) -> Tuple[bool, str]:
    """验证并清洗用户输入。返回 (ok, cleaned_or_error)。"""
    if not text or not text.strip():
        return False, "输入为空，请输入内容。"
    t = text.strip()
    if len(t) > 8000:
        t = t[:8000] + "...(truncated)"
    # 过滤控制字符（保留换行和制表）
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", t)
    return True, t


# ---------------------------------------------------------------------------
# 输出后处理与优化
# ---------------------------------------------------------------------------

_REP_PAT = re.compile(r"(.{8,}?)\1{3,}")

# Cross-turn loop detection
_LOOP_HISTORY: List[str] = []
_MAX_LOOP_HISTORY = 8
_LOOP_SIMILARITY_THRESHOLD = 0.85


def _detect_repetition(text: str) -> bool:
    """检测重复生成（单段重复 3 次以上）。"""
    return bool(_REP_PAT.search(text))


def _detect_cross_turn_loop(text: str) -> bool:
    """检测跨轮次循环：模型对不同的输入始终给出相同回复。"""
    if len(text) < 20:
        return False
    _LOOP_HISTORY.append(text)
    if len(_LOOP_HISTORY) > _MAX_LOOP_HISTORY:
        _LOOP_HISTORY.pop(0)
    if len(_LOOP_HISTORY) < 4:
        return False
    # Check if last 4 responses are near-identical (simple prefix overlap)
    recent = _LOOP_HISTORY[-4:]
    for i in range(len(recent) - 1):
        shorter = min(len(recent[i]), len(recent[i+1]))
        if shorter < 20:
            continue
        overlap = sum(1 for a, b in zip(recent[i][:shorter], recent[i+1][:shorter]) if a == b)
        if overlap / shorter < _LOOP_SIMILARITY_THRESHOLD:
            return False
    return True


def _reset_loop_detection():
    """重置跨轮次循环检测器。"""
    _LOOP_HISTORY.clear()


def _clean_output(raw: str, user_input: str = "") -> str:
    """后处理：截断用户标记、去重、裁剪。"""
    for marker in ["\n用户:", "\nUSER:", "\n助手:", "\nASSISTANT:", "\n---", "\n# "]:
        idx = raw.find(marker)
        if idx > 0:
            raw = raw[:idx]
    raw = raw.strip()
    if _detect_repetition(raw):
        raw = raw[:200] + "...(重复内容已截断)"
        return raw
    if len(raw) > 2000:
        # 尝试在句子边界截断
        cut = raw.rfind("。", 1800, 2000)
        if cut > 0:
            raw = raw[: cut + 1] + "\n(输出较长，已截断)"
        else:
            raw = raw[:2000] + "..."
    return raw


# ---------------------------------------------------------------------------
# 权重自动发现
# ---------------------------------------------------------------------------

def _find_weights() -> Tuple[str, Path]:
    """自动发现最佳可用权重。"""
    weights_dir = ROOT / "weights"
    deep = weights_dir / "lifers_deep_transformer.json"
    if deep.is_file() and deep.stat().st_size > 1024:
        return "lifers", deep
    tiny = weights_dir / "lifers_transformer.json"
    if tiny.is_file() and tiny.stat().st_size > 1024:
        return "transformer", tiny
    markov = weights_dir / "lifers_markov.json"
    if markov.is_file():
        return "markov", markov
    return "none", Path()


def _get_model_info(kind: str, w) -> str:
    if kind == "lifers":
        import numpy as _np
        try:
            json_path = ROOT / "weights" / "lifers_deep_transformer.json"
            if json_path.is_file():
                import json
                meta = json.loads(json_path.read_text(encoding="utf-8"))
                npz_rel = meta.get("_npz", "lifers_deep_transformer.npz")
                npz = ROOT / "weights" / npz_rel
            else:
                npz = ROOT / "weights" / "lifers_deep_transformer.npz"
            mb = npz.stat().st_size / 1e6 if npz.is_file() else 0
        except Exception:
            mb = 0
        params = w.d_model * w.d_model * 4 + w.d_model * w.d_ff * 2
        return (
            f"D={w.d_model} L={w.n_layers} H={w.n_heads} "
            f"V={len(w.vocab)} seq={w.max_seq} ~{params/1e6:.1f}M params {mb:.0f}MB"
        )
    elif kind == "transformer":
        return f"D={w.d_model} V={len(w.vocab)}"
    return f"V={len(w.vocab)} chains"


# ---------------------------------------------------------------------------
# 推理核心 —— 完整 Agent 管线（Taskflow 编排 + 工具 + 记忆），回退到直接 LocalBrain
# ---------------------------------------------------------------------------

def _create_agent(kind: str, weights_path: Path):
    """初始化 LifersAgent（完整管线），失败时回退到直接 LocalBrain。

    Returns: (agent_or_brain, weights, kind, use_full_pipeline)
    """
    from lifers.local_brain import AgentConfig, LocalBrain
    from lifers.model_names import canonical_brain_model

    model = canonical_brain_model(kind)
    cfg = AgentConfig(
        root_dir=ROOT,
        model=model,
        sandbox=True,
        local_lm_max_chars=400,
    )

    # Always create LocalBrain first (needed for stats + fallback)
    brain = LocalBrain(cfg)
    brain.model = model
    wp = brain._weights_path()
    if not wp.exists():
        wp = weights_path
    if model == "lifers":
        w = brain._deep_weights(wp)
    elif model == "transformer":
        w = brain._transformer_weights(wp)
    else:
        w = brain._markov_weights(wp)

    # Try full LifersAgent pipeline
    try:
        os.environ.setdefault("LIFERS_SKIP_HEALTH_CHECK", "1")
        os.environ.setdefault("LIFERS_QUICK_WEB", "0")
        os.environ.setdefault("LIFERS_TASKFLOW", "1")
        from lifers.agent import LifersAgent
        agent = LifersAgent(cfg)
        print(f"[agent] Full pipeline ready (taskflow + {len(agent.tools._tools)} tools + longterm memory)", file=sys.stderr)
        return agent, w, model, True
    except Exception as e:
        print(f"[agent] Full pipeline unavailable ({e}), falling back to direct LocalBrain", file=sys.stderr)
        session = SessionMemory()
        session.max_turns = 12
        return (brain, session), w, model, False


# ---------------------------------------------------------------------------
# 提示词构建（完整管线版）
# ---------------------------------------------------------------------------

SYSTEM_LINES = [
    "你是 Lifers，一个运行在用户本地机器上的智能 AI 助手，由深度 Transformer 模型驱动。",
    "你能够：回答问题、编写代码、逻辑推理、知识检索、深入分析讨论。",
    "回答要求：简洁准确、中文优先、基于上下文推理、避免编造不确定的内容。",
    "对于不清楚的内容，坦诚说明而非猜测。",
]


def _build_prompt(
    session, user_msg: str, kind: str, persona: str = "", mode: str = "chat"
) -> str:
    """构建完整推理提示词（含 system / 历史 / 上下文）。"""
    parts = []

    # System
    parts.append("\n".join(SYSTEM_LINES))
    if persona:
        parts.append(f"\n角色设定: {persona}")

    # Session history (最近 8 轮)
    if session is not None:
        hist = session.context_text()
        if hist:
            parts.append(f"\n--- 对话历史 ---\n{hist}")

    # Mode-specific prefix
    if mode == "code":
        parts.append(f"\n请根据以下需求编写代码，包含必要的注释和错误处理：\n{user_msg}")
    elif mode == "think":
        parts.append(f"\n请对以下话题进行深入分析，从多个角度展开推理：\n{user_msg}")
    else:
        parts.append(f"\n用户: {user_msg}")

    parts.append("\n助手: ")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 单次生成
# ---------------------------------------------------------------------------

def _generate_once(
    brain, w, kind: str, prompt: str, max_chars: int = 400, temperature: float = 0.8
) -> Tuple[str, float]:
    """单次生成，返回 (文本, 耗时秒)。"""
    t0 = time.perf_counter()
    if kind == "lifers":
        from lifers.deep_transformer import generate_text
        d_temp = float(os.environ.get("LIFERS_TEMP", str(temperature)))
        text = generate_text(
            w, prompt,
            max_new_tokens=max(32, min(max_chars, 800)),
            temperature=d_temp,
            seed=random.randint(0, 2**31 - 1),
        )
    elif kind == "transformer":
        from lifers.transformer_lm import generate_text
        text = generate_text(
            w, prompt,
            max_chars=max_chars,
            temperature=temperature,
            seed=random.randint(0, 2**31 - 1),
        )
    else:
        from lifers.markov_lm import generate
        text = generate(w, prompt, max_chars=max_chars)
    elapsed = time.perf_counter() - t0
    return text, elapsed


# ---------------------------------------------------------------------------
# 交互对话
# ---------------------------------------------------------------------------

def interactive_chat(
    agent_data, w, kind: str = "lifers",
    use_full_pipeline: bool = False,
    max_chars: int = 400, temperature: float = 0.8,
) -> None:
    """交互对话主循环。"""
    info = _get_model_info(kind, w)
    label = {"lifers": "Lifers (Deep Transformer)", "transformer": "Transformer", "markov": "Markov"}.get(kind, kind)
    pipeline_label = "Full Agent Pipeline" if use_full_pipeline else "Direct LocalBrain"

    print(f"\n  {'='*50}")
    print(f"  Lifers AI -- Local Assistant")
    print(f"  Model: {label}")
    print(f"  {info}")
    print(f"  Mode: {pipeline_label}")
    print(f"  {'='*50}")
    if use_full_pipeline:
        print(f"  Commands: /clear | /info | /stats | /tools | exit")
    else:
        print(f"  Commands: /clear | /info | /stats | exit")
    print()

    turn = 0
    while True:
        try:
            user_input = input(f"  [{turn + 1}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("  Goodbye!\n")
            break
        if user_input == "/clear":
            if use_full_pipeline:
                agent_data.session.clear()
            else:
                agent_data[1].clear()
            _reset_loop_detection()
            turn = 0
            print("  [Conversation cleared]")
            continue
        if user_input == "/info":
            print(f"  Model: {kind} | {_get_model_info(kind, w)}")
            print(f"  Pipeline: {pipeline_label}")
            continue
        if user_input == "/stats":
            _print_stats(kind, w)
            continue
        if user_input == "/tools" and use_full_pipeline:
            for name in sorted(agent_data.tools._tools.keys()):
                print(f"  - {name}")
            continue

        # Input validation
        ok, cleaned = validate_input(user_input)
        if not ok:
            print(f"  ! {cleaned}")
            continue

        print(f"  [{turn + 1}] Lifers: ", end="", flush=True)
        t0 = time.perf_counter()

        if use_full_pipeline:
            # Full agent pipeline: intent classification -> dispatch -> tools -> memory -> generation
            try:
                from lifers.taskflow.orchestrator import run_lifers_turn
                response = run_lifers_turn(agent_data, cleaned)
            except Exception as e:
                print(f"\n  [Agent error: {e}]")
                continue
        else:
            # Fallback: direct LocalBrain generation
            brain, session = agent_data
            prompt = _build_prompt(session, cleaned, kind)
            try:
                raw, _ = _generate_once(brain, w, kind, prompt, max_chars, temperature)
            except Exception as e:
                print(f"\n  [Generation error: {e}]")
                continue
            response = _clean_output(raw, cleaned)
            if not response:
                response = "(Empty reply, try a more complete sentence)"

            # Cross-turn loop detection
            if _detect_cross_turn_loop(response):
                print(f"  [Loop detected — auto-resetting session]")
                _reset_loop_detection()
                if use_full_pipeline:
                    agent_data.session.clear()
                else:
                    agent_data[1].clear()
                turn = 0

        elapsed = time.perf_counter() - t0
        print(f"{response}")
        print(f"  [{elapsed:.1f}s, {len(response)} chars]\n")

        if not use_full_pipeline:
            brain, session = agent_data
            session.add_turn("user", cleaned)
            session.add_turn("assistant", response)
        turn += 1


# ---------------------------------------------------------------------------
# 单次模式
# ---------------------------------------------------------------------------

def single_ask(agent_data, w, kind: str, query: str,
              use_full_pipeline: bool = False,
              max_chars: int = 400, temperature: float = 0.8) -> int:
    ok, cleaned = validate_input(query)
    if not ok:
        print(f"Error: {cleaned}", file=sys.stderr)
        return 1
    if use_full_pipeline:
        from lifers.taskflow.orchestrator import run_lifers_turn
        try:
            reply = run_lifers_turn(agent_data, cleaned)
        except Exception as e:
            print(f"Agent error: {e}", file=sys.stderr)
            return 1
        print(reply)
    else:
        brain, session = agent_data
        prompt = _build_prompt(None, cleaned, kind, "chat")
        try:
            raw, elapsed = _generate_once(brain, w, kind, prompt, max_chars, temperature)
        except Exception as e:
            print(f"Generation error: {e}", file=sys.stderr)
            return 1
        print(_clean_output(raw, cleaned))
        if os.environ.get("LIFERS_CHAT_VERBOSE"):
            print(f"\n[{elapsed:.1f}s, {len(raw)} chars]", file=sys.stderr)
    return 0


def single_code(agent_data, w, kind: str, task: str,
                use_full_pipeline: bool = False,
                max_chars: int = 500, temperature: float = 0.6) -> int:
    ok, cleaned = validate_input(task)
    if not ok:
        print(f"Error: {cleaned}", file=sys.stderr)
        return 1
    if use_full_pipeline:
        from lifers.taskflow.orchestrator import run_lifers_turn
        try:
            reply = run_lifers_turn(agent_data, f"请根据以下需求编写代码：\n{cleaned}")
        except Exception as e:
            print(f"Agent error: {e}", file=sys.stderr)
            return 1
        print(reply)
    else:
        brain, _ = agent_data
        prompt = _build_prompt(None, cleaned, kind, "code")
        try:
            raw, _ = _generate_once(brain, w, kind, prompt, max_chars, temperature)
        except Exception as e:
            print(f"Generation error: {e}", file=sys.stderr)
            return 1
        text = _clean_output(raw, cleaned)
        if "```" in text:
            text = text.split("```", 2)[0] if text.startswith("```") else text
        print(text.strip())
    return 0


def single_think(agent_data, w, kind: str, topic: str,
                 use_full_pipeline: bool = False,
                 max_chars: int = 600, temperature: float = 0.7) -> int:
    ok, cleaned = validate_input(topic)
    if not ok:
        print(f"Error: {cleaned}", file=sys.stderr)
        return 1
    if use_full_pipeline:
        from lifers.taskflow.orchestrator import run_lifers_turn
        try:
            reply = run_lifers_turn(agent_data, f"请对以下话题进行深入分析：\n{cleaned}")
        except Exception as e:
            print(f"Agent error: {e}", file=sys.stderr)
            return 1
        print(reply)
    else:
        brain, _ = agent_data
        prompt = _build_prompt(None, cleaned, kind, "think")
        try:
            raw, _ = _generate_once(brain, w, kind, prompt, max_chars, temperature)
        except Exception as e:
            print(f"Generation error: {e}", file=sys.stderr)
            return 1
        print(_clean_output(raw, cleaned))
    return 0


# ---------------------------------------------------------------------------
# 状态 / 测试
# ---------------------------------------------------------------------------

def _print_stats(kind: str, w) -> None:
    """打印模型详细状态。"""
    print(f"\n  -- 模型状态 --")
    print(f"  类型: {kind}")
    if kind == "lifers":
        print(f"  d_model: {w.d_model}  d_ff: {w.d_ff}")
        print(f"  layers: {w.n_layers}  heads: {w.n_heads}")
        print(f"  vocab: {len(w.vocab)}  max_seq: {w.max_seq}")
        params = w.d_model * w.d_model * 4 + w.d_model * w.d_ff * 2
        print(f"  参数: ~{params/1e6:.1f}M")
    json_path = ROOT / "weights" / "lifers_deep_transformer.json"
    if json_path.is_file():
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            npz_rel = meta.get("_npz", "lifers_deep_transformer.npz")
            npz = ROOT / "weights" / npz_rel
        except Exception:
            npz = ROOT / "weights" / "lifers_deep_transformer.npz"
    else:
        npz = ROOT / "weights" / "lifers_deep_transformer.npz"
    if npz.is_file():
        print(f"  权重文件: {npz.stat().st_size/1e6:.1f} MB")
    corpus = ROOT / "weights" / "training_corpus.txt"
    if corpus.is_file():
        kb = corpus.stat().st_size / 1024
        chars = len(corpus.read_text(encoding="utf-8"))
        print(f"  语料: {kb:.0f} KB ({chars} chars)")
    hb = ROOT / "weights" / ".train_heartbeat.json"
    if hb.is_file():
        import json
        try:
            d = json.loads(hb.read_text(encoding="utf-8"))
            print(f"  训练: tier {d.get('ramp_iter','?')}/{d.get('ramp_max','?')} step {d.get('sgd_step','?')}/{d.get('sgd_total','?')} loss={d.get('loss','?'):.4f}")
        except Exception:
            pass
    print()


def cmd_test(agent_data, w, kind: str, use_full_pipeline: bool = False) -> int:
    """基础能力测试：验证输入→推理→输出链路完整。"""
    print("\n  -- Lifers Basic Capability Test --\n")
    tests = [
        ("Short QA", "1+1=?"),
        ("Chinese", "Introduce Beijing in one sentence."),
        ("Code", "Write a Python fibonacci function."),
        ("Reasoning", "If all cats fear water, and Tom is a cat, does Tom fear water?"),
    ]
    passed = 0
    for name, query in tests:
        try:
            if use_full_pipeline:
                from lifers.taskflow.orchestrator import run_lifers_turn
                response = run_lifers_turn(agent_data, query)
            else:
                brain, _ = agent_data
                prompt = _build_prompt(None, query, kind, "chat")
                raw, elapsed = _generate_once(brain, w, kind, prompt, max_chars=200)
                response = _clean_output(raw, query)
            ok = len(response) >= 2 and not response.startswith("(Empty")
            status = "OK" if ok else "FAIL"
            if ok:
                passed += 1
            print(f"  [{status}] {name}: {query}")
            print(f"       -> {response[:120]}{'...' if len(response)>120 else ''}")
            if not use_full_pipeline:
                print(f"       {elapsed:.1f}s")
            print()
        except Exception as e:
            print(f"  [FAIL] {name}: {e}\n")
    print(f"  Passed: {passed}/{len(tests)}\n")
    return 0 if passed >= len(tests) * 0.5 else 1


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lifers Chat — 本地 AI 终端助手",
        epilog="直接运行 lifers 进入交互对话。",
    )
    sub = parser.add_subparsers(dest="cmd", help="命令")

    chat_p = sub.add_parser("chat", help="交互对话")
    chat_p.add_argument("--max-chars", type=int, default=400, help="最大输出字符数")
    chat_p.add_argument("--temperature", type=float, default=0.8, help="采样温度")
    chat_p.add_argument("--weights", type=str, help="权重文件路径")

    ask_p = sub.add_parser("ask", help="单次问答")
    ask_p.add_argument("query", type=str, nargs="+", help="问题")
    ask_p.add_argument("--max-chars", type=int, default=400)
    ask_p.add_argument("--temperature", type=float, default=0.8)

    code_p = sub.add_parser("code", help="代码生成")
    code_p.add_argument("task", type=str, nargs="+", help="需求描述")
    code_p.add_argument("--max-chars", type=int, default=500)
    code_p.add_argument("--temperature", type=float, default=0.6)

    think_p = sub.add_parser("think", help="深度推理")
    think_p.add_argument("topic", type=str, nargs="+", help="分析话题")
    think_p.add_argument("--max-chars", type=int, default=600)
    think_p.add_argument("--temperature", type=float, default=0.7)

    sub.add_parser("stats", help="查看模型状态")
    sub.add_parser("test", help="运行基础能力测试")

    args = parser.parse_args()

    # 解析权重
    if hasattr(args, "weights") and args.weights:
        weights_path = Path(args.weights)
        if not weights_path.is_file():
            print(f"错误: 权重文件不存在: {weights_path}", file=sys.stderr)
            return 1
        if "deep" in weights_path.name:
            kind = "lifers"
        elif "markov" in weights_path.name:
            kind = "markov"
        else:
            kind = "transformer"
    else:
        kind, weights_path = _find_weights()

    if kind == "none":
        print("Error: No usable weights found.", file=sys.stderr)
        print(f"Expected one of the following in {ROOT / 'weights'}/", file=sys.stderr)
        print("  lifers_deep_transformer.json  (recommended)", file=sys.stderr)
        print("  lifers_transformer.json", file=sys.stderr)
        print("  lifers_markov.json", file=sys.stderr)
        print("\nRun: python scripts/train_deep_escalate.py", file=sys.stderr)
        return 1

    # Initialize agent/brain
    try:
        agent_data, w, kind, use_full_pipeline = _create_agent(kind, weights_path)
    except Exception as e:
        print(f"Error: Failed to load weights: {e}", file=sys.stderr)
        return 1

    info = _get_model_info(kind, w)
    print(f"[{kind}] {info}", file=sys.stderr)

    cmd = args.cmd or "chat"

    if cmd == "chat":
        max_chars = getattr(args, "max_chars", 400)
        temp = getattr(args, "temperature", 0.8)
        interactive_chat(agent_data, w, kind,
                        use_full_pipeline=use_full_pipeline,
                        max_chars=max_chars, temperature=temp)
    elif cmd == "ask":
        query = " ".join(args.query)
        return single_ask(agent_data, w, kind, query,
                         use_full_pipeline=use_full_pipeline,
                         max_chars=getattr(args, "max_chars", 400),
                         temperature=getattr(args, "temperature", 0.8))
    elif cmd == "code":
        task = " ".join(args.task)
        return single_code(agent_data, w, kind, task,
                          use_full_pipeline=use_full_pipeline,
                          max_chars=getattr(args, "max_chars", 500),
                          temperature=getattr(args, "temperature", 0.6))
    elif cmd == "think":
        topic = " ".join(args.topic)
        return single_think(agent_data, w, kind, topic,
                           use_full_pipeline=use_full_pipeline,
                           max_chars=getattr(args, "max_chars", 600),
                           temperature=getattr(args, "temperature", 0.7))
    elif cmd == "stats":
        _print_stats(kind, w)
    elif cmd == "test":
        return cmd_test(agent_data, w, kind, use_full_pipeline=use_full_pipeline)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
