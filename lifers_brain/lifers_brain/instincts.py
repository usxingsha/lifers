"""
本能层（自动、无需口令）：日常聊天姿态、内务整理、反思（思考）、睡眠（记忆巩固）。

与 runtime 正交：任意宿主均可启用；配置见 config/stack.json → instincts。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from .memory import MemoryItem


@dataclass
class InstinctState:
    prev_user_ts_ms: int = 0
    user_turn_count: int = 0
    last_instinct_bucket: int = 0


def _state_path(root: Path) -> Path:
    return root / "state" / "instincts_state.json"


def load_instinct_state(root: Path) -> InstinctState:
    p = _state_path(root)
    if not p.is_file():
        return InstinctState()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return InstinctState(
            prev_user_ts_ms=int(raw.get("prev_user_ts_ms", 0)),
            user_turn_count=int(raw.get("user_turn_count", 0)),
            last_instinct_bucket=int(raw.get("last_instinct_bucket", 0)),
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return InstinctState()


def save_instinct_state(root: Path, st: InstinctState) -> None:
    p = _state_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(st), ensure_ascii=False, indent=2), encoding="utf-8")


def instinct_cfg(stack: Dict[str, Any]) -> Dict[str, Any]:
    d = stack.get("instincts") or {}
    return {
        "enabled": bool(d.get("enabled", True)),
        "sleep_after_idle_sec": float(d.get("sleep_after_idle_sec", 3600)),
        "think_after_idle_sec": float(d.get("think_after_idle_sec", 420)),
        "idle_chat_soft_sec": float(d.get("idle_chat_soft_sec", 180)),
        "presence_reset_idle_sec": float(d.get("presence_reset_idle_sec", 120)),
        "micro_think_every_user_turns": int(d.get("micro_think_every_user_turns", 6)),
        "operations_trim_scratchpad": bool(d.get("operations_trim_scratchpad", True)),
        "operations_max_scratch_items": int(d.get("operations_max_scratch_items", 24)),
        "sleep_consolidate_to_longterm": bool(d.get("sleep_consolidate_to_longterm", True)),
        "sleep_compact_session": bool(d.get("sleep_compact_session", True)),
        "notes_in_context": bool(d.get("notes_in_context", True)),
        "use_local_brain_for_instincts": bool(d.get("use_local_brain_for_instincts", True)),
    }


def _idle_bucket(idle_sec: float, icfg: Dict[str, Any]) -> int:
    if idle_sec >= icfg["sleep_after_idle_sec"]:
        return 3
    if idle_sec >= icfg["think_after_idle_sec"]:
        return 2
    if idle_sec >= icfg["idle_chat_soft_sec"]:
        return 1
    return 0


def tick_instincts_start(agent: Any, idle_sec: float, stack: Dict[str, Any]) -> List[str]:
    """
    在用户本轮输入处理前调用。返回本回合注入上下文的简短本能提示（中文）。
    """
    icfg = instinct_cfg(stack)
    notes: List[str] = []
    if not icfg["enabled"]:
        return notes

    root = agent.cfg.root_dir
    st = getattr(agent, "_instinct_state", None)
    if st is None:
        st = load_instinct_state(root)
        agent._instinct_state = st

    # 短间隔对话：重置阶梯，便于下一次「离开很久」再触发本能
    if idle_sec < icfg["presence_reset_idle_sec"]:
        st.last_instinct_bucket = 0

    if idle_sec <= 0:
        return notes

    b = _idle_bucket(idle_sec, icfg)
    if b <= st.last_instinct_bucket:
        # 同一轮「长时间离线」只向上触发一次；再次进来已由 presence_reset 清掉
        pass
    elif b >= 3:
        notes.extend(_run_sleep(agent, stack, icfg))
        st.last_instinct_bucket = 3
    elif b >= 2:
        notes.extend(_run_think_idle(agent, stack, icfg, idle_sec))
        st.last_instinct_bucket = 2
    elif b >= 1:
        notes.append("【本能·日常】对方有一段时间未发言，语气可更自然、简短、像日常寒暄那样接过话题。")
        st.last_instinct_bucket = 1

    _run_operations(agent, icfg)
    save_instinct_state(root, st)
    return notes


def tick_instincts_end(agent: Any, stack: Dict[str, Any], user_input: str) -> None:
    """回合结束：微反思（高频）、更新计数。"""
    icfg = instinct_cfg(stack)
    if not icfg["enabled"]:
        return

    root = agent.cfg.root_dir
    st: InstinctState = getattr(agent, "_instinct_state", None) or load_instinct_state(root)
    agent._instinct_state = st
    st.user_turn_count += 1
    n = icfg["micro_think_every_user_turns"]
    if n > 0 and st.user_turn_count % n == 0:
        text = _micro_think(agent, user_input, icfg)
        if text and "(missing weights)" not in text:
            agent.scratch.add(
                MemoryItem(
                    type="reflection",
                    content={"kind": "micro_think", "text": text},
                    importance=0.25,
                    source="instinct:micro_think",
                    ts_ms=int(time.time() * 1000),
                )
            )
    save_instinct_state(root, st)


def sync_prev_user_ts(agent: Any, ts_ms: int) -> None:
    root = agent.cfg.root_dir
    st: InstinctState = getattr(agent, "_instinct_state", None) or load_instinct_state(root)
    st.prev_user_ts_ms = ts_ms
    agent._instinct_state = st
    save_instinct_state(root, st)


def _run_sleep(agent: Any, stack: Dict[str, Any], icfg: Dict[str, Any]) -> List[str]:
    notes: List[str] = []
    blob = ""
    if icfg["sleep_compact_session"]:
        blob = agent.session.sleep_compact()
    else:
        blob = agent.session.context_text()[:4000]

    if icfg["sleep_consolidate_to_longterm"] and blob.strip():
        agent.longterm.add(
            MemoryItem(
                type="instinct",
                content={
                    "kind": "sleep_consolidation",
                    "idle_note": "offline_sleep",
                    "session_excerpt": blob[:3500],
                },
                importance=0.55,
                source="instinct:sleep",
                ts_ms=int(time.time() * 1000),
            )
        )
    notes.append("【本能·睡眠】离线间隔较长；已将会话要点卷入长期记忆并收紧短期窗口（类比睡眠巩固）。")

    try:
        from .audit import audit_log

        audit_log({"event": "instinct_sleep", "chars": len(blob)})
    except Exception:
        pass
    return notes


def _run_think_idle(agent: Any, stack: Dict[str, Any], icfg: Dict[str, Any], idle_sec: float) -> List[str]:
    prompt = (
        f"用两到三句中文写一段内心反思：刚才可能有较长间隔（约{int(idle_sec)}秒）。"
        "回顾对话意图、未竟事宜；不说教。"
    )
    text = _brain_line(agent, prompt, icfg)
    if not text.strip():
        return []
    agent.longterm.add(
        MemoryItem(
            type="reflection",
            content={"kind": "idle_think", "text": text, "idle_sec": idle_sec},
            importance=0.42,
            source="instinct:think",
            ts_ms=int(time.time() * 1000),
        )
    )
    agent.scratch.add(
        MemoryItem(
            type="reflection",
            content={"kind": "idle_think_scratch", "text": text[:400]},
            importance=0.28,
            source="instinct:think",
            ts_ms=int(time.time() * 1000),
        )
    )
    return ["【本能·思考】刚做过一段离线反思，已写入长期记忆。"]


def _micro_think(agent: Any, user_input: str, icfg: Dict[str, Any]) -> str:
    prompt = f"用不超过45字写一句内心独白（反思对话节奏，不说教）：用户最近一句：{user_input[:120]}"
    return _brain_line(agent, prompt, icfg)


def _brain_line(agent: Any, prompt: str, icfg: Dict[str, Any]) -> str:
    if not icfg["use_local_brain_for_instincts"]:
        return ""
    try:
        out = agent.brain.generate(prompt).strip()
        if "(missing weights)" in out:
            return "（本能）本地权重未就绪；跳过生成。"
        return out[:500]
    except Exception:
        return ""


def _run_operations(agent: Any, icfg: Dict[str, Any]) -> None:
    if not icfg["operations_trim_scratchpad"]:
        return
    mx = max(4, icfg["operations_max_scratch_items"])
    items = agent.scratch.items()
    if len(items) <= mx:
        return
    # 丢弃最旧的若干条 tool_result，保留最近的反思等
    keep = items[-mx:]
    agent.scratch.clear()
    for it in keep:
        agent.scratch.add(it)


def instinct_notes_for_context(agent: Any, stack: Dict[str, Any]) -> str:
    icfg = instinct_cfg(stack)
    if not icfg["notes_in_context"]:
        return ""
    notes: List[str] = getattr(agent, "_instinct_turn_notes", []) or []
    if not notes:
        return ""
    return "\n".join(notes)
