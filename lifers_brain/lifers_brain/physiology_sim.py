"""
生理仿真：工程类比标量 + 离散时间步（非医学、非诊断）。

enabled=true 时：每轮对话根据 idle 与用户输入更新三维状态，持久化到 state/physiology_sim.json，
并向 SYSTEM 注入路线图 + 当前标量（供本地小模型调节语气/节奏，不构成生理监测结论）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class PhysioState:
    """抽象仿真状态，范围 [0,1]。"""

    arousal: float = 0.42
    fatigue: float = 0.28
    recovery: float = 0.70
    schema_version: int = 1


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _state_path(root: Path) -> Path:
    return root / "state" / "physiology_sim.json"


def load_physio_state(root: Path) -> PhysioState:
    p = _state_path(root)
    if not p.is_file():
        return PhysioState()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return PhysioState(
            arousal=_clamp01(float(raw.get("arousal", 0.42))),
            fatigue=_clamp01(float(raw.get("fatigue", 0.28))),
            recovery=_clamp01(float(raw.get("recovery", 0.70))),
            schema_version=int(raw.get("schema_version", 1)),
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return PhysioState()


def save_physio_state(root: Path, st: PhysioState) -> None:
    p = _state_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(asdict(st), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def physiology_sim_cfg(stack: Dict[str, Any]) -> Dict[str, Any]:
    d = stack.get("physiology_sim") or {}
    try:
        cap = int(d.get("max_inject_chars", 2400) or 2400)
    except (TypeError, ValueError):
        cap = 2400
    dyn = d.get("dynamics") if isinstance(d.get("dynamics"), dict) else {}
    return {
        "enabled": bool(d.get("enabled", False)),
        "max_inject_chars": max(200, min(cap, 32000)),
        "roadmap": str(d.get("roadmap_for_llm") or "").strip(),
        "run_dynamics": bool(d.get("run_dynamics", True)),
        "dynamics": {
            "idle_recovery_rate_per_sec": float(dyn.get("idle_recovery_rate_per_sec", 3.5e-4)),
            "idle_fatigue_decay_per_sec": float(dyn.get("idle_fatigue_decay_per_sec", 4.5e-4)),
            "idle_arousal_decay_per_sec": float(dyn.get("idle_arousal_decay_per_sec", 2.5e-4)),
            "interaction_arousal_bump": float(dyn.get("interaction_arousal_bump", 0.055)),
            "interaction_fatigue_bump": float(dyn.get("interaction_fatigue_bump", 0.038)),
            "interaction_recovery_dip": float(dyn.get("interaction_recovery_dip", 0.018)),
            "idle_cap_sec": float(dyn.get("idle_cap_sec", 7200.0)),
        },
    }


DEFAULT_ROADMAP = """离散仿真：arousal / fatigue / recovery 三标量；每步先按 idle 松弛，再按用户回合施加交互项。
可与 instincts 的空闲阈值直觉对齐（非同一套参数）。声明：标量为软件仿真，非人体测量。"""


def _tick(st: PhysioState, idle_sec: float, user_nonempty: bool, dyn: Dict[str, float]) -> PhysioState:
    dt = max(0.0, float(idle_sec))
    cap = max(60.0, float(dyn.get("idle_cap_sec", 7200.0)))
    dt = min(dt, cap)

    rr = float(dyn["idle_recovery_rate_per_sec"])
    fd = float(dyn["idle_fatigue_decay_per_sec"])
    ad = float(dyn["idle_arousal_decay_per_sec"])

    # 离开键盘越久：疲劳略降、恢复略升、觉醒略降（抽象松弛）
    st.recovery = _clamp01(st.recovery + rr * dt * (1.0 - st.recovery))
    st.fatigue = _clamp01(st.fatigue - fd * dt)
    st.arousal = _clamp01(st.arousal - ad * dt)

    if user_nonempty:
        st.arousal = _clamp01(st.arousal + float(dyn["interaction_arousal_bump"]))
        st.fatigue = _clamp01(st.fatigue + float(dyn["interaction_fatigue_bump"]))
        st.recovery = _clamp01(st.recovery - float(dyn["interaction_recovery_dip"]))

    return st


def update_physiology_and_format(root: Path, stack: Dict[str, Any], idle_sec: float, user_nonempty: bool) -> str:
    """
    若未启用则返回空串；否则 tick、落盘并返回注入 SYSTEM 的文本。
    """
    cfg = physiology_sim_cfg(stack)
    if not cfg["enabled"]:
        return ""

    roadmap = cfg["roadmap"] or DEFAULT_ROADMAP
    lines = ["PHYSIOLOGY_SIM_ROADMAP:\n" + roadmap.strip()]

    if cfg["run_dynamics"]:
        st = load_physio_state(root)
        st = _tick(st, idle_sec, user_nonempty, cfg["dynamics"])
        save_physio_state(root, st)
        lines.append(
            "PHYSIO_STATE (simulation, not medical):\n"
            f"  arousal={st.arousal:.3f} fatigue={st.fatigue:.3f} recovery={st.recovery:.3f}\n"
            "  hint: higher fatigue -> shorter replies optional; higher recovery -> calmer tone optional.\n"
        )
    else:
        lines.append("PHYSIO_STATE: dynamics disabled (run_dynamics=false); roadmap only.\n")

    text = "\n".join(lines) + "\n"
    cap = cfg["max_inject_chars"]
    if len(text) > cap:
        return text[: cap - 40].rstrip() + "\n…(physiology_sim truncated)\n"
    return text


def format_physiology_sim_context(stack: Dict[str, Any]) -> str:
    """兼容旧调用：仅渲染路线图（无状态）；新建 Agent 请用 update_physiology_and_format。"""
    cfg = physiology_sim_cfg(stack)
    if not cfg["enabled"]:
        return ""
    body = cfg["roadmap"] or DEFAULT_ROADMAP
    text = "PHYSIOLOGY_SIM_ROADMAP:\n" + body.strip() + "\n"
    if len(text) > cfg["max_inject_chars"]:
        c = cfg["max_inject_chars"]
        return text[: c - 36].rstrip() + "\n…(physiology_sim truncated)\n"
    return text
