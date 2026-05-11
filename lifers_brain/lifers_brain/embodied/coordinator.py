"""Physical step + vision + heuristic decision; respects weights/.train_control when enabled.

多体 / NPC 扩展以 `stack.embodied_world.dynamic_npc` 与 `PhysWorld` 为准；入口仅本模块 +
`scripts/embodied_tick_once.py`，不在 `LifersAgent` 或 taskflow 内重复物化 tick。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from lifers_brain.embodied.physics import PhysWorld
from lifers_brain.embodied.vision import VisionSummary, observe
from lifers_brain.stack_env import load_stack
from lifers_brain.train_control import control_file_path, read_train_control


@dataclass
class EmbodiedState:
    thrust: float
    yaw_rate: float
    tick: int


def _default_obstacles(w: float, h: float) -> list[tuple[float, float, float]]:
    return [(w * 0.35, h * 0.45, 0.25), (w * 0.72, h * 0.30, 0.18)]


def _state_path(root: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (root / p)


def load_embodied_bundle(root: Path) -> Tuple[Dict[str, Any], PhysWorld, EmbodiedState]:
    stack = load_stack(root)
    cfg = stack.get("embodied_world") or {}
    rel = str(cfg.get("state_relpath") or "state/embodied_world.json").strip() or "state/embodied_world.json"
    path = _state_path(root, rel)
    world: PhysWorld
    ctrl = EmbodiedState(thrust=0.35, yaw_rate=0.0, tick=0)
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            world = PhysWorld.from_dict(raw.get("world") or {})
            c = raw.get("control") or {}
            ctrl = EmbodiedState(
                thrust=float(c.get("thrust", 0.35)),
                yaw_rate=float(c.get("yaw_rate", 0.0)),
                tick=int(c.get("tick", 0)),
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            world = PhysWorld(obstacles=_default_obstacles(4, 4))
    else:
        world = PhysWorld(obstacles=_default_obstacles(4, 4))
    return cfg, world, ctrl


def save_embodied_bundle(root: Path, cfg: Dict[str, Any], world: PhysWorld, ctrl: EmbodiedState) -> None:
    rel = str(cfg.get("state_relpath") or "state/embodied_world.json").strip() or "state/embodied_world.json"
    path = _state_path(root, rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"world": world.to_dict(), "control": asdict(ctrl)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _heuristic_policy(vision: VisionSummary, world: PhysWorld) -> Tuple[float, float]:
    """Return thrust, yaw_rate from coarse vision + pose."""
    yaw = 0.4 * (vision.brightness - 0.5) + 0.15 * (world.body.x - world.width * 0.5)
    thrust = 0.55 + 0.35 * (0.5 - vision.brightness)
    thrust = max(0.05, min(1.2, thrust))
    yaw = max(-1.8, min(1.8, yaw))
    return thrust, yaw


def run_embodied_tick(root: Path) -> Dict[str, Any]:
    cfg, world, ctrl = load_embodied_bundle(root)
    now_ms = int(time.time() * 1000)
    if not (cfg.get("enabled") is True or str(cfg.get("enabled")).lower() == "true"):
        return {"ok": True, "skipped": True, "reason": "embodied_world.enabled is false", "as_of_unix_ms": now_ms}

    ctl = control_file_path(root / "weights")
    mode = read_train_control(ctl)
    if mode == "stop":
        return {"ok": True, "skipped": True, "reason": "train_control=stop", "control": mode, "as_of_unix_ms": now_ms}
    if mode == "pause":
        return {"ok": True, "skipped": True, "reason": "train_control=pause", "control": mode, "as_of_unix_ms": now_ms}

    dt = float(cfg.get("dt_sec") or 0.05)
    vision = observe(cfg, root)
    pol = str((cfg.get("decision") or {}).get("policy") or "heuristic_v1")
    if pol == "heuristic_v1":
        thrust, yaw = _heuristic_policy(vision, world)
    else:
        thrust, yaw = ctrl.thrust, ctrl.yaw_rate

    world.step(dt, thrust=thrust, yaw_rate=yaw)
    ctrl.thrust, ctrl.yaw_rate = thrust, yaw
    ctrl.tick += 1
    save_embodied_bundle(root, cfg, world, ctrl)

    return {
        "ok": True,
        "tick": ctrl.tick,
        "vision": asdict(vision),
        "body": world.to_dict()["body"],
        "policy": pol,
        "control": mode,
        "as_of_unix_ms": int(time.time() * 1000),
        "realtime_note": "物化步为当次 tick 的快照；与 Agents 对话时钟独立；dynamic_npc 多体未接时仅单刚体。",
    }


class EmbodiedCoordinator:
    """OO façade for tests / future robot bridge."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def tick(self) -> Dict[str, Any]:
        return run_embodied_tick(self.root)
