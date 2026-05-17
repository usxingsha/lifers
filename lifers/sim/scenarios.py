"""
Lifers World — 仿真场景库
任务场景、程序化世界生成、基准评估
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# World Model
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WorldObject:
    id: str
    obj_type: str  # cube, sphere, table, door, button, light, robot, human
    position: Tuple[float, float, float] = (0, 0, 0)
    rotation: Tuple[float, float, float] = (0, 0, 0)
    size: Tuple[float, float, float] = (1, 1, 1)
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    props: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldConfig:
    name: str
    width: float = 20.0
    height: float = 20.0
    objects: List[WorldObject] = field(default_factory=list)
    lights: List[Dict] = field(default_factory=list)
    gravity: Tuple[float, float, float] = (0, 0, -9.81)


class WorldBuilder:
    """Procedural world/scenario generation."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.RandomState(seed)

    def empty_room(self, width: float = 10, height: float = 10) -> WorldConfig:
        walls = []
        wall_thickness = 0.2
        # Floor
        walls.append(WorldObject("floor", "plane", (0, 0, 0), (0, 0, 0), (width, height, 0.1), (0.3, 0.3, 0.3)))
        # Walls
        for i, (x, y, w, h) in enumerate([
            (0, height / 2, width, wall_thickness),
            (0, -height / 2, width, wall_thickness),
            (-width / 2, 0, wall_thickness, height),
            (width / 2, 0, wall_thickness, height),
        ]):
            walls.append(WorldObject(f"wall_{i}", "wall", (x, y, 1), (0, 0, 0), (w, h, 2), (0.7, 0.7, 0.7)))
        return WorldConfig(name="empty_room", width=width, height=height, objects=walls)

    def navigation_obstacle_course(self, n_obstacles: int = 10) -> WorldConfig:
        world = self.empty_room(15, 15)
        for i in range(n_obstacles):
            x = self._rng.uniform(-6, 6)
            y = self._rng.uniform(-6, 6)
            s = self._rng.uniform(0.3, 1.5)
            world.objects.append(WorldObject(
                f"obs_{i}", self._rng.choice(["cube", "sphere"]),
                (x, y, s / 2), (0, 0, 0), (s, s, s),
                tuple(self._rng.uniform(0.2, 0.9, 3)),
            ))
        # Start and goal markers
        world.objects.append(WorldObject("start", "marker", (-6, -6, 0.05), color=(0, 1, 0), size=(0.5, 0.5, 0.05)))
        world.objects.append(WorldObject("goal", "marker", (6, 6, 0.05), color=(1, 0, 0), size=(0.5, 0.5, 0.05)))
        return world

    def manipulation_table(self) -> WorldConfig:
        world = WorldConfig(name="manipulation_table", width=3, height=3)
        world.objects.append(WorldObject("table", "table", (0, 0, 0.5), size=(2, 1, 1), color=(0.5, 0.3, 0.2)))
        objects = [
            ("cube_a", "cube", (-0.3, 0.2, 1.05), (0.1, 0.1, 0.1), (1, 0, 0)),
            ("cube_b", "cube", (0.3, -0.1, 1.05), (0.08, 0.08, 0.08), (0, 0, 1)),
        ]
        for oid, otype, pos, size, color in objects:
            world.objects.append(WorldObject(oid, otype, pos, size=size, color=color))
        return world

    def social_scene(self, n_humans: int = 3) -> WorldConfig:
        world = self.empty_room(10, 10)
        names = ["Alice", "Bob", "Carol", "Dave", "Eve"]
        for i in range(min(n_humans, len(names))):
            x = self._rng.uniform(-3, 3)
            y = self._rng.uniform(-3, 3)
            world.objects.append(WorldObject(
                f"human_{names[i]}", "human", (x, y, 1),
                props={"name": names[i], "pose": "standing", "emotion": self._rng.choice(["neutral", "happy", "curious"])},
                color=(0.8, 0.6, 0.4),
            ))
        return world


# ═══════════════════════════════════════════════════════════════════════════════
# Scenarios
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    id: str
    name: str
    description: str
    category: str  # navigation, manipulation, dialogue, exploration
    world: WorldConfig
    initial_state: Dict[str, Any] = field(default_factory=dict)
    success_conditions: List[Dict[str, Any]] = field(default_factory=list)
    max_steps: int = 200
    difficulty: str = "medium"


class ScenarioLibrary:
    """Pre-built scenario library for Lifers evaluation."""

    def __init__(self) -> None:
        self._scenarios: Dict[str, Scenario] = {}
        self._builder = WorldBuilder()
        self._register_defaults()

    def _register_defaults(self) -> None:
        # Navigation
        self.add(Scenario(
            id="nav_obstacle", name="障碍导航", category="navigation",
            description="在有障碍物的房间内从起点导航到目标点",
            world=self._builder.navigation_obstacle_course(15),
            success_conditions=[{"type": "distance_to_goal", "threshold": 0.5}],
            max_steps=300, difficulty="easy",
        ))
        self.add(Scenario(
            id="nav_maze", name="迷宫探索", category="navigation",
            description="在密集障碍中导航（仿迷宫）",
            world=self._builder.navigation_obstacle_course(25),
            success_conditions=[{"type": "distance_to_goal", "threshold": 0.5}],
            max_steps=500, difficulty="hard",
        ))
        # Manipulation
        self.add(Scenario(
            id="man_pick_place", name="拾取放置", category="manipulation",
            description="拾取立方体A并放到指定位置",
            world=self._builder.manipulation_table(),
            success_conditions=[{"type": "object_at_position", "object": "cube_a", "position": [0, 0.3, 1.05]}],
            max_steps=150, difficulty="medium",
        ))
        # Social / Dialogue
        self.add(Scenario(
            id="social_greet", name="社交问候", category="dialogue",
            description="在房间内识别人类并打招呼",
            world=self._builder.social_scene(3),
            success_conditions=[{"type": "greeted_all_humans"}],
            max_steps=100, difficulty="easy",
        ))
        # Exploration
        self.add(Scenario(
            id="explore_unknown", name="未知探索", category="exploration",
            description="探索未知环境，构建地图",
            world=self._builder.empty_room(20, 20),
            success_conditions=[{"type": "coverage_percent", "threshold": 0.8}],
            max_steps=400, difficulty="medium",
        ))

    def add(self, scenario: Scenario) -> None:
        self._scenarios[scenario.id] = scenario

    def get(self, scenario_id: str) -> Optional[Scenario]:
        return self._scenarios.get(scenario_id)

    def list_by_category(self, category: str) -> List[Scenario]:
        return [s for s in self._scenarios.values() if s.category == category]

    def all(self) -> List[Scenario]:
        return list(self._scenarios.values())


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    scenario_id: str
    success: bool
    steps_taken: int
    final_state: Dict[str, Any]
    metrics: Dict[str, float]
    duration_ms: int


class BenchmarkRunner:
    """Run scenarios and collect evaluation results."""

    def __init__(self, library: Optional[ScenarioLibrary] = None) -> None:
        self.library = library or ScenarioLibrary()

    def evaluate_scenario(
        self,
        scenario_id: str,
        agent_fn: callable,  # (observation) -> action
    ) -> EvalResult:
        scenario = self.library.get(scenario_id)
        if scenario is None:
            raise ValueError(f"Unknown scenario: {scenario_id}")
        t0 = int(time.time() * 1000)
        observations = []
        step = 0
        success = False
        for step in range(scenario.max_steps):
            obs = self._get_observation(scenario, step)
            observations.append(obs)
            action = agent_fn(obs)
            self._apply_action(scenario, action)
            if self._check_success(scenario):
                success = True
                break
        return EvalResult(
            scenario_id=scenario_id,
            success=success,
            steps_taken=step + 1,
            final_state={"observations": len(observations)},
            metrics=self._compute_metrics(scenario, success, step + 1),
            duration_ms=int(time.time() * 1000) - t0,
        )

    def _get_observation(self, scenario: Scenario, step: int) -> Dict:
        return {"step": step, "world_objects": len(scenario.world.objects)}

    def _apply_action(self, scenario: Scenario, action: Any) -> None:
        pass  # stub

    def _check_success(self, scenario: Scenario) -> bool:
        return False  # stub — to be implemented per success condition type

    def _compute_metrics(self, scenario: Scenario, success: bool, steps: int) -> Dict[str, float]:
        return {
            "success": 1.0 if success else 0.0,
            "efficiency": (1.0 - steps / scenario.max_steps) if success else 0.0,
            "steps_normalized": steps / max(scenario.max_steps, 1),
        }

    def run_all(self, agent_fn: callable, category: Optional[str] = None) -> List[EvalResult]:
        scenarios = self.library.all()
        if category:
            scenarios = [s for s in scenarios if s.category == category]
        return [self.evaluate_scenario(s.id, agent_fn) for s in scenarios]
