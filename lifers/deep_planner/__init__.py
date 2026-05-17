"""
Lifers DeepPlan — 分层规划与蒙特卡洛树搜索
纯 NumPy 实现，品牌化推理引擎
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ── Plan / Task representations ───────────────────────────────────────────────

@dataclass
class Task:
    id: str
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    priority: float = 0.5
    deadline_ms: int = 0
    parent: Optional[str] = None
    status: str = "pending"  # pending, in_progress, done, failed, blocked


@dataclass
class Plan:
    id: str
    goal: str
    tasks: List[Task]
    created_ms: int = 0
    score: float = 0.0


# ── HTN Planner ───────────────────────────────────────────────────────────────

class HTNPlanner:
    """Hierarchical Task Network: decompose high-level goals into primitive tasks."""

    def __init__(self) -> None:
        self._methods: Dict[str, List[Tuple[str, List[Task], Callable]]] = {}
        self._primitives: Dict[str, Callable] = {}
        self._seed()

    def _seed(self) -> None:
        """Built-in decomposition rules."""
        # Navigation
        self.add_method("navigate_to", self._decompose_navigate)
        self._primitives["sense_position"] = lambda **kw: {"x": 0, "y": 0}
        self._primitives["plan_path"] = lambda **kw: {"waypoints": [(0, 0), (1, 1)]}
        self._primitives["move_step"] = lambda **kw: {"moved": True}
        self._primitives["check_arrived"] = lambda **kw: {"arrived": True}

        # Manipulation
        self.add_method("manipulate_object", self._decompose_manipulate)
        self._primitives["locate_object"] = lambda **kw: {"found": True, "x": 0}
        self._primitives["approach_object"] = lambda **kw: {"approached": True}
        self._primitives["grasp_object"] = lambda **kw: {"grasped": True}
        self._primitives["release_object"] = lambda **kw: {"released": True}

        # Query/Research
        self.add_method("answer_question", self._decompose_research)
        self._primitives["understand_query"] = lambda **kw: {"intent": "factual"}
        self._primitives["search_memory"] = lambda **kw: {"results": []}
        self._primitives["synthesize_answer"] = lambda **kw: {"answer": ""}
        self._primitives["verify_answer"] = lambda **kw: {"confidence": 0.8}

        # Self-improvement
        self.add_method("self_improve", self._decompose_self_improve)
        self._primitives["reflect_on_mistakes"] = lambda **kw: {"insights": []}
        self._primitives["identify_skill_gap"] = lambda **kw: {"gap": None}
        self._primitives["practice_skill"] = lambda **kw: {"improved": True}

    def add_method(self, goal_type: str, decomposer: Callable) -> None:
        self._methods[goal_type] = decomposer

    def plan(self, goal: str, goal_type: str = "default", params: Optional[Dict] = None) -> Plan:
        tasks: List[Task] = []
        decomposer = self._methods.get(goal_type, self._default_decompose)
        decomposer(goal, params or {}, tasks, 0)
        return Plan(
            id=f"plan_{int(time.time()*1000)}",
            goal=goal,
            tasks=tasks,
            created_ms=int(time.time() * 1000),
        )

    # ── Decomposers ───────────────────────────────────────────────────

    def _decompose_navigate(self, goal: str, params: dict, tasks: list, depth: int) -> None:
        tid = lambda n: f"{goal}_{n}_{depth}"
        tasks.append(Task(id=tid("sense"), name="sense_position", params=params, priority=1.0))
        tasks.append(Task(id=tid("path"), name="plan_path", params=params, priority=0.9))
        tasks.append(Task(id=tid("move"), name="move_step", params=params, priority=0.8))
        tasks.append(Task(id=tid("check"), name="check_arrived", params=params, priority=0.7))

    def _decompose_manipulate(self, goal: str, params: dict, tasks: list, depth: int) -> None:
        tid = lambda n: f"{goal}_{n}_{depth}"
        tasks.append(Task(id=tid("loc"), name="locate_object", params=params))
        tasks.append(Task(id=tid("app"), name="approach_object", params=params))
        tasks.append(Task(id=tid("grasp"), name="grasp_object", params=params))
        tasks.append(Task(id=tid("rel"), name="release_object", params=params))

    def _decompose_research(self, goal: str, params: dict, tasks: list, depth: int) -> None:
        tid = lambda n: f"{goal}_{n}_{depth}"
        tasks.append(Task(id=tid("und"), name="understand_query", params=params))
        tasks.append(Task(id=tid("srch"), name="search_memory", params=params))
        tasks.append(Task(id=tid("syn"), name="synthesize_answer", params=params))
        tasks.append(Task(id=tid("ver"), name="verify_answer", params=params))

    def _decompose_self_improve(self, goal: str, params: dict, tasks: list, depth: int) -> None:
        tid = lambda n: f"{goal}_{n}_{depth}"
        tasks.append(Task(id=tid("ref"), name="reflect_on_mistakes", params=params))
        tasks.append(Task(id=tid("gap"), name="identify_skill_gap", params=params))
        tasks.append(Task(id=tid("prac"), name="practice_skill", params=params))

    def _default_decompose(self, goal: str, params: dict, tasks: list, depth: int) -> None:
        tasks.append(Task(id=f"{goal}_exec", name="execute", params={"goal": goal, **params}))

    def execute_plan(self, plan: Plan) -> List[Dict[str, Any]]:
        results = []
        for task in plan.tasks:
            if task.name in self._primitives:
                try:
                    result = self._primitives[task.name](**task.params)
                    task.status = "done"
                    results.append({"task": task.id, "status": "done", "result": result})
                except Exception as e:
                    task.status = "failed"
                    results.append({"task": task.id, "status": "failed", "error": str(e)})
            else:
                task.status = "blocked"
                results.append({"task": task.id, "status": "blocked"})
        return results


# ── MCTS ──────────────────────────────────────────────────────────────────────

class MCTSNode:
    def __init__(self, state: Any, parent: Optional["MCTSNode"] = None, action: Optional[str] = None) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List[MCTSNode] = []
        self.visits: int = 0
        self.value: float = 0.0
        self.untried_actions: List[str] = []

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0 and len(self.children) > 0


class MCTS:
    """Monte Carlo Tree Search for decision making under uncertainty."""

    def __init__(self, exploration_weight: float = 1.414, max_iterations: int = 200, max_depth: int = 20) -> None:
        self.C = exploration_weight  # UCB1 exploration constant
        self.max_iter = max_iterations
        self.max_depth = max_depth
        self._rng = np.random.RandomState()

    def search(
        self,
        initial_state: Any,
        get_actions: Callable[[Any], List[str]],
        simulate: Callable[[Any, str], Tuple[Any, float]],
        is_terminal: Callable[[Any], bool] = lambda s: False,
    ) -> Tuple[str, List[Tuple[str, float, int]]]:
        root = MCTSNode(state=initial_state)
        root.untried_actions = get_actions(initial_state)

        for _ in range(self.max_iter):
            node = self._select(root)
            if not is_terminal(node.state) and node.untried_actions:
                node = self._expand(node, get_actions)
            reward = self._rollout(node, simulate, is_terminal, get_actions)
            self._backpropagate(node, reward)

        # Return best action + stats
        stats = [(c.action, c.value / max(c.visits, 1), c.visits) for c in root.children]
        stats.sort(key=lambda x: x[1], reverse=True)
        best = stats[0][0] if stats else ""
        return best, stats

    def _select(self, node: MCTSNode) -> MCTSNode:
        while not node.is_leaf and node.is_fully_expanded:
            node = self._best_child(node)
        return node

    def _expand(self, node: MCTSNode, get_actions: Callable) -> MCTSNode:
        action = node.untried_actions.pop(0)
        child = MCTSNode(state=node.state, parent=node, action=action)
        node.children.append(child)
        child.untried_actions = get_actions(child.state)
        return child

    def _rollout(self, node: MCTSNode, simulate: Callable, is_terminal: Callable, get_actions: Callable) -> float:
        state = node.state
        total_reward = 0.0
        for _ in range(self.max_depth):
            if is_terminal(state):
                break
            actions = get_actions(state)
            if not actions:
                break
            action = self._rng.choice(actions)
            state, reward = simulate(state, action)
            total_reward += reward
        return total_reward

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _best_child(self, node: MCTSNode) -> MCTSNode:
        best = None
        best_score = -float("inf")
        for c in node.children:
            exploit = c.value / max(c.visits, 1)
            explore = self.C * math.sqrt(math.log(node.visits) / max(c.visits, 1))
            score = exploit + explore
            if score > best_score:
                best_score = score
                best = c
        return best if best is not None else node.children[0]


# ── Reflection Engine ─────────────────────────────────────────────────────────

class ReflectionEngine:
    """Post-action self-reflection for continuous improvement."""

    def __init__(self, max_history: int = 100) -> None:
        self._history: List[Dict[str, Any]] = []
        self._max_history = max_history
        self._patterns: Dict[str, List[float]] = {}  # action -> outcomes

    def record(self, action: str, expected: Any, actual: Any, context: str = "") -> None:
        delta = self._compute_delta(expected, actual)
        entry = {
            "ts_ms": int(time.time() * 1000),
            "action": action,
            "expected": expected,
            "actual": actual,
            "delta": delta,
            "context": context,
        }
        self._history.append(entry)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        if action not in self._patterns:
            self._patterns[action] = []
        self._patterns[action].append(delta)
        if len(self._patterns[action]) > 50:
            self._patterns[action] = self._patterns[action][-50:]

    def reflect(self) -> List[Dict[str, Any]]:
        """Analyze recent history for insights."""
        if len(self._history) < 3:
            return []
        insights = []
        recent = self._history[-10:]
        # Trend: are outcomes improving?
        deltas = [e["delta"] for e in recent]
        if len(deltas) >= 3:
            trend = np.mean(deltas[-3:]) - np.mean(deltas[:3])
            if abs(trend) > 0.1:
                insights.append({
                    "type": "trend",
                    "direction": "improving" if trend < 0 else "worsening",
                    "magnitude": float(trend),
                })
        # Patterns: which actions consistently underperform?
        for action, ds in self._patterns.items():
            if len(ds) >= 5 and np.mean(ds) > 0.3:
                insights.append({
                    "type": "weak_action",
                    "action": action,
                    "mean_delta": float(np.mean(ds)),
                })
        return insights

    def _compute_delta(self, expected: Any, actual: Any) -> float:
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return abs(float(expected) - float(actual)) / max(abs(float(expected)), 1e-6)
        if isinstance(expected, dict) and isinstance(actual, dict):
            keys = set(expected.keys()) | set(actual.keys())
            if not keys:
                return 0.0
            diffs = []
            for k in keys:
                ev = expected.get(k, 0)
                av = actual.get(k, 0)
                if isinstance(ev, (int, float)):
                    diffs.append(abs(float(ev) - float(av)) / max(abs(float(ev)), 1e-6))
                else:
                    diffs.append(1.0 if ev != av else 0.0)
            return float(np.mean(diffs)) if diffs else 0.0
        return 0.0 if expected == actual else 1.0

    def summary(self) -> Dict[str, Any]:
        return {
            "total_actions": len(self._history),
            "tracked_patterns": len(self._patterns),
            "recent_insights": len(self.reflect()),
            "mean_delta_10": float(np.mean([e["delta"] for e in self._history[-10:]])) if self._history else 0,
        }
