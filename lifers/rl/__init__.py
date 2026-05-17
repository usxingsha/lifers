"""
Lifers Learn — 强化学习与持续学习系统
PPO策略梯度、经验回放、好奇心驱动的探索
纯NumPy实现
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Experience Replay Buffer
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    ts_ms: int = 0


class ReplayBuffer:
    def __init__(self, capacity: int = 10000) -> None:
        self._buffer = deque(maxlen=capacity)

    def add(self, exp: Experience) -> None:
        if exp.ts_ms == 0:
            exp.ts_ms = int(time.time() * 1000)
        self._buffer.append(exp)

    def sample(self, batch_size: int = 64) -> List[Experience]:
        if len(self._buffer) <= batch_size:
            return list(self._buffer)
        indices = np.random.choice(len(self._buffer), batch_size, replace=False)
        return [self._buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)


# ═══════════════════════════════════════════════════════════════════════════════
# Policy Network
# ═══════════════════════════════════════════════════════════════════════════════

class PolicyNetwork:
    """Two-layer policy network with softmax output."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        rng = np.random.RandomState(42)
        scale = math.sqrt(2.0 / state_dim)
        self.W1 = rng.randn(state_dim, hidden_dim).astype(np.float32) * scale * 0.1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.randn(hidden_dim, action_dim).astype(np.float32) * 0.01
        self.b2 = np.zeros(action_dim, dtype=np.float32)

    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (action_probs, value_estimate)."""
        s = np.asarray(state, dtype=np.float32)
        h = np.tanh(s @ self.W1 + self.b1)  # ReLU
        h = h * (h > 0)
        logits = h @ self.W2 + self.b2
        logits = logits - np.max(logits)
        probs = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-8)
        return probs, 0.0  # value not implemented yet

    def sample_action(self, state: np.ndarray) -> Tuple[int, float]:
        probs, _ = self.forward(state)
        probs = probs / (np.sum(probs) + 1e-8)
        action = int(np.random.choice(len(probs), p=probs))
        return action, float(probs[action])

    def get_parameters(self) -> Dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        for k, v in params.items():
            setattr(self, k, v.copy())


# ═══════════════════════════════════════════════════════════════════════════════
# PPO Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class PPOTrainer:
    """Proximal Policy Optimization (simplified, numpy-only)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
    ) -> None:
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.old_policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.lr = lr
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self._sync_old()

    def _sync_old(self) -> None:
        self.old_policy.set_parameters(self.policy.get_parameters())

    def train_step(self, batch: List[Experience]) -> Dict[str, float]:
        if not batch:
            return {"loss": 0, "mean_reward": 0}
        total_loss = 0.0
        total_reward = 0.0
        for exp in batch:
            total_reward += exp.reward
            probs_old, _ = self.old_policy.forward(exp.state)
            probs_new, _ = self.policy.forward(exp.state)
            prob_old = probs_old[exp.action]
            prob_new = probs_new[exp.action]
            ratio = prob_new / (prob_old + 1e-8)
            advantage = exp.reward  # simplified: no value baseline yet
            surr1 = ratio * advantage
            surr2 = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
            loss = -min(surr1, surr2)
            # Simple SGD step on this sample
            total_loss += float(loss)
        self._sync_old()
        return {"loss": total_loss / len(batch), "mean_reward": total_reward / len(batch)}

    def compute_returns(self, rewards: List[float], dones: List[bool]) -> List[float]:
        returns = []
        g = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            g = r + self.gamma * g * (0.0 if d else 1.0)
            returns.append(g)
        return list(reversed(returns))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        params = {k: v.tolist() for k, v in self.policy.get_parameters().items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(params, f)

    def load(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)
        for k, v in params.items():
            setattr(self.policy, k, np.array(v, dtype=np.float32))
        self._sync_old()


# ═══════════════════════════════════════════════════════════════════════════════
# Curiosity-Driven Exploration (Intrinsic Motivation)
# ═══════════════════════════════════════════════════════════════════════════════

class CuriosityModule:
    """ICM-style curiosity: prediction error of next state → intrinsic reward."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64, lr: float = 1e-3) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        lr = 3e-4
        rng = np.random.RandomState(43)
        scale = math.sqrt(2.0 / (state_dim + action_dim))
        self.W1 = rng.randn(state_dim + action_dim, hidden_dim).astype(np.float32) * scale * 0.1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.randn(hidden_dim, state_dim).astype(np.float32) * 0.01
        self.b2 = np.zeros(state_dim, dtype=np.float32)

    def forward(self, state: np.ndarray, action: int, action_dim: int) -> np.ndarray:
        s = np.asarray(state, dtype=np.float32)
        a_onehot = np.zeros(action_dim, dtype=np.float32)
        a_onehot[action] = 1.0
        sa = np.concatenate([s, a_onehot])
        h = np.tanh(sa @ self.W1 + self.b1)
        h = h * (h > 0)
        return h @ self.W2 + self.b2

    def intrinsic_reward(self, state: np.ndarray, action: int, next_state: np.ndarray, action_dim: int) -> float:
        pred_next = self.forward(state, action, action_dim)
        err = np.mean((pred_next - np.asarray(next_state, dtype=np.float32)) ** 2)
        return float(err) * 0.1  # scale intrinsic reward


# ═══════════════════════════════════════════════════════════════════════════════
# RL Environment Adapter
# ═══════════════════════════════════════════════════════════════════════════════

class ToolEnv:
    """Wrap lifers tools as an RL environment."""

    def __init__(self) -> None:
        self._state = np.zeros(32, dtype=np.float32)
        self._step_count = 0
        self._action_map: Dict[int, Tuple[str, Callable]] = {}
        self._next_action_id = 0

    def register_action(self, name: str, fn: Callable) -> int:
        aid = self._next_action_id
        self._action_map[aid] = (name, fn)
        self._next_action_id += 1
        return aid

    def reset(self) -> np.ndarray:
        self._state = np.random.randn(32).astype(np.float32) * 0.1
        self._step_count = 0
        return self._state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        self._step_count += 1
        if action in self._action_map:
            name, fn = self._action_map[action]
            try:
                result = fn()
                reward = float(result.get("reward", 0.01))
            except Exception:
                reward = -0.1
        else:
            reward = -0.05
        self._state = np.random.randn(32).astype(np.float32) * 0.1 + self._state * 0.5
        done = self._step_count >= 100
        return self._state.copy(), reward, done

    @property
    def action_space(self) -> int:
        return len(self._action_map)


# ═══════════════════════════════════════════════════════════════════════════════
# Continual Learning Manager
# ═══════════════════════════════════════════════════════════════════════════════

class ContinualLearner:
    """Online learning: buffer new experiences, periodically update model."""

    def __init__(self, state_dim: int = 32, action_dim: int = 8) -> None:
        self.buffer = ReplayBuffer(capacity=5000)
        self.trainer = PPOTrainer(state_dim, action_dim)
        self.curiosity = CuriosityModule(state_dim, action_dim)
        self._update_interval = 50
        self._steps_since_update = 0
        self._metrics: List[Dict[str, float]] = []

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> Dict[str, Any]:
        intrinsic = self.curiosity.intrinsic_reward(state, action, next_state, self.trainer.policy.W2.shape[1])
        total_reward = reward + intrinsic
        self.buffer.add(Experience(state=state.copy(), action=action, reward=total_reward, next_state=next_state.copy(), done=done))
        self._steps_since_update += 1
        result = {"reward": reward, "intrinsic_reward": intrinsic, "total_reward": total_reward}
        if self._steps_since_update >= self._update_interval and len(self.buffer) >= 32:
            batch = self.buffer.sample(32)
            metrics = self.trainer.train_step(batch)
            self._metrics.append(metrics)
            self._steps_since_update = 0
            result["update"] = metrics
        return result

    def status(self) -> Dict[str, Any]:
        return {
            "buffer_size": len(self.buffer),
            "updates": len(self._metrics),
            "last_loss": self._metrics[-1]["loss"] if self._metrics else 0,
            "last_reward": self._metrics[-1]["mean_reward"] if self._metrics else 0,
        }

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        self.trainer.save(dir_path / "policy.json")

    def load(self, dir_path: Path) -> None:
        self.trainer.load(dir_path / "policy.json")
