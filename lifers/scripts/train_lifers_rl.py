"""
Lifers RL v3 — Double DQN + 经验回放 + 好奇心驱动探索 + 多环境训练
品牌化权重: weights/lifers_rl_policy.json
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as cpu_np
from lifers.core.compute_backend import get_compute_backend
np, _DEVICE, _GPU_INFO = get_compute_backend()

ROOT = Path(__file__).resolve().parent.parent.parent


# ═══════════════════════════════════════════════════════════════════════════════
# 环境
# ═══════════════════════════════════════════════════════════════════════════════

class LifersGridWorld:
    """网格世界 — 导航+收集+避障"""

    def __init__(self, size: int = 8, n_objects: int = 3, n_hazards: int = 2):
        self.size = size
        self.n_objects = n_objects
        self.n_hazards = n_hazards
        self._rng = np.random.RandomState()
        self.reset()

    def reset(self) -> np.ndarray:
        self.agent_pos = np.array([0, 0], dtype=np.float32)
        self.objects = self._rng.randint(1, self.size - 1, (self.n_objects, 2)).astype(np.float32)
        self.hazards = self._rng.randint(1, self.size - 1, (self.n_hazards, 2)).astype(np.float32)
        self.collected = 0
        self.steps = 0
        self.max_steps = self.size * 4
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        state = np.concatenate([
            self.agent_pos / self.size,
            self.objects.flatten() / self.size,
            self.hazards.flatten() / self.size,
            np.array([self.collected / self.n_objects, self.steps / self.max_steps], dtype=np.float32),
        ]).astype(np.float32)
        padded = np.zeros(32, dtype=np.float32)
        padded[:len(state)] = state
        return padded

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        moves = {
            0: np.array([0, -1]), 1: np.array([0, 1]),
            2: np.array([-1, 0]), 3: np.array([1, 0]),
            4: np.array([0, 0]), 5: np.array([0, 0]),
            6: np.array([0, -2]), 7: np.array([0, 2]),
        }
        self.steps += 1
        reward = -0.01
        done = False

        delta = moves.get(action, np.array([0, 0]))
        self.agent_pos = np.clip(self.agent_pos + delta, 0, self.size - 1)

        if action == 4:
            for i, obj in enumerate(self.objects):
                if np.array_equal(self.agent_pos.astype(int), obj.astype(int)):
                    reward += 1.0
                    self.collected += 1
                    self.objects[i] = np.array([-99, -99], dtype=np.float32)
                    if self.collected >= self.n_objects:
                        reward += 2.0
                        done = True
                    break

        for haz in self.hazards:
            if np.linalg.norm(self.agent_pos - haz) < 1.5:
                reward -= 0.5

        if action in (6, 7):
            reward -= 0.05

        if self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done


class LifersToolEnv:
    """工具选择环境"""

    def __init__(self, n_tools: int = 8):
        self.n_tools = n_tools
        self._rng = np.random.RandomState()
        self.reset()

    def reset(self) -> np.ndarray:
        self.task_type = self._rng.randint(0, 4)
        self.task_difficulty = self._rng.rand()
        self.tools_used = 0
        self.max_tools = 20
        self._tool_effectiveness = self._rng.rand(self.n_tools).astype(np.float32)
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        state = np.zeros(32, dtype=np.float32)
        state[self.task_type] = 1.0
        state[4] = self.task_difficulty
        state[5] = self.tools_used / self.max_tools
        state[6:6+self.n_tools] = self._tool_effectiveness
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if action >= self.n_tools:
            action = self.n_tools - 1
        self.tools_used += 1
        done = self.tools_used >= self.max_tools

        effectiveness = self._tool_effectiveness[action]
        task_match = 1.0 - abs(action / self.n_tools - self.task_type / 4.0)
        reward = effectiveness * 0.2 + task_match * 0.8

        return self._get_state(), float(reward), done


# ═══════════════════════════════════════════════════════════════════════════════
# Double DQN 网络
# ═══════════════════════════════════════════════════════════════════════════════

class RLQNetwork:
    """3层 Q-network: 32→128→64→action"""

    def __init__(self, state_dim=32, action_dim=8, hidden1=128, hidden2=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        rng = np.random.RandomState(42)
        self.W1 = rng.randn(state_dim, hidden1).astype(np.float32) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(hidden1, dtype=np.float32)
        self.W2 = rng.randn(hidden1, hidden2).astype(np.float32) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2, dtype=np.float32)
        self.W3 = rng.randn(hidden2, action_dim).astype(np.float32) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros(action_dim, dtype=np.float32)

    def forward(self, state):
        s = np.asarray(state, dtype=np.float32)
        if s.ndim == 1:
            s = s.reshape(1, -1)
        h1 = np.maximum(0, s @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        q = h2 @ self.W3 + self.b3
        if q.shape[0] == 1:
            return q[0]
        return q

    def predict(self, state):
        q = self.forward(state)
        return int(np.argmax(q)), q


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size, rng):
        indices = rng.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        batch = [self.buffer[i] for i in indices]
        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int32)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════════════════
# 好奇心模块 (Intrinsic Curiosity Module)
# ═══════════════════════════════════════════════════════════════════════════════

class CuriosityModule:
    """预测下一状态的编码器 — 预测误差作为内在奖励"""

    def __init__(self, state_dim=32, hidden=64):
        rng = np.random.RandomState(99)
        self.W1 = rng.randn(state_dim + 8, hidden).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.randn(hidden, state_dim).astype(np.float32) * 0.1
        self.b2 = np.zeros(state_dim, dtype=np.float32)

    def predict_next_state(self, state, action_onehot):
        x = np.concatenate([state, action_onehot]).reshape(1, -1).astype(np.float32)
        h = np.maximum(0, x @ self.W1 + self.b1)
        return (h @ self.W2 + self.b2)[0]

    def intrinsic_reward(self, state, action, next_state):
        action_onehot = np.zeros(8, dtype=np.float32)
        action_onehot[action] = 1.0
        pred = self.predict_next_state(state, action_onehot)
        error = np.mean((pred - next_state) ** 2)
        return float(error * 0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# RL Trainer v3
# ═══════════════════════════════════════════════════════════════════════════════

def train_lifers_rl(
    total_episodes: int = 500,
    save_path: Path | None = None,
    verbose: bool = True,
) -> RLQNetwork:
    """Double DQN + 经验回放 + 好奇心 + 多环境训练"""
    if save_path is None:
        save_path = ROOT / "weights" / "lifers_rl_policy.json"

    model = RLQNetwork()
    target = RLQNetwork()
    # 复制权重到目标网络
    target.W1 = model.W1.copy(); target.b1 = model.b1.copy()
    target.W2 = model.W2.copy(); target.b2 = model.b2.copy()
    target.W3 = model.W3.copy(); target.b3 = model.b3.copy()

    curiosity = CuriosityModule()
    replay = ReplayBuffer(capacity=20000)
    env_rng = np.random.RandomState(42)
    train_rng = np.random.RandomState(123)

    envs = [LifersToolEnv(), LifersGridWorld()]
    lr = 1e-3
    gamma = 0.95
    batch_size = 64
    epsilon = 0.5
    best_avg = -float("inf")
    ep_rewards = []

    for ep in range(total_episodes):
        env = envs[ep % len(envs)]
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            q_values = model.forward(state)
            if env_rng.random_sample() < epsilon:
                action = env_rng.randint(0, 8)
            else:
                action = int(np.argmax(q_values))

            next_state, ext_reward, done = env.step(action)
            # 好奇心内在奖励
            int_reward = curiosity.intrinsic_reward(state, action, next_state)
            reward = ext_reward + int_reward * 0.1

            total_reward += reward
            replay.push(state.copy(), action, reward, next_state.copy(), float(done))
            state = next_state

            if len(replay) >= batch_size:
                states, actions, rewards, next_states, dones = replay.sample(batch_size, train_rng)

                # Double DQN: 用 online 选动作，用 target 算 Q 值
                next_q_online = model.forward(next_states)  # (batch, n_actions)
                best_actions = np.argmax(next_q_online, axis=1)
                next_q_target = target.forward(next_states)  # (batch, n_actions)
                max_next_q = next_q_target[np.arange(batch_size), best_actions]
                targets = rewards + gamma * max_next_q * (1 - dones)

                # 前向 + 反向传播
                s = states
                h1_pre = s @ model.W1 + model.b1
                h1 = np.maximum(0, h1_pre)
                h2_pre = h1 @ model.W2 + model.b2
                h2 = np.maximum(0, h2_pre)
                Q = h2 @ model.W3 + model.b3

                td_errors = targets - Q[np.arange(batch_size), actions]
                Q[np.arange(batch_size), actions] += lr * td_errors

                dQ = np.zeros_like(Q)
                dQ[np.arange(batch_size), actions] = -2 * td_errors / batch_size
                dQ = np.clip(dQ, -1.0, 1.0)

                # 梯度
                gW3 = h2.T @ dQ
                gb3 = np.sum(dQ, axis=0)
                dh2 = dQ @ model.W3.T
                dh2[h2_pre <= 0] = 0
                gW2 = h1.T @ dh2
                gb2 = np.sum(dh2, axis=0)
                dh1 = dh2 @ model.W2.T
                dh1[h1_pre <= 0] = 0
                gW1 = s.T @ dh1
                gb1 = np.sum(dh1, axis=0)

                model.W3 -= lr * gW3; model.b3 -= lr * gb3
                model.W2 -= lr * gW2; model.b2 -= lr * gb2
                model.W1 -= lr * gW1; model.b1 -= lr * gb1

        ep_rewards.append(total_reward)
        epsilon = max(0.05, epsilon * 0.992)

        # 每50回合更新目标网络
        if ep % 50 == 0:
            target.W1 = model.W1.copy(); target.b1 = model.b1.copy()
            target.W2 = model.W2.copy(); target.b2 = model.b2.copy()
            target.W3 = model.W3.copy(); target.b3 = model.b3.copy()

        if len(ep_rewards) >= 50:
            avg50 = np.mean(ep_rewards[-50:])
            if avg50 > best_avg:
                best_avg = avg50
                _save_rl_model(model, save_path)

        if (ep + 1) % 100 == 0 and verbose:
            avg50 = np.mean(ep_rewards[-50:]) if len(ep_rewards) >= 50 else np.mean(ep_rewards)
            avg20 = np.mean(ep_rewards[-20:]) if len(ep_rewards) >= 20 else avg50
            env_name = type(envs[ep % len(envs)]).__name__
            print(f"[Lifers-RL v3] ep {ep + 1}/{total_episodes}  "
                  f"avg50={avg50:.3f}  avg20={avg20:.3f}  ε={epsilon:.3f}  env={env_name}")

    _save_rl_model(model, save_path)
    if verbose:
        final_avg = np.mean(ep_rewards[-50:]) if len(ep_rewards) >= 50 else np.mean(ep_rewards)
        print(f"[Lifers-RL v3] 训练完成 best_avg50={best_avg:.3f}  final_avg50={final_avg:.3f} → {save_path}")
    return model


def _save_rl_model(model: RLQNetwork, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "brand": "Lifers RL Double DQN v3",
        "version": 3,
        "state_dim": model.state_dim,
        "action_dim": model.action_dim,
        "hidden1": model.hidden1,
        "hidden2": model.hidden2,
        "W1": model.W1.tolist(), "b1": model.b1.tolist(),
        "W2": model.W2.tolist(), "b2": model.b2.tolist(),
        "W3": model.W3.tolist(), "b3": model.b3.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    episodes = int(os.environ.get("LIFERS_RL_EPISODES", "500"))
    out = ROOT / "weights" / "lifers_rl_policy.json"
    print(f"[Lifers-RL v3] Double DQN + Curiosity 训练 episodes={episodes}")
    t0 = time.time()
    train_lifers_rl(total_episodes=episodes, save_path=out, verbose=True)
    print(f"[Lifers-RL v3] 完成 耗时={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
