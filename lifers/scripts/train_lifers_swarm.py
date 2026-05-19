"""
Lifers Swarm v2 — 多智能体Q-learning + 任务分配网络
品牌化权重: weights/lifers_swarm_policy.json
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as cpu_np
from lifers.core.compute_backend import get_compute_backend
np, _DEVICE, _GPU_INFO = get_compute_backend()

ROOT = Path(__file__).resolve().parent.parent.parent

N_AGENTS = 8
N_TASKS = 10
STATE_DIM = N_AGENTS + N_TASKS + 2  # agent states + task states + global


class SwarmEnv:
    """多智能体任务分配环境 — 奖励塑形版本"""

    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
        self.reset()

    def reset(self):
        self.task_difficulty = self.rng.uniform(0.2, 1.0, N_TASKS).astype(np.float32)
        self.task_urgency = self.rng.uniform(0.1, 1.0, N_TASKS).astype(np.float32)
        self.agent_skills = self.rng.uniform(0.3, 1.0, (N_AGENTS, N_TASKS)).astype(np.float32)
        self.agent_cooldown = np.zeros(N_AGENTS, dtype=np.float32)
        self.active_tasks = self.rng.choice(N_TASKS, size=min(5, N_TASKS), replace=False)
        self.tasks_completed = 0
        self.step_count = 0
        return self._state()

    def _state(self) -> np.ndarray:
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[:N_TASKS] = self.task_difficulty * self.task_urgency
        state[N_TASKS:N_TASKS + N_AGENTS] = (self.agent_cooldown > 0).astype(np.float32)
        state[-2] = self.step_count / 40.0
        state[-1] = len(self.active_tasks) / N_TASKS
        return state

    def step(self, agent_idx: int, task_idx: int) -> tuple:
        self.step_count += 1
        reward = 0.0

        # 冷却衰减
        self.agent_cooldown = np.maximum(0, self.agent_cooldown - 1)

        if task_idx not in self.active_tasks:
            reward -= 0.1  # 轻微惩罚（避免选择无效任务）
            return self._state(), reward, self.step_count >= 20

        if self.agent_cooldown[agent_idx] > 0:
            reward -= 0.2  # 智能体忙碌中
            return self._state(), reward, self.step_count >= 20

        skill = self.agent_skills[agent_idx, task_idx]
        difficulty = self.task_difficulty[task_idx]
        urgency = self.task_urgency[task_idx]
        success_prob = skill * (1.0 - difficulty * 0.4)

        reward += 0.05  # 有效任务选择微奖励

        if self.rng.random() < success_prob:
            reward += urgency * (0.5 + skill * 0.5)  # 技能越高奖励越多
            self.active_tasks = self.active_tasks[self.active_tasks != task_idx]
            self.tasks_completed += 1
        else:
            reward -= 0.05 * difficulty  # 轻微惩罚
            self.agent_cooldown[agent_idx] = 2  # 冷却2步

        # 完成所有任务
        if len(self.active_tasks) == 0:
            reward += 2.0 + 0.1 * self.tasks_completed
            n_new = min(3, N_TASKS)
            self.active_tasks = self.rng.choice(N_TASKS, size=n_new, replace=False)
            self.agent_cooldown[:] = 0  # 清除冷却

        done = self.step_count >= 40
        return self._state(), reward, done


class SwarmQNetwork:
    """Q网络: STATE_DIM→96→48→(N_AGENTS*N_TASKS)"""

    def __init__(self):
        rng = np.random.RandomState(42)
        self.W1 = rng.randn(STATE_DIM, 96).astype(np.float32) * np.sqrt(2.0 / STATE_DIM)
        self.b1 = np.zeros(96, dtype=np.float32)
        self.W2 = rng.randn(96, 48).astype(np.float32) * np.sqrt(2.0 / 96)
        self.b2 = np.zeros(48, dtype=np.float32)
        self.W3 = rng.randn(48, N_AGENTS * N_TASKS).astype(np.float32) * np.sqrt(2.0 / 48)
        self.b3 = np.zeros(N_AGENTS * N_TASKS, dtype=np.float32)

    def forward(self, state):
        s = np.asarray(state, dtype=np.float32)
        if s.ndim == 1:
            s = s.reshape(1, -1)
        h1 = np.maximum(0, s @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        return (h2 @ self.W3 + self.b3).reshape(-1, N_AGENTS, N_TASKS)

    def predict(self, state):
        q = self.forward(state)[0]
        return np.unravel_index(np.argmax(q), q.shape)


class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size, rng):
        indices = rng.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        states = np.array([self.buffer[i][0] for i in indices], dtype=np.float32)
        actions = np.array([self.buffer[i][1] for i in indices], dtype=np.int32)
        rewards = np.array([self.buffer[i][2] for i in indices], dtype=np.float32)
        next_states = np.array([self.buffer[i][3] for i in indices], dtype=np.float32)
        dones = np.array([self.buffer[i][4] for i in indices], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def train_swarm_policy(
    n_episodes: int = 600,
    save_path: Optional[Path] = None,
    verbose: bool = True,
) -> SwarmQNetwork:
    if save_path is None:
        save_path = ROOT / "weights" / "lifers_swarm_policy.json"

    model = SwarmQNetwork()
    target = SwarmQNetwork()
    tau = 0.005  # Polyak 软更新系数
    for attr in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        setattr(target, attr, getattr(model, attr).copy())

    env_rng = np.random.RandomState(42)
    train_rng = np.random.RandomState(123)
    env = SwarmEnv(env_rng)
    replay = ReplayBuffer(capacity=20000)

    epsilon = 0.6
    gamma = 0.95
    lr = 3e-4
    batch_size = 64
    best_avg = -float("inf")
    ep_rewards = []
    step_count = 0

    for ep in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            q_values = model.forward(state)[0]
            if env_rng.random() < epsilon:
                agent_idx = env_rng.randint(0, N_AGENTS)
                task_idx = env_rng.randint(0, N_TASKS)
            else:
                agent_idx, task_idx = np.unravel_index(np.argmax(q_values), q_values.shape)

            action = agent_idx * N_TASKS + task_idx
            next_state, reward, done = env.step(agent_idx, task_idx)
            total_reward += reward
            replay.push(state.copy(), action, reward, next_state.copy(), float(done))
            state = next_state
            step_count += 1

            if len(replay) >= batch_size * 4:
                states, actions, rewards, next_states, dones = replay.sample(batch_size, train_rng)
                B = len(states)

                next_q = target.forward(next_states).reshape(B, -1)
                max_next_q = np.max(next_q, axis=1)
                y_target = rewards + gamma * max_next_q * (1 - dones)

                h1_pre = states @ model.W1 + model.b1
                h1 = np.maximum(0, h1_pre)
                h2_pre = h1 @ model.W2 + model.b2
                h2 = np.maximum(0, h2_pre)
                Q = h2 @ model.W3 + model.b3

                td_errors = Q[np.arange(B), actions] - y_target

                dQ = np.zeros_like(Q)
                dQ[np.arange(B), actions] = 2 * td_errors / B
                dQ = np.clip(dQ, -2.0, 2.0)

                gW3 = h2.T @ dQ
                gb3 = np.sum(dQ, axis=0)
                dh2 = dQ @ model.W3.T
                dh2[h2_pre <= 0] = 0
                gW2 = h1.T @ dh2
                gb2 = np.sum(dh2, axis=0)
                dh1 = dh2 @ model.W2.T
                dh1[h1_pre <= 0] = 0
                gW1 = states.T @ dh1
                gb1 = np.sum(dh1, axis=0)

                # 梯度裁剪
                for g in [gW1, gW2, gW3]:
                    gn = np.linalg.norm(g)
                    if gn > 5.0:
                        g *= 5.0 / gn

                model.W3 -= lr * gW3; model.b3 -= lr * gb3
                model.W2 -= lr * gW2; model.b2 -= lr * gb2
                model.W1 -= lr * gW1; model.b1 -= lr * gb1

                # Polyak 软更新 target 网络
                for attr in ["W1", "b1", "W2", "b2", "W3", "b3"]:
                    mv = getattr(model, attr)
                    tv = getattr(target, attr)
                    setattr(target, attr, tau * mv + (1 - tau) * tv)

        ep_rewards.append(total_reward)
        epsilon = max(0.05, epsilon * 0.998)

        if len(ep_rewards) >= 20:
            avg20 = np.mean(ep_rewards[-20:])
            if avg20 > best_avg:
                best_avg = avg20
                _save_swarm_model(model, save_path)

        if (ep + 1) % 60 == 0 and verbose:
            avg20 = np.mean(ep_rewards[-20:]) if len(ep_rewards) >= 20 else np.mean(ep_rewards)
            print(f"[Lifers-Swarm v3] ep {ep + 1}/{n_episodes}  "
                  f"avg20={avg20:.3f}  ε={epsilon:.3f}")

    _save_swarm_model(model, save_path)
    if verbose:
        final_avg = np.mean(ep_rewards[-20:]) if len(ep_rewards) >= 20 else np.mean(ep_rewards)
        print(f"[Lifers-Swarm v3] 完成 best_avg20={best_avg:.3f}  final_avg20={final_avg:.3f} → {save_path}")
    return model


def _save_swarm_model(model: SwarmQNetwork, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "brand": "Lifers Swarm DQN v3",
        "version": 3,
        "n_agents": N_AGENTS,
        "n_tasks": N_TASKS,
        "state_dim": STATE_DIM,
        "W1": model.W1.tolist(), "b1": model.b1.tolist(),
        "W2": model.W2.tolist(), "b2": model.b2.tolist(),
        "W3": model.W3.tolist(), "b3": model.b3.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    episodes = int(os.environ.get("LIFERS_SWARM_EPISODES", "600"))
    out = ROOT / "weights" / "lifers_swarm_policy.json"
    print("[Lifers-Swarm v3] DQN 多智能体任务分配 v2.1")
    t0 = time.time()
    train_swarm_policy(n_episodes=episodes, save_path=out, verbose=True)
    print(f"[Lifers-Swarm v3] 耗时={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
