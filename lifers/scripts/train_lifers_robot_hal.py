"""
Lifers Robot HAL v2 — 具身机器人行动策略 (DQN + 经验回放 + 奖励塑形)
品牌化权重: weights/lifers_robot_hal_policy.json
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent

ROBOT_ACTIONS = ["forward", "backward", "turn_left", "turn_right", "stop", "grasp", "release", "idle"]
N_ACTIONS = len(ROBOT_ACTIONS)


class RobotEnv:
    """2D连续空间物理环境"""

    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
        self.arena_size = 10.0
        self.reset()

    def reset(self):
        self.robot_x = self.rng.uniform(1, self.arena_size - 1)
        self.robot_y = self.rng.uniform(1, self.arena_size - 1)
        self.robot_theta = self.rng.uniform(0, 2 * math.pi)
        self.goal_x = self.rng.uniform(1, self.arena_size - 1)
        self.goal_y = self.rng.uniform(1, self.arena_size - 1)
        self.obstacles = [(self.rng.uniform(2, 8), self.rng.uniform(2, 8), 0.5) for _ in range(3)]
        self.step_count = 0
        self.grasped = False
        self._prev_goal_dist = math.sqrt((self.robot_x - self.goal_x)**2 + (self.robot_y - self.goal_y)**2)
        return self._state()

    def _state(self) -> np.ndarray:
        goal_dx = self.goal_x - self.robot_x
        goal_dy = self.goal_y - self.robot_y
        goal_dist = math.sqrt(goal_dx**2 + goal_dy**2)
        min_obs_dist = self.arena_size
        min_obs_dx = min_obs_dy = 0.0
        for ox, oy, r in self.obstacles:
            dx = ox - self.robot_x
            dy = oy - self.robot_y
            d = math.sqrt(dx**2 + dy**2) - r
            if d < min_obs_dist:
                min_obs_dist = d
                min_obs_dx = dx
                min_obs_dy = dy
        return np.array([
            self.robot_x / self.arena_size, self.robot_y / self.arena_size,
            math.cos(self.robot_theta), math.sin(self.robot_theta),
            goal_dx / self.arena_size, goal_dy / self.arena_size,
            min(goal_dist / self.arena_size, 1.0),
            min_obs_dx / self.arena_size, min_obs_dy / self.arena_size,
            max(min(min_obs_dist / self.arena_size, 1.0), -1.0),
            float(self.grasped),
        ], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        self.step_count += 1
        speed = 0.3
        reward = 0.0
        prev_dist = self._prev_goal_dist

        if action == 0:
            self.robot_x += speed * math.cos(self.robot_theta)
            self.robot_y += speed * math.sin(self.robot_theta)
        elif action == 1:
            self.robot_x -= speed * math.cos(self.robot_theta)
            self.robot_y -= speed * math.sin(self.robot_theta)
        elif action == 2:
            self.robot_theta += 0.4
        elif action == 3:
            self.robot_theta -= 0.4
        elif action == 5 and self._near_goal():
            self.grasped = True
            reward += 3.0
        elif action == 6:
            self.grasped = False

        # 边界软约束
        self.robot_x = max(0.2, min(self.arena_size - 0.2, self.robot_x))
        self.robot_y = max(0.2, min(self.arena_size - 0.2, self.robot_y))

        # 碰撞检测 (只有真正碰撞才惩罚)
        collided = False
        for ox, oy, r in self.obstacles:
            if math.sqrt((self.robot_x - ox)**2 + (self.robot_y - oy)**2) < r:
                collided = True
                break

        curr_dist = math.sqrt((self.robot_x - self.goal_x)**2 + (self.robot_y - self.goal_y)**2)
        self._prev_goal_dist = curr_dist

        # 奖励塑形: 接近目标给正奖励
        reward += (prev_dist - curr_dist) * 2.0  # 距离改善奖励
        if collided:
            reward -= 2.0
        if curr_dist < 1.0:
            reward += 0.5  # 接近目标
        if self._near_goal() and self.grasped:
            reward += 10.0
        reward -= 0.05  # 时间惩罚

        done = self.step_count >= 60 or (self._near_goal() and self.grasped)
        return self._state(), reward, done

    def _near_goal(self) -> bool:
        return math.sqrt((self.robot_x - self.goal_x)**2 + (self.robot_y - self.goal_y)**2) < 0.5


class LifersRobotHALPolicy:
    """DQN — 2层Q-network"""

    def __init__(self, state_dim=11, hidden_dim=96, n_actions=N_ACTIONS):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        rng = np.random.RandomState(42)
        self.W1 = rng.randn(state_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.randn(hidden_dim, n_actions).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(n_actions, dtype=np.float32)

    def forward(self, state):
        h = np.maximum(0, state @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

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


def train_robot_hal(
    n_episodes=800, lr=3e-3, gamma=0.95,
    save_path: Optional[Path] = None, verbose=True,
) -> LifersRobotHALPolicy:
    if save_path is None:
        save_path = ROOT / "weights" / "lifers_robot_hal_policy.json"

    model = LifersRobotHALPolicy()
    target_model = LifersRobotHALPolicy()
    target_model.W1 = model.W1.copy()
    target_model.b1 = model.b1.copy()
    target_model.W2 = model.W2.copy()
    target_model.b2 = model.b2.copy()

    env_rng = np.random.RandomState(42)
    train_rng = np.random.RandomState(123)
    env = RobotEnv(env_rng)
    replay = ReplayBuffer(capacity=20000)

    epsilon = 0.5
    best_avg = -float("inf")
    ep_rewards = []
    batch_size = 64

    for ep in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            q_values = model.forward(state)
            if env_rng.random() < epsilon:
                action = env_rng.randint(0, N_ACTIONS)
            else:
                action = int(np.argmax(q_values))

            next_state, reward, done = env.step(action)
            total_reward += reward
            replay.push(state.copy(), action, reward, next_state.copy(), float(done))
            state = next_state

            # 经验回放训练
            if len(replay) >= batch_size:
                states, actions, rewards, next_states, dones = replay.sample(batch_size, train_rng)

                # 目标网络
                next_q = target_model.forward(next_states).T  # (n_actions, batch)
                max_next_q = np.max(next_q, axis=0)  # (batch,)
                targets = rewards + gamma * max_next_q * (1 - dones)

                # 当前Q值
                H_pre = states @ model.W1 + model.b1
                H = np.maximum(0, H_pre)
                Q = H @ model.W2 + model.b2  # (batch, n_actions)

                # Q-learning梯度 (只更新被选动作)
                td_errors = targets - Q[np.arange(batch_size), actions]
                Q[np.arange(batch_size), actions] += lr * td_errors  # 目标Q值

                # 均方误差梯度
                dQ = np.zeros_like(Q)
                dQ[np.arange(batch_size), actions] = -2 * td_errors / batch_size
                dQ = np.clip(dQ, -1.0, 1.0)

                # 反向传播
                gW2 = H.T @ dQ
                gb2 = np.sum(dQ, axis=0)
                dH = dQ @ model.W2.T
                dH[H_pre <= 0] = 0
                gW1 = states.T @ dH
                gb1 = np.sum(dH, axis=0)

                model.W2 -= lr * gW2
                model.b2 -= lr * gb2
                model.W1 -= lr * gW1
                model.b1 -= lr * gb1

        ep_rewards.append(total_reward)
        epsilon = max(0.05, epsilon * 0.993)

        # 更新目标网络
        if ep % 50 == 0:
            target_model.W1 = model.W1.copy()
            target_model.b1 = model.b1.copy()
            target_model.W2 = model.W2.copy()
            target_model.b2 = model.b2.copy()

        if len(ep_rewards) >= 50:
            avg50 = np.mean(ep_rewards[-50:])
            if avg50 > best_avg:
                best_avg = avg50
                _save_robot_hal_model(model, save_path)

        if (ep + 1) % 100 == 0 and verbose:
            avg50 = np.mean(ep_rewards[-50:]) if len(ep_rewards) >= 50 else np.mean(ep_rewards)
            avg20 = np.mean(ep_rewards[-20:]) if len(ep_rewards) >= 20 else avg50
            print(f"[Lifers-RobotHAL v2] ep {ep + 1}/{n_episodes}  "
                  f"avg50={avg50:.2f}  avg20={avg20:.2f}  epsilon={epsilon:.3f}")

    _save_robot_hal_model(model, save_path)
    if verbose:
        final_avg = np.mean(ep_rewards[-50:]) if len(ep_rewards) >= 50 else np.mean(ep_rewards)
        print(f"[Lifers-RobotHAL v2] 训练完成 best_avg50={best_avg:.2f}  final_avg50={final_avg:.2f} -> {save_path}")
    return model


def _save_robot_hal_model(model: LifersRobotHALPolicy, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "brand": "Lifers Robot HAL Policy v2",
        "version": 2,
        "state_dim": model.state_dim,
        "hidden_dim": model.hidden_dim,
        "n_actions": model.n_actions,
        "actions": ROBOT_ACTIONS,
        "W1": model.W1.tolist(),
        "b1": model.b1.tolist(),
        "W2": model.W2.tolist(),
        "b2": model.b2.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    episodes = int(os.environ.get("LIFERS_ROBOTHAL_EPISODES", "800"))
    out = ROOT / "weights" / "lifers_robot_hal_policy.json"
    print(f"[Lifers-RobotHAL v2] 品牌化机器人DQN训练 episodes={episodes}")
    t0 = time.time()
    train_robot_hal(n_episodes=episodes, save_path=out, verbose=True)
    print(f"[Lifers-RobotHAL v2] 完成 耗时={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
