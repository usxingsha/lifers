"""
Lifers RL 训练 — PPO策略网络 + 好奇心驱动探索
品牌化权重: weights/lifers_rl_policy.json
纯numpy实现，无需外部依赖
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent


# ═══════════════════════════════════════════════════════════════════════════════
# Lifers 仿真训练环境
# ═══════════════════════════════════════════════════════════════════════════════

class LifersGridWorld:
    """Lifers 网格世界 — 智能体导航+收集+避障训练环境"""

    def __init__(self, size: int = 8, n_objects: int = 3, n_hazards: int = 2):
        self.size = size
        self.n_objects = n_objects
        self.n_hazards = n_hazards
        self._rng = np.random.RandomState()
        self.reset()

    def reset(self) -> np.ndarray:
        """重置环境，返回初始状态"""
        self.agent_pos = np.array([0, 0], dtype=np.float32)
        self.objects = self._rng.randint(1, self.size - 1, (self.n_objects, 2)).astype(np.float32)
        self.hazards = self._rng.randint(1, self.size - 1, (self.n_hazards, 2)).astype(np.float32)
        self.collected = 0
        self.steps = 0
        self.max_steps = self.size * 4
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """构建状态向量: [agent_x, agent_y, obj1_x, obj1_y, ..., haz1_x, haz1_y, ..., collected, steps]"""
        state = np.concatenate([
            self.agent_pos / self.size,
            self.objects.flatten() / self.size,
            self.hazards.flatten() / self.size,
            [self.collected / self.n_objects, self.steps / self.max_steps],
        ]).astype(np.float32)
        # 填充到固定维度
        padded = np.zeros(32, dtype=np.float32)
        padded[:len(state)] = state
        return padded

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """执行动作: 0=上 1=下 2=左 3=右 4=收集 5=停留 6=加速上 7=加速下"""
        moves = {
            0: np.array([0, -1]), 1: np.array([0, 1]),
            2: np.array([-1, 0]), 3: np.array([1, 0]),
            4: np.array([0, 0]), 5: np.array([0, 0]),
            6: np.array([0, -2]), 7: np.array([0, 2]),
        }
        self.steps += 1
        reward = -0.01  # 时间惩罚
        done = False

        # 移动
        delta = moves.get(action, np.array([0, 0]))
        new_pos = self.agent_pos + delta
        new_pos = np.clip(new_pos, 0, self.size - 1)
        self.agent_pos = new_pos

        # 收集物品
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

        # 碰撞危险
        for haz in self.hazards:
            if np.linalg.norm(self.agent_pos - haz) < 1.5:
                reward -= 0.5

        # 加速动作额外代价
        if action in (6, 7):
            reward -= 0.05

        # 步数限制
        if self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done


class LifersToolEnv:
    """Lifers 工具选择环境 — 学习根据任务类型选择最优工具"""

    def __init__(self, n_tools: int = 8):
        self.n_tools = n_tools
        self._rng = np.random.RandomState()
        self.reset()

    def reset(self) -> np.ndarray:
        self.task_type = self._rng.randint(0, 4)
        self.task_difficulty = self._rng.rand()
        self.tools_used = 0
        self.max_tools = 20
        # 每回合随机化工具效能，使不同任务类型的最优工具不同
        self._tool_effectiveness = self._rng.rand(self.n_tools).astype(np.float32)
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        # one-hot task_type (4) + difficulty (1) + progress (1) = 6 meaningful dims
        state = np.zeros(32, dtype=np.float32)
        state[self.task_type] = 1.0  # one-hot task type
        state[4] = self.task_difficulty
        state[5] = self.tools_used / self.max_tools
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if action >= self.n_tools:
            action = self.n_tools - 1
        self.tools_used += 1
        done = self.tools_used >= self.max_tools

        effectiveness = self._tool_effectiveness[action]
        task_match = 1.0 - abs(action / self.n_tools - self.task_type / 4.0)
        # task_match权重80%: 最优动作主要由task_type决定
        reward = effectiveness * 0.2 + task_match * 0.8

        return self._get_state(), float(reward), done


# ═══════════════════════════════════════════════════════════════════════════════
# Lifers RL Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class LifersRLTrainer:
    """Lifers Q-learning 训练器 — ε-greedy 探索 + TD学习"""

    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 8,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon: float = 0.3,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self._init_network()
        self._metrics: Dict[str, List[float]] = {
            "loss": [], "reward": [], "epsilon": [],
        }

    def _init_network(self):
        rng = np.random.RandomState(42)
        scale_w1 = math.sqrt(2.0 / self.state_dim)
        self.W1 = rng.randn(self.state_dim, self.hidden_dim).astype(np.float32) * scale_w1
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        scale_w2 = math.sqrt(2.0 / self.hidden_dim)
        self.W2 = rng.randn(self.hidden_dim, self.action_dim).astype(np.float32) * scale_w2
        self.b2 = np.zeros(self.action_dim, dtype=np.float32)

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Q值推理"""
        s = np.asarray(state, dtype=np.float32).reshape(1, -1)
        h = np.maximum(0, s @ self.W1 + self.b1)
        q_values = (h @ self.W2 + self.b2)[0]
        return q_values

    def sample_action(self, state: np.ndarray) -> Tuple[int, np.ndarray]:
        """ε-greedy 动作选择"""
        q_values = self.forward(state)
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.action_dim)
        else:
            action = int(np.argmax(q_values))
        return action, q_values

    def train_step(self, state, action, reward, next_state, done) -> float:
        """Q-learning TD更新"""
        q_values = self.forward(state)
        q_old = q_values[action]

        if done:
            q_target = reward
        else:
            next_q = self.forward(next_state)
            q_target = reward + self.gamma * np.max(next_q)

        td_error = q_target - q_old
        loss = td_error ** 2

        # 仅更新选中动作的Q值 → 梯度只流向 W2[:, action] 和 b2[action]
        s = np.asarray(state, dtype=np.float32).reshape(1, -1)
        h_pre = s @ self.W1 + self.b1
        h = np.maximum(0, h_pre)

        # dQ/daction: 增大选中动作的Q值方向
        grad = self.lr * td_error
        self.W2[:, action] += (h[0] * grad).astype(np.float32)
        self.b2[action] += grad

        # W1 梯度: 通过 ReLU 反向传播
        dh = grad * self.W2[:, action]
        dh[h_pre[0] <= 0] = 0
        self.W1 += self.lr * 0.1 * (s.T @ dh.reshape(1, -1))
        self.b1 += self.lr * 0.1 * dh

        return float(loss)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "brand": "Lifers RL Q-Network",
            "version": 2,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "LifersRLTrainer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        trainer = cls(
            state_dim=data["state_dim"],
            action_dim=data["action_dim"],
            hidden_dim=data["hidden_dim"],
        )
        trainer.W1 = np.array(data["W1"], dtype=np.float32)
        trainer.b1 = np.array(data["b1"], dtype=np.float32)
        trainer.W2 = np.array(data["W2"], dtype=np.float32)
        trainer.b2 = np.array(data["b2"], dtype=np.float32)
        return trainer


# ═══════════════════════════════════════════════════════════════════════════════
# 训练主循环
# ═══════════════════════════════════════════════════════════════════════════════

def train_lifers_rl(
    total_episodes: int = 1000,
    save_path: Path | None = None,
    verbose: bool = True,
) -> LifersRLTrainer:
    """Lifers RL 品牌化训练主循环"""
    if save_path is None:
        save_path = ROOT / "weights" / "lifers_rl_policy.json"

    trainer = LifersRLTrainer(epsilon=0.3)
    env = LifersToolEnv()
    env_name = "ToolEnv"

    best_reward = -float("inf")
    episode_rewards: List[float] = []
    total_steps = 0

    for ep in range(total_episodes):
        state = env.reset()
        ep_reward = 0.0
        ep_loss = 0.0
        done = False
        steps = 0

        while not done:
            action, q_values = trainer.sample_action(state)
            next_state, reward, done = env.step(action)
            loss = trainer.train_step(state, action, reward, next_state, done)
            ep_reward += reward
            ep_loss += loss
            state = next_state
            steps += 1
            total_steps += 1

        episode_rewards.append(ep_reward)

        # 衰减探索率
        trainer.epsilon = max(0.05, 0.3 * (0.995 ** total_steps))

        if (ep + 1) % 100 == 0:
            avg_reward = float(np.mean(episode_rewards[-100:]))
            if verbose:
                print(f"[Lifers-RL] ep {ep + 1}/{total_episodes}  "
                      f"env={env_name}  avg_reward={avg_reward:.3f}  "
                      f"loss={ep_loss/max(steps,1):.4f}  ε={trainer.epsilon:.3f}")

            if avg_reward > best_reward:
                best_reward = avg_reward
                trainer.save(save_path)
                if verbose:
                    print(f"[Lifers-RL] best model saved → {save_path}")

    trainer.save(save_path)
    if verbose:
        print(f"[Lifers-RL] training complete, weights → {save_path}")
    return trainer


def main():
    episodes = int(os.environ.get("LIFERS_RL_EPISODES", "1000"))
    out = ROOT / "weights" / "lifers_rl_policy.json"

    print(f"[Lifers-RL] 开始品牌化RL训练")
    print(f"[Lifers-RL] episodes={episodes}  output={out}")
    t0 = time.time()

    train_lifers_rl(total_episodes=episodes, save_path=out, verbose=True)

    elapsed = time.time() - t0
    print(f"[Lifers-RL] 完成 耗时={elapsed:.1f}s")


if __name__ == "__main__":
    main()
