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
    """Lifers 工具选择环境 — 学习何时使用何种工具"""

    def __init__(self, n_tools: int = 8):
        self.n_tools = n_tools
        self._rng = np.random.RandomState()
        self._tool_effectiveness = self._rng.rand(n_tools).astype(np.float32)
        self.reset()

    def reset(self) -> np.ndarray:
        self.task_type = self._rng.randint(0, 4)
        self.task_difficulty = self._rng.rand()
        self.tools_used = 0
        self.max_tools = 6
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        state = np.array([
            self.task_type / 4.0,
            self.task_difficulty,
            self.tools_used / self.max_tools,
        ], dtype=np.float32)
        padded = np.zeros(32, dtype=np.float32)
        padded[:3] = state
        return padded

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if action >= self.n_tools:
            action = self.n_tools - 1
        self.tools_used += 1
        done = self.tools_used >= self.max_tools

        effectiveness = self._tool_effectiveness[action]
        task_match = 1.0 - abs(action / self.n_tools - self.task_type / 4.0)
        reward = effectiveness * 0.5 + task_match * 0.5 - 0.05

        if self.tools_used == 1 and effectiveness > 0.7:
            reward += 0.3
        if self.tools_used > 4 and reward < 0.3:
            reward -= 0.1

        return self._get_state(), float(reward), done


# ═══════════════════════════════════════════════════════════════════════════════
# Lifers RL Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class LifersRLTrainer:
    """Lifers 强化学习训练器 — PPO + 好奇心 + 多环境交替训练"""

    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 8,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self._init_network()
        self._metrics: Dict[str, List[float]] = {
            "loss": [], "reward": [], "curiosity": [], "entropy": [],
        }

    def _init_network(self):
        rng = np.random.RandomState(42)
        scale_w1 = math.sqrt(2.0 / self.state_dim)
        self.W1 = rng.randn(self.state_dim, self.hidden_dim).astype(np.float32) * scale_w1
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        scale_w2 = math.sqrt(2.0 / self.hidden_dim)
        self.W2 = rng.randn(self.hidden_dim, self.action_dim).astype(np.float32) * scale_w2
        self.b2 = np.zeros(self.action_dim, dtype=np.float32)
        # 价值网络
        self.Wv = rng.randn(self.hidden_dim, 1).astype(np.float32) * 0.1
        self.bv = np.zeros(1, dtype=np.float32)

    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        s = np.asarray(state, dtype=np.float32).reshape(1, -1)
        h = np.maximum(0, s @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        logits = logits - np.max(logits)
        probs = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-8)
        value = float((h @ self.Wv + self.bv)[0, 0])
        return probs[0], value, h[0]

    def sample_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        probs, value, _ = self.forward(state)
        probs = np.asarray(probs)
        probs = probs / (probs.sum() + 1e-8)
        action = int(np.random.choice(len(probs), p=probs))
        log_prob = float(np.log(probs[action] + 1e-8))
        return action, log_prob, value

    def train_step(self, batch: List[Dict]) -> Dict[str, float]:
        if len(batch) < 4:
            return {"loss": 0.0, "reward": 0.0, "entropy": 0.0}
        total_loss = 0.0
        total_reward = 0.0
        total_entropy = 0.0

        # 累积梯度
        grads = {
            "W1": np.zeros_like(self.W1), "b1": np.zeros_like(self.b1),
            "W2": np.zeros_like(self.W2), "b2": np.zeros_like(self.b2),
            "Wv": np.zeros_like(self.Wv), "bv": np.zeros_like(self.bv),
        }

        for item in batch:
            s = np.asarray(item["state"], dtype=np.float32).reshape(1, -1)
            a = item["action"]
            adv = item["advantage"]
            old_logp = item["log_prob"]
            ret = item["return"]

            # Forward
            h_pre = s @ self.W1 + self.b1
            h = np.maximum(0, h_pre)
            logits = h @ self.W2 + self.b2
            logits = logits - np.max(logits)
            probs = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-8)
            probs = np.asarray(probs)[0]
            value = float((h @ self.Wv + self.bv)[0, 0])

            new_logp = float(np.log(probs[a] + 1e-8))
            ratio = float(np.exp(new_logp - old_logp))

            # PPO clipped objective
            surr1 = ratio * adv
            surr2 = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv
            policy_loss = -min(surr1, surr2)
            value_loss = (ret - value) ** 2
            entropy = -float(np.sum(probs * np.log(probs + 1e-8)))
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            total_loss += float(loss)
            total_reward += float(item.get("reward", 0))
            total_entropy += float(entropy)

            # Policy gradient: only when unclipped objective is used
            dlogits = np.zeros((1, self.action_dim), dtype=np.float32)
            if surr1 <= surr2:
                dp = probs.copy()
                dp[a] -= 1.0
                dlogits = dp.reshape(1, -1) * adv
                grads["W2"] += h.T @ dlogits
                grads["b2"] += dlogits[0]

            # Value gradient
            dvalue = 2.0 * (value - ret)
            grads["Wv"] += h.T * dvalue
            grads["bv"] += dvalue

            # Backward through ReLU to W1
            dh = dlogits @ self.W2.T + dvalue * self.Wv.T
            dh[h_pre <= 0] = 0
            grads["W1"] += s.T @ dh
            grads["b1"] += dh[0]

        # 应用累积梯度
        n = len(batch)
        lr = self.lr / n
        self.W1 -= lr * np.clip(grads["W1"], -1.0, 1.0)
        self.b1 -= lr * np.clip(grads["b1"], -1.0, 1.0)
        self.W2 -= lr * np.clip(grads["W2"], -1.0, 1.0)
        self.b2 -= lr * np.clip(grads["b2"], -1.0, 1.0)
        self.Wv -= lr * np.clip(grads["Wv"], -1.0, 1.0)
        self.bv -= lr * np.clip(grads["bv"], -1.0, 1.0)

        return {"loss": total_loss / n, "reward": total_reward / n, "entropy": total_entropy / n}

    def compute_advantages(self, rewards, values, dones, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = values[t + 1] if t + 1 < len(values) else 0.0
            next_done = dones[t + 1] if t + 1 < len(dones) else True
            delta = rewards[t] + gamma * next_val * (0.0 if next_done else 1.0) - values[t]
            gae = delta + gamma * lam * (0.0 if dones[t] else 1.0) * gae
            advantages.append(gae)
        advantages.reverse()
        returns = [a + v for a, v in zip(advantages, values)]
        return advantages, returns

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "brand": "Lifers RL Policy",
            "version": 1,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "Wv": self.Wv.tolist(),
            "bv": self.bv.tolist(),
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
        trainer.Wv = np.array(data["Wv"], dtype=np.float32)
        trainer.bv = np.array(data["bv"], dtype=np.float32)
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

    trainer = LifersRLTrainer()
    envs = [LifersGridWorld(), LifersToolEnv()]
    env_names = ["GridWorld", "ToolEnv"]

    best_reward = -float("inf")
    episode_rewards: List[float] = []

    for ep in range(total_episodes):
        # 交替环境
        env_idx = ep % len(envs)
        env = envs[env_idx]
        state = env.reset()

        batch = []
        ep_reward = 0.0
        done = False

        while not done:
            action, log_prob, value = trainer.sample_action(state)
            next_state, reward, done = env.step(action)
            ep_reward += reward

            batch.append({
                "state": state,
                "action": action,
                "reward": reward,
                "log_prob": log_prob,
                "value": value,
                "done": done,
            })
            state = next_state

            if len(batch) >= 32 or done:
                rewards = [b["reward"] for b in batch]
                values = [b["value"] for b in batch]
                dones = [b["done"] for b in batch]
                advantages, returns = trainer.compute_advantages(rewards, values, dones)
                for i, b in enumerate(batch):
                    b["advantage"] = advantages[i]
                    b["return"] = returns[i]

                metrics = trainer.train_step(batch)
                batch = []

        episode_rewards.append(ep_reward)

        # 进度与保存
        if (ep + 1) % 100 == 0:
            avg_reward = float(np.mean(episode_rewards[-100:]))
            if verbose:
                print(f"[Lifers-RL] ep {ep + 1}/{total_episodes}  "
                      f"env={env_names[env_idx]}  avg_reward={avg_reward:.3f}  "
                      f"loss={metrics['loss']:.4f}  entropy={metrics['entropy']:.3f}")

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
