"""
Lifers Simulation v2 — 仿真质量评估器 (n-gram + 3层MLP + 生成场景)
品牌化权重: weights/lifers_simulation_evaluator.json
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as cpu_np
from lifers.core.compute_backend import get_compute_backend
np, _DEVICE, _GPU_INFO = get_compute_backend()

ROOT = Path(__file__).resolve().parent.parent.parent
N_OUTPUTS = 4
OUTPUT_NAMES = ["physics_accuracy", "timing_precision", "collision_rate", "success_rate"]

# 扩展场景模板 — 涵盖8个机器人领域
_SCENE_TEMPLATES = [
    ("{env}机器人{task}，{condition}", [0.85, 0.80, 0.06, 0.85]),
    ("{env}{task}操作，{condition}", [0.82, 0.85, 0.04, 0.88]),
    ("{env}自主{task}系统，{condition}", [0.88, 0.82, 0.05, 0.86]),
    ("多机器人{env}{task}，{condition}", [0.78, 0.75, 0.08, 0.80]),
    ("人机协作{env}{task}", [0.84, 0.88, 0.03, 0.90]),
    ("{env}智能{task}，{condition}", [0.86, 0.83, 0.04, 0.87]),
    ("{env}高精度{task}，{condition}", [0.92, 0.91, 0.02, 0.93]),
    ("{env}大规模{task}仿真", [0.76, 0.78, 0.09, 0.79]),
]

_ENVS = ["室内", "户外", "水下", "太空", "工业车间", "仓储", "农田", "城市道路",
         "矿场", "医院", "港口", "变电站"]
_TASKS = ["导航", "巡检", "搬运", "装配", "焊接", "喷涂", "分拣", "检测",
          "清扫", "配送", "搜救", "监控"]
_CONDITIONS = ["复杂光照条件", "动态障碍物环境", "多干扰源", "高噪声传感器",
               "极端温度", "强电磁干扰", "GPS拒止环境", "密集人群",
               "多变天气", "通信受限", "能源约束", "实时性要求高"]


def _generate_sim_scenes(n: int = 200) -> List:
    rng = np.random.RandomState(42)
    scenes = []
    for _ in range(n):
        t_idx = rng.randint(0, len(_SCENE_TEMPLATES))
        tmpl, base_scores = _SCENE_TEMPLATES[t_idx]
        env = rng.choice(_ENVS)
        task = rng.choice(_TASKS)
        cond = rng.choice(_CONDITIONS)
        text = tmpl.format(env=env, task=task, condition=cond)
        # 添加噪声
        noise = rng.uniform(-0.03, 0.03, 4)
        scores = np.clip(np.array(base_scores) + noise, 0, 1)
        scenes.append((text, scores[0], scores[1], scores[2], scores[3]))
    return scenes


def _extract_ngrams(texts: List[str], max_features: int = 256) -> dict:
    counter = Counter()
    for t in texts:
        for i in range(len(t) - 1):
            counter[t[i:i+2]] += 1
        for i in range(len(t) - 2):
            counter[t[i:i+3]] += 1
        for w in ["机器人", "导航", "巡检", "装配", "焊接", "检测", "无人机",
                  "机械臂", "自主", "协作", "仿真", "室内", "户外", "水下", "太空"]:
            if w in t:
                counter[f"W:{w}"] += 1
    return {k: i for i, (k, _) in enumerate(counter.most_common(max_features))}


def _text_features(texts: List[str], vocab: dict, n_features: int) -> np.ndarray:
    X = np.zeros((len(texts), n_features), dtype=np.float32)
    for i, t in enumerate(texts):
        for j in range(len(t) - 1):
            idx = vocab.get(t[j:j+2], -1)
            if idx >= 0:
                X[i, idx] += 1.0
        for j in range(len(t) - 2):
            idx = vocab.get(t[j:j+3], -1)
            if idx >= 0:
                X[i, idx] += 1.0
        for w in ["机器人", "导航", "巡检", "装配", "焊接", "检测", "无人机",
                  "机械臂", "自主", "协作", "仿真", "室内", "户外", "水下", "太空"]:
            if w in t:
                idx = vocab.get(f"W:{w}", -1)
                if idx >= 0:
                    X[i, idx] += 1.0
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / norms


class SimulationEvaluator:
    def __init__(self, n_features=256, hidden1=128, hidden2=64):
        self.n_features = n_features
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        rng = np.random.RandomState(42)
        self.W1 = rng.randn(n_features, hidden1).astype(np.float32) * np.sqrt(2.0 / n_features)
        self.b1 = np.zeros(hidden1, dtype=np.float32)
        self.W2 = rng.randn(hidden1, hidden2).astype(np.float32) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2, dtype=np.float32)
        self.W3 = rng.randn(hidden2, N_OUTPUTS).astype(np.float32) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros(N_OUTPUTS, dtype=np.float32)

    def forward(self, X):
        h1 = np.maximum(0, X @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        return h2 @ self.W3 + self.b3

    def predict(self, X):
        raw = self.forward(X)
        return 1.0 / (1.0 + np.exp(-raw))


def _adam_update(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    return param - lr * m_hat / (np.sqrt(v_hat) + 1e-8), m, v


def train_simulation_evaluator(
    n_epochs: int = 60, lr: float = 1e-3,
    save_path: Optional[Path] = None, verbose: bool = True,
) -> SimulationEvaluator:
    if save_path is None:
        save_path = ROOT / "weights" / "lifers_simulation_evaluator.json"

    scenes = _generate_sim_scenes(200)
    texts = [s[0] for s in scenes]
    y = np.array([[s[1], s[2], 1-s[3], s[4]] for s in scenes], dtype=np.float32)

    if verbose:
        print(f"[Lifers-Simulation v2] 场景: {len(scenes)} 个")

    vocab = _extract_ngrams(texts, max_features=256)
    n_feat = len(vocab)
    X = _text_features(texts, vocab, n_feat)

    n_train = int(len(X) * 0.85)
    idx = np.random.RandomState(42).permutation(len(X))
    X_train, y_train = X[idx[:n_train]], y[idx[:n_train]]
    X_val, y_val = X[idx[n_train:]], y[idx[n_train:]]

    model = SimulationEvaluator(n_features=n_feat)
    batch_size = 32
    best_loss = float("inf")
    t = 0

    ms = {k: np.zeros_like(v) for k, v in vars(model).items() if k.startswith("W") or k.startswith("b")}
    vs = {k: np.zeros_like(v) for k, v in vars(model).items() if k.startswith("W") or k.startswith("b")}

    for epoch in range(n_epochs):
        perm = np.random.permutation(n_train)
        total_loss = 0.0

        for start in range(0, n_train, batch_size):
            batch_idx = perm[start:start + batch_size]
            Xb, yb = X_train[batch_idx], y_train[batch_idx]
            t += 1

            h1_pre = Xb @ model.W1 + model.b1
            h1 = np.maximum(0, h1_pre)
            h2_pre = h1 @ model.W2 + model.b2
            h2 = np.maximum(0, h2_pre)
            logits = h2 @ model.W3 + model.b3
            preds = 1.0 / (1.0 + np.exp(-logits))

            loss = np.mean((preds - yb) ** 2)
            total_loss += loss

            dlogits = 2 * (preds - yb) * preds * (1 - preds) / len(yb)

            gW3 = h2.T @ dlogits
            gb3 = np.sum(dlogits, axis=0)
            dh2 = dlogits @ model.W3.T
            dh2[h2_pre <= 0] = 0
            gW2 = h1.T @ dh2
            gb2 = np.sum(dh2, axis=0)
            dh1 = dh2 @ model.W2.T
            dh1[h1_pre <= 0] = 0
            gW1 = Xb.T @ dh1
            gb1 = np.sum(dh1, axis=0)

            grads = {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2, "W3": gW3, "b3": gb3}
            for k in grads:
                v = getattr(model, k)
                new_v, new_m, new_vv = _adam_update(v, grads[k], ms[k], vs[k], t, lr)
                setattr(model, k, new_v)
                ms[k], vs[k] = new_m, new_vv

        val_loss = np.mean((model.predict(X_val) - y_val) ** 2)
        if val_loss < best_loss:
            best_loss = val_loss
            _save_sim_model(model, save_path)

        if (epoch + 1) % 15 == 0 and verbose:
            print(f"[Lifers-Simulation v2] epoch {epoch + 1}/{n_epochs}  "
                  f"loss={total_loss/(n_train//batch_size+1):.6f}  val_loss={val_loss:.6f}")

    _save_sim_model(model, save_path)
    if verbose:
        print(f"[Lifers-Simulation v2] 完成 best_val_loss={best_loss:.6f} → {save_path}")
    return model


def _save_sim_model(model: SimulationEvaluator, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "brand": "Lifers Simulation Evaluator v2",
        "version": 2,
        "n_features": model.n_features,
        "hidden1": model.hidden1,
        "hidden2": model.hidden2,
        "outputs": OUTPUT_NAMES,
        "W1": model.W1.tolist(), "b1": model.b1.tolist(),
        "W2": model.W2.tolist(), "b2": model.b2.tolist(),
        "W3": model.W3.tolist(), "b3": model.b3.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    epochs = int(os.environ.get("LIFERS_SIMULATION_EPOCHS", "60"))
    out = ROOT / "weights" / "lifers_simulation_evaluator.json"
    print("[Lifers-Simulation v2] n-gram + 3层MLP + 200生成场景")
    t0 = time.time()
    train_simulation_evaluator(n_epochs=epochs, save_path=out, verbose=True)
    print(f"[Lifers-Simulation v2] 耗时={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
