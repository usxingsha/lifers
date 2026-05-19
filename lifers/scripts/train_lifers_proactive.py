"""
Lifers Proactive v2 — 二分类主动行为预测器 (n-gram特征 + 3层MLP + Adam)
品牌化权重: weights/lifers_proactive_predictor.json
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional

import numpy as cpu_np
from lifers.core.compute_backend import get_compute_backend
np, _DEVICE, _GPU_INFO = get_compute_backend()

ROOT = Path(__file__).resolve().parent.parent.parent


def _extract_ngrams(texts: List[str], max_features: int = 256) -> Dict[str, int]:
    counter = Counter()
    for t in texts:
        for i in range(len(t) - 1):
            counter[t[i:i+2]] += 1
        for i in range(len(t) - 2):
            counter[t[i:i+3]] += 1
        for w in ["检测", "主动", "用户", "提醒", "发现", "异常", "通知", "预警",
                  "任务", "危险", "系统", "安全", "更新", "报告", "同步", "忽略"]:
            if w in t:
                counter[f"W:{w}"] += 1
    return {k: i for i, (k, _) in enumerate(counter.most_common(max_features))}


def _text_features(texts: List[str], vocab: Dict[str, int], n_features: int) -> np.ndarray:
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
        for w in ["检测", "主动", "用户", "提醒", "发现", "异常", "通知", "预警",
                  "任务", "危险", "系统", "安全", "更新", "报告", "同步", "忽略"]:
            if w in t:
                idx = vocab.get(f"W:{w}", -1)
                if idx >= 0:
                    X[i, idx] += 1.0
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / norms


class ProactivePredictor:
    def __init__(self, n_features=256, hidden1=128, hidden2=64):
        self.n_features = n_features
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        rng = np.random.RandomState(42)
        self.W1 = rng.randn(n_features, hidden1).astype(np.float32) * np.sqrt(2.0 / n_features)
        self.b1 = np.zeros(hidden1, dtype=np.float32)
        self.W2 = rng.randn(hidden1, hidden2).astype(np.float32) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2, dtype=np.float32)
        self.W3 = rng.randn(hidden2, 1).astype(np.float32) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros(1, dtype=np.float32)

    def forward(self, X):
        h1 = np.maximum(0, X @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        return h2 @ self.W3 + self.b3

    def predict(self, X):
        logits = self.forward(X)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return (probs > 0.5).astype(np.int32).flatten(), probs.flatten()


def _adam_update(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    return param - lr * m_hat / (np.sqrt(v_hat) + 1e-8), m, v


def train_proactive_predictor(
    n_epochs: int = 80, lr: float = 1e-3,
    save_path: Optional[Path] = None, verbose: bool = True,
    max_samples: int = 50000,
) -> ProactivePredictor:
    if save_path is None:
        save_path = ROOT / "weights" / "lifers_proactive_predictor.json"

    data_dir = ROOT / "data"
    texts, labels = [], []
    jsonl_path = data_dir / "proactive_samples.jsonl"
    if jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                try:
                    obj = json.loads(line)
                    texts.append(obj.get("text", ""))
                    labels.append(int(obj.get("label", 0)))
                except (json.JSONDecodeError, ValueError):
                    continue

    if len(texts) < 100:
        if verbose:
            print("[Lifers-Proactive v2] 数据不足")
        return None

    if verbose:
        print(f"[Lifers-Proactive v2] 数据: {len(texts):,} 样本")

    vocab = _extract_ngrams(texts, max_features=256)
    n_feat = len(vocab)
    X = _text_features(texts, vocab, n_feat)
    y = np.array(labels, dtype=np.float32)

    n_train = int(len(X) * 0.8)
    idx = np.random.RandomState(42).permutation(len(X))
    X_train, y_train = X[idx[:n_train]], y[idx[:n_train]]
    X_val, y_val = X[idx[n_train:]], y[idx[n_train:]]

    model = ProactivePredictor(n_features=n_feat)
    batch_size = 128
    best_acc = 0.0
    t = 0

    mW1, vW1 = np.zeros_like(model.W1), np.zeros_like(model.W1)
    mb1, vb1 = np.zeros_like(model.b1), np.zeros_like(model.b1)
    mW2, vW2 = np.zeros_like(model.W2), np.zeros_like(model.W2)
    mb2, vb2 = np.zeros_like(model.b2), np.zeros_like(model.b2)
    mW3, vW3 = np.zeros_like(model.W3), np.zeros_like(model.W3)
    mb3, vb3 = np.zeros_like(model.b3), np.zeros_like(model.b3)

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
            logits = (h2 @ model.W3 + model.b3).flatten()

            # BCE loss
            probs = 1.0 / (1.0 + np.exp(-logits))
            loss = -np.mean(yb * np.log(probs + 1e-8) + (1 - yb) * np.log(1 - probs + 1e-8))
            total_loss += loss

            dlogits = (probs - yb) / len(yb)

            gW3 = h2.T @ dlogits.reshape(-1, 1)
            gb3 = np.sum(dlogits)
            dh2 = dlogits.reshape(-1, 1) @ model.W3.T
            dh2[h2_pre <= 0] = 0
            gW2 = h1.T @ dh2
            gb2 = np.sum(dh2, axis=0)
            dh1 = dh2 @ model.W2.T
            dh1[h1_pre <= 0] = 0
            gW1 = Xb.T @ dh1
            gb1 = np.sum(dh1, axis=0)

            model.W1, mW1, vW1 = _adam_update(model.W1, gW1, mW1, vW1, t, lr)
            model.b1, mb1, vb1 = _adam_update(model.b1, gb1, mb1, vb1, t, lr)
            model.W2, mW2, vW2 = _adam_update(model.W2, gW2, mW2, vW2, t, lr)
            model.b2, mb2, vb2 = _adam_update(model.b2, gb2, mb2, vb2, t, lr)
            model.W3, mW3, vW3 = _adam_update(model.W3, gW3, mW3, vW3, t, lr)
            model.b3, mb3, vb3 = _adam_update(model.b3, np.array([gb3]), mb3, vb3, t, lr)

        val_preds, val_probs = model.predict(X_val)
        val_acc = np.mean(val_preds == y_val)
        if val_acc > best_acc:
            best_acc = val_acc
            _save_proactive_model(model, save_path)

        if (epoch + 1) % 20 == 0 and verbose:
            print(f"[Lifers-Proactive v2] epoch {epoch + 1}/{n_epochs}  "
                  f"loss={total_loss/(n_train//batch_size+1):.4f}  val_acc={val_acc:.3f}")

        if val_acc > 0.97 and epoch > 10:
            if verbose:
                print(f"[Lifers-Proactive v2] 早停 epoch {epoch+1}")
            break

    _save_proactive_model(model, save_path)
    if verbose:
        print(f"[Lifers-Proactive v2] 完成 best_val_acc={best_acc:.3f} → {save_path}")
    return model


def _save_proactive_model(model: ProactivePredictor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "brand": "Lifers Proactive Predictor v2",
        "version": 2,
        "n_features": model.n_features,
        "hidden1": model.hidden1,
        "hidden2": model.hidden2,
        "W1": model.W1.tolist(), "b1": model.b1.tolist(),
        "W2": model.W2.tolist(), "b2": model.b2.tolist(),
        "W3": model.W3.tolist(), "b3": model.b3.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    epochs = int(os.environ.get("LIFERS_PROACTIVE_EPOCHS", "80"))
    out = ROOT / "weights" / "lifers_proactive_predictor.json"
    print("[Lifers-Proactive v2] n-gram + 3层MLP + Adam")
    t0 = time.time()
    train_proactive_predictor(n_epochs=epochs, save_path=out, verbose=True)
    print(f"[Lifers-Proactive v2] 耗时={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
