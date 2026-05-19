"""
Lifers Telemetry v2 — 自编码器异常检测 (Autoencoder + 重建误差)
品牌化权重: weights/lifers_telemetry_detector.json
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
N_FEATURES = 8
FEATURE_NAMES = ["cpu_pct", "ram_pct", "disk_pct", "net_kbps", "temp_c",
                 "proc_count", "io_wait", "ctx_switches"]


def _generate_telemetry_data(n_normal: int = 2000, n_anomaly: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """生成合成遥测数据"""
    rng = np.random.RandomState(42)

    normal_centers = [
        [25, 45, 30, 120, 42, 8, 2, 5000],
        [15, 35, 28, 80, 38, 5, 1, 3000],
        [30, 50, 32, 200, 45, 10, 3, 7000],
        [20, 40, 30, 100, 40, 7, 2, 4500],
        [35, 55, 33, 300, 48, 12, 4, 9000],
    ]

    normal = []
    for _ in range(n_normal):
        center = normal_centers[rng.randint(0, len(normal_centers))]
        noise = rng.randn(N_FEATURES) * np.array([3, 2, 0.5, 20, 1.5, 1, 0.5, 500])
        sample = np.array(center) + noise
        sample = np.maximum(sample, [1, 5, 10, 1, 20, 1, 0, 100])
        normal.append(sample.astype(np.float32))
    normal = np.array(normal)

    # 异常类型
    anomalies = []
    anomaly_types = [
        lambda: normal[rng.randint(0, len(normal))] * np.array([3.5, 1, 1, 1, 1.8, 1, 1, 1]),  # CPU暴增
        lambda: normal[rng.randint(0, len(normal))] * np.array([1, 2.5, 1, 1, 1.5, 2, 1, 1]),  # RAM泄漏
        lambda: normal[rng.randint(0, len(normal))] * np.array([1, 1, 3.5, 1, 1, 1, 3, 1]),    # 磁盘异常
        lambda: normal[rng.randint(0, len(normal))] * np.array([1, 1, 1, 30, 1, 1, 1, 5]),      # 网络洪水
        lambda: normal[rng.randint(0, len(normal))] * np.array([2, 1.8, 1, 1, 2.2, 2, 2, 2]),   # 综合过载
        lambda: normal[rng.randint(0, len(normal))] * np.array([0.1, 0.3, 1, 0.05, 0.5, 0.2, 0.1, 0.1]),  # 异常低
    ]
    for _ in range(n_anomaly):
        fn = rng.choice(anomaly_types)
        anomalies.append(fn().astype(np.float32))
    anomalies = np.array(anomalies)

    return normal, anomalies


class TelemetryAutoencoder:
    """自编码器: 8→32→16→8→32→8"""

    def __init__(self):
        rng = np.random.RandomState(42)
        self.W1 = rng.randn(8, 32).astype(np.float32) * np.sqrt(2.0 / 8)
        self.b1 = np.zeros(32, dtype=np.float32)
        self.W2 = rng.randn(32, 16).astype(np.float32) * np.sqrt(2.0 / 32)
        self.b2 = np.zeros(16, dtype=np.float32)
        self.W3 = rng.randn(16, 8).astype(np.float32) * np.sqrt(2.0 / 16)
        self.b3 = np.zeros(8, dtype=np.float32)
        self.W4 = rng.randn(8, 32).astype(np.float32) * np.sqrt(2.0 / 8)
        self.b4 = np.zeros(32, dtype=np.float32)
        self.W5 = rng.randn(32, 8).astype(np.float32) * np.sqrt(2.0 / 32)
        self.b5 = np.zeros(8, dtype=np.float32)

    def forward(self, X):
        h1 = np.maximum(0, X @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        latent = h2 @ self.W3 + self.b3
        h4 = np.maximum(0, latent @ self.W4 + self.b4)
        return h4 @ self.W5 + self.b5

    def reconstruction_error(self, X):
        recon = self.forward(X)
        return np.mean((X - recon) ** 2, axis=1)


def _adam_update(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    return param - lr * m_hat / (np.sqrt(v_hat) + 1e-8), m, v


def train_telemetry_detector(
    n_epochs: int = 50, lr: float = 1e-3,
    save_path: Optional[Path] = None, verbose: bool = True,
) -> TelemetryAutoencoder:
    if save_path is None:
        save_path = ROOT / "weights" / "lifers_telemetry_detector.json"

    normal, anomalies = _generate_telemetry_data()
    if verbose:
        print(f"[Lifers-Telemetry v2] 生成数据: {len(normal):,} normal + {len(anomalies):,} anomaly")

    # 标准化
    mean = normal.mean(axis=0)
    std = normal.std(axis=0) + 1e-8
    normal_norm = (normal - mean) / std
    anomalies_norm = (anomalies - mean) / std

    n_train = int(len(normal_norm) * 0.8)
    X_train = normal_norm[:n_train]
    X_val = normal_norm[n_train:]

    model = TelemetryAutoencoder()
    batch_size = 64
    best_loss = float("inf")
    t = 0

    ms = {k: np.zeros_like(v) for k, v in vars(model).items() if k.startswith("W") or k.startswith("b")}
    vs = {k: np.zeros_like(v) for k, v in vars(model).items() if k.startswith("W") or k.startswith("b")}

    for epoch in range(n_epochs):
        perm = np.random.permutation(n_train)
        total_loss = 0.0

        for start in range(0, n_train, batch_size):
            batch_idx = perm[start:start + batch_size]
            Xb = X_train[batch_idx]
            t += 1

            # Encoder
            h1_pre = Xb @ model.W1 + model.b1
            h1 = np.maximum(0, h1_pre)
            h2_pre = h1 @ model.W2 + model.b2
            h2 = np.maximum(0, h2_pre)
            latent = h2 @ model.W3 + model.b3

            # Decoder
            h4_pre = latent @ model.W4 + model.b4
            h4 = np.maximum(0, h4_pre)
            recon = h4 @ model.W5 + model.b5

            # MSE loss
            diff = recon - Xb
            loss = np.mean(diff ** 2)
            total_loss += loss

            # 反向传播
            dout = 2 * diff / len(Xb)

            # Decoder gradients
            gW5 = h4.T @ dout
            gb5 = np.sum(dout, axis=0)
            dh4 = dout @ model.W5.T
            dh4[h4_pre <= 0] = 0
            gW4 = latent.T @ dh4
            gb4 = np.sum(dh4, axis=0)
            dlatent = dh4 @ model.W4.T

            # Latent layer (no activation)
            gW3 = h2.T @ dlatent
            gb3 = np.sum(dlatent, axis=0)
            dh2 = dlatent @ model.W3.T
            dh2[h2_pre <= 0] = 0
            gW2 = h1.T @ dh2
            gb2 = np.sum(dh2, axis=0)
            dh1 = dh2 @ model.W2.T
            dh1[h1_pre <= 0] = 0
            gW1 = Xb.T @ dh1
            gb1 = np.sum(dh1, axis=0)

            grads = {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2,
                     "W3": gW3, "b3": gb3, "W4": gW4, "b4": gb4,
                     "W5": gW5, "b5": gb5}

            for k in grads:
                v = getattr(model, k)
                m_key, v_key = ms[k], vs[k]
                new_v, new_m, new_vv = _adam_update(v, grads[k], m_key, v_key, t, lr)
                setattr(model, k, new_v)
                ms[k], vs[k] = new_m, new_vv

        # 验证
        val_loss = np.mean((model.forward(X_val) - X_val) ** 2)
        if val_loss < best_loss:
            best_loss = val_loss

        if (epoch + 1) % 15 == 0 and verbose:
            print(f"[Lifers-Telemetry v2] epoch {epoch + 1}/{n_epochs}  "
                  f"train_loss={total_loss/(n_train//batch_size+1):.6f}  val_loss={val_loss:.6f}")

    # 计算阈值
    recon_errors = model.reconstruction_error(X_val)
    threshold = float(np.mean(recon_errors) + 3 * np.std(recon_errors))

    # 评估
    normal_errors = model.reconstruction_error(anomalies_norm[:100])
    anomaly_errors = model.reconstruction_error(anomalies_norm[:100])
    tp = np.sum(anomaly_errors > threshold)
    fn = np.sum(anomaly_errors <= threshold)
    recall = tp / max(tp + fn, 1)
    if verbose:
        print(f"[Lifers-Telemetry v2] val_loss={best_loss:.6f}  threshold={threshold:.4f}  "
              f"recall={recall:.3f}  anomalies_detected={tp}/100")

    _save_telemetry_model(model, save_path, mean, std, threshold)
    return model


def _save_telemetry_model(model, path, mean, std, threshold):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "brand": "Lifers Telemetry Autoencoder v2",
        "version": 2,
        "n_features": N_FEATURES,
        "features": FEATURE_NAMES,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "threshold": float(threshold),
        "W1": model.W1.tolist(), "b1": model.b1.tolist(),
        "W2": model.W2.tolist(), "b2": model.b2.tolist(),
        "W3": model.W3.tolist(), "b3": model.b3.tolist(),
        "W4": model.W4.tolist(), "b4": model.b4.tolist(),
        "W5": model.W5.tolist(), "b5": model.b5.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    out = ROOT / "weights" / "lifers_telemetry_detector.json"
    print("[Lifers-Telemetry v2] Autoencoder异常检测")
    t0 = time.time()
    train_telemetry_detector(save_path=out, verbose=True)
    print(f"[Lifers-Telemetry v2] 耗时={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
