"""
Lifers Social v2 — 6类社交意图分类器 (n-gram特征 + 3层MLP + Adam)
品牌化权重: weights/lifers_social_classifier.json
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as cpu_np
from lifers.core.compute_backend import get_compute_backend
np, _DEVICE, _GPU_INFO = get_compute_backend()

ROOT = Path(__file__).resolve().parent.parent.parent
N_CLASSES = 6
CLASS_NAMES = ["问候", "协作", "情感支持", "信息分享", "提醒催促", "社交协调"]


def _extract_ngrams(texts: List[str], max_features: int = 256) -> Dict[str, int]:
    """从文本列表中提取高频n-gram特征"""
    counter = Counter()
    for t in texts:
        # 字符级 bigram + trigram
        for i in range(len(t) - 1):
            counter[t[i:i+2]] += 1
        for i in range(len(t) - 2):
            counter[t[i:i+3]] += 1
        # 词级关键词 (中文按字切分)
        for w in ["一起", "帮", "合作", "理解", "发现", "分享", "记得", "任务",
                  "联系", "协调", "邀请", "团队", "同步", "早安", "抱歉", "感谢",
                  "提醒", "会议", "进展", "建议", "报告", "通知", "检查", "更新"]:
            if w in t:
                counter[f"W:{w}"] += 1
    return {k: i for i, (k, _) in enumerate(counter.most_common(max_features))}


def _text_features(texts: List[str], vocab: Dict[str, int], n_features: int) -> np.ndarray:
    """文本 → n-gram特征矩阵"""
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
        for w in ["一起", "帮", "合作", "理解", "发现", "分享", "记得", "任务",
                  "联系", "协调", "邀请", "团队", "同步", "早安", "抱歉", "感谢",
                  "提醒", "会议", "进展", "建议", "报告", "通知", "检查", "更新"]:
            if w in t:
                idx = vocab.get(f"W:{w}", -1)
                if idx >= 0:
                    X[i, idx] += 1.0
        # 统计特征
        X[i, idx + 1] if idx + 1 < n_features else None
    # 归一化
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X = X / norms
    return X


class SocialClassifier:
    """3层MLP: n_feat→128→64→6"""

    def __init__(self, n_features=256, hidden1=128, hidden2=64, n_classes=N_CLASSES):
        self.n_features = n_features
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.n_classes = n_classes
        rng = np.random.RandomState(42)
        self.W1 = rng.randn(n_features, hidden1).astype(np.float32) * np.sqrt(2.0 / n_features)
        self.b1 = np.zeros(hidden1, dtype=np.float32)
        self.W2 = rng.randn(hidden1, hidden2).astype(np.float32) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2, dtype=np.float32)
        self.W3 = rng.randn(hidden2, n_classes).astype(np.float32) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros(n_classes, dtype=np.float32)

    def forward(self, X):
        h1 = np.maximum(0, X @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        return h2 @ self.W3 + self.b3

    def predict(self, X):
        logits = self.forward(X)
        return np.argmax(logits, axis=1), logits


def _adam_update(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return param, m, v


def train_social_classifier(
    n_epochs: int = 60, lr: float = 1e-3,
    save_path: Optional[Path] = None, verbose: bool = True,
    max_samples: int = 50000,
) -> SocialClassifier:
    if save_path is None:
        save_path = ROOT / "weights" / "lifers_social_classifier.json"

    # 加载JSONL数据
    data_dir = ROOT / "data"
    texts, labels = [], []
    jsonl_path = data_dir / "social_samples.jsonl"
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
            print("[Lifers-Social v2] 数据不足，使用内置样本")
        return _train_fallback(n_epochs, lr, save_path, verbose)

    if verbose:
        print(f"[Lifers-Social v2] 数据: {len(texts):,} 样本")

    # 构建n-gram词表
    vocab = _extract_ngrams(texts, max_features=256)
    n_feat = len(vocab)
    X = _text_features(texts, vocab, n_feat)
    y = np.array(labels, dtype=np.int32)

    # 训练/验证集 80/20
    n_train = int(len(X) * 0.8)
    idx = np.random.RandomState(42).permutation(len(X))
    X_train, y_train = X[idx[:n_train]], y[idx[:n_train]]
    X_val, y_val = X[idx[n_train:]], y[idx[n_train:]]

    model = SocialClassifier(n_features=n_feat)
    batch_size = 128
    best_acc = 0.0

    # Adam状态
    t = 0
    mW1 = np.zeros_like(model.W1); vW1 = np.zeros_like(model.W1)
    mb1 = np.zeros_like(model.b1); vb1 = np.zeros_like(model.b1)
    mW2 = np.zeros_like(model.W2); vW2 = np.zeros_like(model.W2)
    mb2 = np.zeros_like(model.b2); vb2 = np.zeros_like(model.b2)
    mW3 = np.zeros_like(model.W3); vW3 = np.zeros_like(model.W3)
    mb3 = np.zeros_like(model.b3); vb3 = np.zeros_like(model.b3)

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

            # Softmax + CrossEntropy
            logits_max = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            loss = -np.mean(np.log(probs[np.arange(len(yb)), yb] + 1e-8))
            total_loss += loss

            dlogits = probs.copy()
            dlogits[np.arange(len(yb)), yb] -= 1.0
            dlogits /= len(yb)

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

            model.W1, mW1, vW1 = _adam_update(model.W1, gW1, mW1, vW1, t, lr)
            model.b1, mb1, vb1 = _adam_update(model.b1, gb1, mb1, vb1, t, lr)
            model.W2, mW2, vW2 = _adam_update(model.W2, gW2, mW2, vW2, t, lr)
            model.b2, mb2, vb2 = _adam_update(model.b2, gb2, mb2, vb2, t, lr)
            model.W3, mW3, vW3 = _adam_update(model.W3, gW3, mW3, vW3, t, lr)
            model.b3, mb3, vb3 = _adam_update(model.b3, gb3, mb3, vb3, t, lr)

        # 验证
        val_preds, _ = model.predict(X_val)
        val_acc = np.mean(val_preds == y_val)
        if val_acc > best_acc:
            best_acc = val_acc
            _save_social_model(model, vocab, save_path)

        if (epoch + 1) % 15 == 0 and verbose:
            print(f"[Lifers-Social v2] epoch {epoch + 1}/{n_epochs}  "
                  f"loss={total_loss/(n_train//batch_size+1):.4f}  val_acc={val_acc:.3f}")

        if val_acc > 0.97 and epoch > 10:
            if verbose:
                print(f"[Lifers-Social v2] 早停 epoch {epoch+1} val_acc={val_acc:.3f}")
            break

    _save_social_model(model, vocab, save_path)
    if verbose:
        print(f"[Lifers-Social v2] 完成 best_val_acc={best_acc:.3f} → {save_path}")
    return model


# 内置6类社交意图样本 (text, label) — label对应 CLASS_NAMES
_SOCIAL_SAMPLES = [
    ("主人早上好", 0), ("嗨Lifers", 0), ("你好", 0), ("早安", 0), ("下午好", 0),
    ("帮我看看这个代码", 1), ("一起完成这个任务", 1), ("协助我处理下数据", 1),
    ("配合我做下分析", 1), ("帮我分析一下", 1),
    ("最近好累啊", 2), ("心情不太好", 2), ("我需要安慰", 2), ("听我说说心事", 2),
    ("和你聊聊我的感受", 2), ("好难过", 2),
    ("分享一下最近的新闻", 3), ("看看这个有趣的发现", 3), ("告诉你一个好消息", 3),
    ("我发现了一个技巧", 3), ("整理些资料给我", 3),
    ("记得提醒我开会", 4), ("别忘了提交报告", 4), ("帮忙催下进度", 4),
    ("检查下任务列表", 4), ("提醒我下午有约会", 4),
    ("晚上一起吃饭吧", 5), ("周末有什么计划", 5), ("约个时间讨论", 5),
    ("我们一起去健身", 5), ("团队活动安排好了吗", 5),
]

def _social_feature(text: str) -> np.ndarray:
    """简单特征：字符bigram计数 + 关键词匹配 (256维)"""
    feats = np.zeros(256, dtype=np.float32)
    for i in range(len(text) - 1):
        idx = hash(text[i:i+2]) % 256
        feats[idx] += 1.0
    norm = np.linalg.norm(feats) + 1e-8
    return feats / norm

def _train_fallback(n_epochs, lr, save_path, verbose):
    """使用内置数据作为回退"""
    samples = list(_SOCIAL_SAMPLES)
    X = np.array([_social_feature(t) for t, _ in samples], dtype=np.float32)
    y = np.array([l for _, l in samples], dtype=np.int32)
    model = SocialClassifier(n_features=256)
    rng = cpu_np.random.RandomState(123)
    best_acc = 0.0
    for epoch in range(n_epochs):
        indices = cpu_np.arange(len(X), dtype=cpu_np.int32)
        rng.shuffle(indices)
        correct = 0
        for idx in indices:
            x_i, y_i = X[idx], y[idx]
            logits = model.forward(x_i.reshape(1, -1))
            probs = np.exp(logits - np.max(logits))
            probs = probs / probs.sum()
            p = probs[0]
            dlogits = p.copy()
            dlogits[y_i] -= 1.0
            h_pre = x_i @ model.W1 + model.b1
            h = np.maximum(0, h_pre)
            model.W2 -= lr * np.clip(np.outer(h, dlogits), -0.5, 0.5)
            model.b2 -= lr * np.clip(dlogits, -0.5, 0.5)
            dh = dlogits @ model.W2.T
            dh[h_pre <= 0] = 0
            model.W1 -= lr * np.clip(np.outer(x_i, dh), -0.5, 0.5)
            model.b1 -= lr * np.clip(dh, -0.5, 0.5)
            if np.argmax(p) == y_i:
                correct += 1
        acc = correct / len(indices)
        if acc > best_acc:
            best_acc = acc
        if (epoch + 1) % 15 == 0 and verbose:
            print(f"[Lifers-Social v2] fallback epoch {epoch + 1}/{n_epochs} acc={acc:.3f}")
    _save_social_model(model, {}, save_path)
    if verbose:
        print(f"[Lifers-Social v2] fallback best_acc={best_acc:.3f}")
    if verbose:
        print(f"[Lifers-Social v2] fallback best_acc={best_acc:.3f}")
    return model


def _save_social_model(model, vocab, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "brand": "Lifers Social Classifier v2",
        "version": 2,
        "n_features": model.n_features,
        "hidden1": model.hidden1,
        "hidden2": model.hidden2,
        "n_classes": model.n_classes,
        "classes": CLASS_NAMES,
        "W1": model.W1.tolist(), "b1": model.b1.tolist(),
        "W2": model.W2.tolist(), "b2": model.b2.tolist(),
        "W3": model.W3.tolist(), "b3": model.b3.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    epochs = int(os.environ.get("LIFERS_SOCIAL_EPOCHS", "60"))
    out = ROOT / "weights" / "lifers_social_classifier.json"
    print("[Lifers-Social v2] n-gram + 3层MLP + Adam")
    t0 = time.time()
    train_social_classifier(n_epochs=epochs, save_path=out, verbose=True)
    print(f"[Lifers-Social v2] 耗时={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
