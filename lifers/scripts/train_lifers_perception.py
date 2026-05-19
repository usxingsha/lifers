"""
Lifers Perception v2 — 6类场景分类器 (n-gram特征 + 3层MLP + Adam)
品牌化权重: weights/lifers_perception_classifier.json
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
N_CLASSES = 6
CLASS_NAMES = ["室内办公", "室内家庭", "室内公共", "户外街道", "户外自然", "户外运动"]


# 每类场景的高区分度关键词（必须出现在特征词表中）
PERCEPTION_KEYWORDS = [
    # 室内办公
    "办公桌", "键盘声", "打印机", "会议室", "工位", "白板", "投影", "邮件", "汇报", "打卡",
    # 室内家庭
    "沙发", "卧室", "厨房", "饭菜", "洗衣", "阳台", "拖鞋", "窗帘", "冰箱", "洗澡",
    # 室内公共
    "收银台", "货架", "购物车", "电梯", "挂号", "排队", "试衣间", "菜单", "阅览室", "展厅",
    # 户外街道
    "马路", "红绿灯", "斑马线", "公交站", "人行道", "鸣笛", "路灯", "交警", "堵车", "停车位",
    # 户外自然
    "鸟鸣", "溪流", "森林", "海边", "沙滩", "山峰", "花草", "瀑布", "星空", "露珠",
    # 户外运动
    "操场", "球场", "篮球", "足球", "跑步", "游泳", "哨声", "教练", "健身", "热身",
]


def _extract_ngrams(texts: List[str], max_features: int = 512) -> Dict[str, int]:
    counter = Counter()
    for t in texts:
        for i in range(len(t) - 1):
            counter[t[i:i+2]] += 1
        for i in range(len(t) - 2):
            counter[t[i:i+3]] += 1
    kw_count = len(PERCEPTION_KEYWORDS)
    ngram_slots = max(0, max_features - kw_count)
    # 强制保留所有感知关键词（索引 0..kw_count-1）
    vocab: Dict[str, int] = {}
    for i, w in enumerate(PERCEPTION_KEYWORDS):
        vocab[w] = i
    # n-gram 特征索引从 kw_count 开始
    for i, (k, _) in enumerate(counter.most_common(ngram_slots)):
        if k not in vocab:  # 避免关键词重复
            vocab[k] = kw_count + i
    return vocab


def _text_features(texts: List[str], vocab: Dict[str, int], n_features: int) -> np.ndarray:
    X = np.zeros((len(texts), n_features), dtype=np.float32)
    kw_count = len(PERCEPTION_KEYWORDS)
    for i, t in enumerate(texts):
        # 关键词特征（高权重二值）
        for j, w in enumerate(PERCEPTION_KEYWORDS):
            if w in t:
                X[i, j] = 3.0
        # n-gram 特征
        for j in range(len(t) - 1):
            idx = vocab.get(t[j:j+2], -1)
            if 0 <= idx < n_features and idx >= kw_count:
                X[i, idx] += 1.0
        for j in range(len(t) - 2):
            idx = vocab.get(t[j:j+3], -1)
            if 0 <= idx < n_features and idx >= kw_count:
                X[i, idx] += 1.0
    # 仅对 n-gram 部分做 L2 归一化
    ngram_part = X[:, kw_count:]
    norms = np.linalg.norm(ngram_part, axis=1, keepdims=True) + 1e-8
    X[:, kw_count:] = ngram_part / norms
    return X


class PerceptionClassifier:
    def __init__(self, n_features=512, hidden1=256, hidden2=128, n_classes=N_CLASSES):
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
    return param - lr * m_hat / (np.sqrt(v_hat) + 1e-8), m, v


def train_perception_classifier(
    n_epochs: int = 120, lr: float = 1e-3,
    save_path: Optional[Path] = None, verbose: bool = True,
    max_samples: int = 200000,
) -> PerceptionClassifier:
    if save_path is None:
        save_path = ROOT / "weights" / "lifers_perception_classifier.json"

    data_dir = ROOT / "data"
    texts, labels = [], []
    jsonl_path = data_dir / "perception_samples.jsonl"
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
            print("[Lifers-Perception v2] 数据不足，使用内置样本")
        return _train_perception_fallback(n_epochs, lr, save_path, verbose)

    if verbose:
        print(f"[Lifers-Perception v2] 数据: {len(texts):,} 样本")

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

    model = PerceptionClassifier(n_features=n_feat)
    batch_size = 128
    best_acc = 0.0
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
            _save_perception_model(model, vocab, save_path)

        if (epoch + 1) % 15 == 0 and verbose:
            print(f"[Lifers-Perception v2] epoch {epoch + 1}/{n_epochs}  "
                  f"loss={total_loss/(n_train//batch_size+1):.4f}  val_acc={val_acc:.3f}")

        if val_acc > 0.97 and epoch > 10:
            if verbose:
                print(f"[Lifers-Perception v2] 早停 epoch {epoch+1} val_acc={val_acc:.3f}")
            break

    _save_perception_model(model, vocab, save_path)
    if verbose:
        print(f"[Lifers-Perception v2] 完成 best_val_acc={best_acc:.3f} → {save_path}")
    return model


_PERCEPTION_SAMPLES = [
    # 室内办公 (0)
    ("办公室里键盘声不断有人在敲代码", 0), ("会议室里白板上写满了方案", 0), ("工位上电脑屏幕亮着", 0),
    ("打印机正在输出文件", 0), ("茶水间里有人在休息聊天", 0),
    # 室内家庭 (1)
    ("客厅阳光透过窗帘照进来", 1), ("厨房里飘出饭菜的香味", 1), ("卧室里灯光柔和温暖", 1),
    ("阳台上植物长势很好", 1), ("书房里书架上摆满了书", 1),
    # 室内公共 (2)
    ("商场里人来人往热闹非凡", 2), ("图书馆安静得能听到翻书声", 2), ("咖啡厅里有人在用电脑工作", 2),
    ("医院走廊里护士在巡视", 2), ("超市里顾客在挑选商品", 2),
    # 户外街道 (3)
    ("街道上车辆川流不息", 3), ("路口红绿灯交替变换", 3), ("人行道上行人匆匆走过", 3),
    ("路边商铺的招牌亮着灯", 3), ("公交站台有人在等车", 3),
    # 户外自然 (4)
    ("公园里鸟语花香阳光明媚", 4), ("海边浪花拍打着沙滩", 4), ("森林里树木茂密空气清新", 4),
    ("山间小溪流水潺潺", 4), ("田野上金黄的麦浪随风起伏", 4),
    # 户外运动 (5)
    ("操场上人们正在跑步锻炼", 5), ("球场上比赛进行得很激烈", 5), ("健身房里器械声此起彼伏", 5),
    ("游泳池里水花四溅", 5), ("跑道上运动员在冲刺", 5),
]

def _perception_feature(text: str) -> np.ndarray:
    feats = np.zeros(256, dtype=np.float32)
    for i in range(len(text) - 1):
        idx = hash(text[i:i+2]) % 256
        feats[idx] += 1.0
    norm = np.linalg.norm(feats) + 1e-8
    return feats / norm

def _train_perception_fallback(n_epochs, lr, save_path, verbose):
    """使用内置数据训练感知分类器"""
    samples = list(_PERCEPTION_SAMPLES)
    X = np.array([_perception_feature(t) for t, _ in samples], dtype=np.float32)
    y = np.array([l for _, l in samples], dtype=np.int32)
    model = PerceptionClassifier(n_features=256)
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
            _save_perception_model(model, {}, save_path)
        if (epoch + 1) % 15 == 0 and verbose:
            print(f"[Lifers-Perception v2] fallback epoch {epoch + 1}/{n_epochs} acc={acc:.3f}")
    _save_perception_model(model, {}, save_path)
    if verbose:
        print(f"[Lifers-Perception v2] fallback best_acc={best_acc:.3f}")
    return model


def _save_perception_model(model: PerceptionClassifier, vocab: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "brand": "Lifers Perception Classifier v2",
        "version": 3,
        "n_features": model.n_features,
        "hidden1": model.hidden1,
        "hidden2": model.hidden2,
        "n_classes": model.n_classes,
        "classes": CLASS_NAMES,
        "_vocab": dict(sorted(vocab.items(), key=lambda x: x[1])),
        "W1": model.W1.tolist(), "b1": model.b1.tolist(),
        "W2": model.W2.tolist(), "b2": model.b2.tolist(),
        "W3": model.W3.tolist(), "b3": model.b3.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    epochs = int(os.environ.get("LIFERS_PERCEPTION_EPOCHS", "80"))
    out = ROOT / "weights" / "lifers_perception_classifier.json"
    print("[Lifers-Perception v2] n-gram + 3层MLP + Adam")
    t0 = time.time()
    train_perception_classifier(n_epochs=epochs, save_path=out, verbose=True)
    print(f"[Lifers-Perception v2] 耗时={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
