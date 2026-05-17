"""
Lifers Perception 场景分类器训练 — 情境识别 (室内/室外/办公/家庭/街道)
品牌化权重: weights/lifers_perception_classifier.json
纯numpy MLP + 品牌化场景语料特征
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent

# ═══════════════════════════════════════════════════════════════════════════════
# 品牌化场景训练数据
# ═══════════════════════════════════════════════════════════════════════════════

_SCENE_SAMPLES: List[Tuple[str, int]] = [
    # 室内办公 (0)
    ("办公室内，阳光从窗户斜照进来，有人在工位上敲键盘，绿植叶子微微晃动。整体色调温暖偏黄，光线柔和。", 0),
    ("白色灯光均匀照明，实验台上摆放着各种仪器。一个人穿着白大褂正在操作显微镜，背景中电脑屏幕显示着数据图表。", 0),
    ("办公室环境，键盘敲击声中等节奏，远处有打印机运作的低频嗡嗡声，偶尔有人声交流。整体噪声水平45dB。", 0),
    ("开放式工位区域，多个显示器亮着，白板上写着流程图。有人在站立开会讨论项目进展。", 0),
    ("格子间办公区，电话铃声偶尔响起，有人在视频会议中发言。台灯在桌面投下暖光。", 0),

    # 室内家庭 (1)
    ("厨房内，暖色灯光，水流声从水龙头传来。抽油烟机低频噪声持续。微波炉显示倒计时。", 1),
    ("客厅里，电视低声播放新闻，沙发上放着几本书。窗帘半拉，外面的光线透过缝隙照进来。", 1),
    ("卧室环境安静，床头灯亮着柔光。窗外有微风，窗帘轻轻飘动。", 1),
    ("家庭餐厅，餐桌上摆放着水果盘，椅子整齐排列。墙上挂着一幅风景画。", 1),
    ("浴室，镜子上有水汽，水龙头滴水声有节奏。抽风机持续运作。", 1),

    # 室内咖啡/社交 (2)
    ("咖啡店内，暖色灯光，吧台后的咖啡师正在制作拿铁。蒸汽声和咖啡研磨声交织。顾客坐在角落的沙发上阅读。", 2),
    ("餐厅里，餐具碰撞声此起彼伏，服务员在桌间穿行。厨房传出炒菜声，香气四溢。", 2),
    ("书店角落，安静阅读区，翻书声轻微。背景有轻音乐。有人在书架间安静地浏览。", 2),
    ("图书馆阅览室，安静至极，只有偶尔的翻书声和键盘敲击声。学生们在专注学习。", 2),
    ("会议室，投影仪亮着，参会者围坐在长桌旁。白板上写着会议议程，有人在发言。", 2),

    # 户外街道 (3)
    ("户外街道，傍晚时分，路灯刚亮起。行人在人行道上走动，速度各异。远处车辆驶过，车灯在黄昏中格外明亮。", 3),
    ("城市十字路口，红绿灯交替变化。行人匆匆过马路，机动车有序等待。公交站有人在看手机等待。", 3),
    ("居民区街道，安静整洁。路边停着几辆车，有人在遛狗。偶尔有骑自行车的人经过。", 3),
    ("商业街，店铺橱窗明亮，逛街的人流不断。有人在露天咖啡座休息，街上气氛热闹。", 3),
    ("雨后街道，地面湿润反光。空气清新，行人稀少。远处有汽车驶过溅起水花的声音。", 3),

    # 户外自然 (4)
    ("公园里，绿树成荫，鸟鸣声此起彼伏。有人在长椅上看书，儿童在草坪上玩耍。阳光透过树叶洒下光斑。", 4),
    ("森林小径，树冠遮蔽天空，空气潮湿清新。踩在落叶上沙沙作响。远处有溪流潺潺声。", 4),
    ("海边沙滩，海浪一波波拍打海岸。海风带着咸味，远处有海鸥盘旋。有人在沙滩上散步。", 4),
    ("山间小路，视野开阔，远山层层叠叠。天很蓝，有白云飘过。野花在路边摇曳。", 4),
    ("湖边，水面平静如镜，倒映着对岸的树木。微风拂过，水面泛起细小涟漪。有人在垂钓。", 4),
]

N_SCENES = 6


def _scene_feature(text: str, dim: int = 32) -> np.ndarray:
    """场景文本 → 特征向量 (关键词袋 + 字符级统计)"""
    feat = np.zeros(dim, dtype=np.float32)

    # 关键词匹配
    keywords = {
        "室内": 0, "办公室": 0, "实验": 0, "电脑": 0, "键盘": 0, "白板": 0,
        "厨房": 1, "客厅": 1, "卧室": 1, "浴室": 1, "餐厅": 1, "沙发": 1,
        "咖啡": 2, "书店": 2, "图书": 2, "会议": 2, "吧台": 2,
        "街道": 3, "路口": 3, "行人": 3, "车辆": 3, "商业街": 3, "公交": 3,
        "公园": 4, "森林": 4, "海边": 4, "山间": 4, "湖畔": 4, "沙滩": 4,
        "阳光": 5, "灯光": 5, "明亮": 5, "昏暗": 5, "暖色": 5, "柔和": 6,
        "安静": 7, "热闹": 7, "微风": 8, "鸟鸣": 8, "水声": 9, "车流": 10,
    }
    text_lower = text.lower()
    for kw, slot in keywords.items():
        if kw in text_lower:
            feat[slot] += 1.0

    # 字符统计特征
    feat[12] = len(text) / 200.0  # 文本长度归一化
    feat[13] = text.count("。") / 10.0  # 句子数
    feat[14] = text.count("，") / 20.0  # 逗号数
    feat[15] = sum(1 for c in text if '一' <= c <= '鿿') / len(text)  # 中文比例

    feat = feat / (np.linalg.norm(feat) + 1e-8)
    return feat


# ═══════════════════════════════════════════════════════════════════════════════
# Lifers 场景分类器
# ═══════════════════════════════════════════════════════════════════════════════

class LifersPerceptionClassifier:
    """MLP场景分类器 — 特征 → 场景类别"""

    def __init__(self, input_dim: int = 32, hidden_dim: int = 32, n_classes: int = N_SCENES):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        rng = np.random.RandomState(42)
        self.W1 = rng.randn(input_dim, hidden_dim).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.randn(hidden_dim, n_classes).astype(np.float32) * 0.1
        self.b2 = np.zeros(n_classes, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0, x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return logits

    def predict(self, x: np.ndarray) -> Tuple[int, np.ndarray]:
        logits = self.forward(x)
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        return int(np.argmax(probs)), probs


def train_perception_classifier(
    n_epochs: int = 300,
    lr: float = 1e-2,
    save_path: Optional[Path] = None,
    verbose: bool = True,
) -> LifersPerceptionClassifier:
    """训练场景分类器"""

    if save_path is None:
        save_path = ROOT / "weights" / "lifers_perception_classifier.json"

    X = np.array([_scene_feature(t) for t, _ in _SCENE_SAMPLES], dtype=np.float32)
    y = np.array([l for _, l in _SCENE_SAMPLES], dtype=np.int32)

    model = LifersPerceptionClassifier()
    rng = np.random.RandomState(123)
    best_acc = 0.0

    for epoch in range(n_epochs):
        indices = list(range(len(X)))
        rng.shuffle(indices)
        total_loss = 0.0
        correct = 0

        for idx in indices:
            x_i, y_i = X[idx], y[idx]
            logits = model.forward(x_i)
            probs = np.exp(logits - np.max(logits))
            probs = probs / probs.sum()

            loss = -np.log(probs[y_i] + 1e-8)
            total_loss += loss

            if np.argmax(probs) == y_i:
                correct += 1

            # 梯度
            dlogits = probs.copy()
            dlogits[y_i] -= 1.0

            h_pre = x_i @ model.W1 + model.b1
            h = np.maximum(0, h_pre)

            # W2, b2
            model.W2 -= lr * np.outer(h, dlogits)
            model.b2 -= lr * dlogits

            # W1, b1
            dh = dlogits @ model.W2.T
            dh[h_pre <= 0] = 0
            model.W1 -= lr * np.outer(x_i, dh)
            model.b1 -= lr * dh

        acc = correct / len(indices)
        if acc > best_acc:
            best_acc = acc
            _save_perception_model(model, save_path)

        if (epoch + 1) % 60 == 0 and verbose:
            print(f"[Lifers-Perception] epoch {epoch + 1}/{n_epochs}  "
                  f"loss={total_loss/len(indices):.4f}  acc={acc:.3f}")

    _save_perception_model(model, save_path)
    if verbose:
        print(f"[Lifers-Perception] 训练完成 best_acc={best_acc:.3f} → {save_path}")
    return model


def _save_perception_model(model: LifersPerceptionClassifier, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "brand": "Lifers Perception Classifier",
        "version": 1,
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "n_classes": model.n_classes,
        "classes": ["室内办公", "室内家庭", "室内社交", "户外街道", "户外自然"],
        "W1": model.W1.tolist(),
        "b1": model.b1.tolist(),
        "W2": model.W2.tolist(),
        "b2": model.b2.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_perception_model(path: Path) -> LifersPerceptionClassifier:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    model = LifersPerceptionClassifier(
        input_dim=data["input_dim"],
        hidden_dim=data["hidden_dim"],
        n_classes=data["n_classes"],
    )
    model.W1 = np.array(data["W1"], dtype=np.float32)
    model.b1 = np.array(data["b1"], dtype=np.float32)
    model.W2 = np.array(data["W2"], dtype=np.float32)
    model.b2 = np.array(data["b2"], dtype=np.float32)
    return model


def main():
    epochs = int(os.environ.get("LIFERS_PERCEPTION_EPOCHS", "300"))
    out = ROOT / "weights" / "lifers_perception_classifier.json"

    print(f"[Lifers-Perception] 品牌化场景分类器训练 epochs={epochs}")
    t0 = time.time()
    train_perception_classifier(n_epochs=epochs, save_path=out, verbose=True)
    print(f"[Lifers-Perception] 完成 耗时={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
