"""
Lifers Knowledge Graph 训练 — 实体嵌入学习 + 关系预测
品牌化权重: weights/lifers_kg_embeddings.json
翻译距离模型 (TransE-style) + 对比学习
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
# Lifers KG 训练数据 — 内置知识三元组
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_kg_triples() -> List[Tuple[str, str, str]]:
    """生成品牌化知识图谱三元组训练数据"""
    triples = [
        # 人工智能概念
        ("machine_learning", "子领域", "artificial_intelligence"),
        ("deep_learning", "子领域", "machine_learning"),
        ("neural_network", "实现方式", "deep_learning"),
        ("transformer", "属于", "neural_network"),
        ("attention", "核心机制", "transformer"),
        ("lstm", "属于", "neural_network"),
        ("cnn", "属于", "neural_network"),
        ("reinforcement_learning", "子领域", "machine_learning"),
        ("supervised_learning", "子领域", "machine_learning"),
        ("unsupervised_learning", "子领域", "machine_learning"),
        ("nlp", "应用领域", "deep_learning"),
        ("computer_vision", "应用领域", "deep_learning"),
        ("speech_recognition", "应用领域", "deep_learning"),
        ("embedding", "表示方法", "neural_network"),
        ("tokenization", "预处理", "nlp"),
        ("backpropagation", "训练方法", "neural_network"),
        ("gradient_descent", "优化算法", "backpropagation"),
        ("adam", "属于", "gradient_descent"),
        ("sgd", "属于", "gradient_descent"),
        ("overfitting", "问题", "machine_learning"),
        ("regularization", "解决方案", "overfitting"),
        ("dropout", "属于", "regularization"),
        ("batch_norm", "属于", "regularization"),
        # 计算机科学
        ("algorithm", "基础概念", "computer_science"),
        ("data_structure", "基础概念", "computer_science"),
        ("array", "属于", "data_structure"),
        ("linked_list", "属于", "data_structure"),
        ("hash_table", "属于", "data_structure"),
        ("binary_tree", "属于", "data_structure"),
        ("graph", "属于", "data_structure"),
        ("sorting", "属于", "algorithm"),
        ("searching", "属于", "algorithm"),
        ("dynamic_programming", "属于", "algorithm"),
        ("greedy", "属于", "algorithm"),
        ("divide_conquer", "属于", "algorithm"),
        ("bfs", "属于", "searching"),
        ("dfs", "属于", "searching"),
        ("dijkstra", "属于", "graph"),
        # 编程语言
        ("python", "属于", "programming_language"),
        ("javascript", "属于", "programming_language"),
        ("rust", "属于", "programming_language"),
        ("c++", "属于", "programming_language"),
        ("java", "属于", "programming_language"),
        ("compiler", "工具", "programming_language"),
        ("interpreter", "工具", "programming_language"),
        ("type_system", "特性", "programming_language"),
        ("garbage_collection", "特性", "programming_language"),
        # 数学
        ("calculus", "分支", "mathematics"),
        ("linear_algebra", "分支", "mathematics"),
        ("probability", "分支", "mathematics"),
        ("statistics", "分支", "mathematics"),
        ("optimization", "分支", "mathematics"),
        ("matrix", "概念", "linear_algebra"),
        ("vector", "概念", "linear_algebra"),
        ("eigenvalue", "概念", "linear_algebra"),
        ("derivative", "概念", "calculus"),
        ("integral", "概念", "calculus"),
        ("bayes_theorem", "定理", "probability"),
        ("central_limit_theorem", "定理", "statistics"),
        # 物理学
        ("quantum_mechanics", "分支", "physics"),
        ("relativity", "分支", "physics"),
        ("thermodynamics", "分支", "physics"),
        ("electromagnetism", "分支", "physics"),
        ("newton_mechanics", "分支", "physics"),
        ("photon", "粒子", "quantum_mechanics"),
        ("electron", "粒子", "physics"),
        ("quark", "粒子", "physics"),
        ("entropy", "概念", "thermodynamics"),
        ("energy", "概念", "physics"),
        ("momentum", "概念", "physics"),
        ("gravity", "力", "physics"),
        ("electromagnetic_force", "力", "physics"),
        # 哲学
        ("epistemology", "分支", "philosophy"),
        ("ethics", "分支", "philosophy"),
        ("metaphysics", "分支", "philosophy"),
        ("logic", "工具", "philosophy"),
        ("socrates", "人物", "philosophy"),
        ("plato", "人物", "philosophy"),
        ("aristotle", "人物", "philosophy"),
        ("kant", "人物", "philosophy"),
        ("nietzsche", "人物", "philosophy"),
        ("existentialism", "学派", "philosophy"),
        ("stoicism", "学派", "philosophy"),
        ("utilitarianism", "理论", "ethics"),
        ("categorical_imperative", "概念", "ethics"),
        # 生物学
        ("dna", "分子", "genetics"),
        ("rna", "分子", "genetics"),
        ("protein", "分子", "biology"),
        ("cell", "基本单位", "biology"),
        ("mitochondria", "细胞器", "cell"),
        ("nucleus", "细胞器", "cell"),
        ("enzyme", "属于", "protein"),
        ("evolution", "理论", "biology"),
        ("natural_selection", "机制", "evolution"),
        ("mutation", "机制", "evolution"),
        ("ecosystem", "概念", "ecology"),
        ("biodiversity", "概念", "ecology"),
        # 社交关系 (为Social支柱提供知识基础)
        ("trust", "基础", "social_relationship"),
        ("communication", "基础", "social_relationship"),
        ("empathy", "基础", "social_relationship"),
        ("respect", "基础", "social_relationship"),
        ("friendship", "类型", "social_relationship"),
        ("family_bond", "类型", "social_relationship"),
        ("professional_relationship", "类型", "social_relationship"),
        ("conflict_resolution", "技能", "social_relationship"),
        ("active_listening", "技能", "communication"),
        ("emotional_intelligence", "能力", "social_relationship"),
        # 机器人学
        ("sensor", "组件", "robot"),
        ("actuator", "组件", "robot"),
        ("controller", "组件", "robot"),
        ("kinematics", "理论", "robot"),
        ("dynamics", "理论", "robot"),
        ("path_planning", "算法", "robot"),
        ("slam", "算法", "robot"),
        ("computer_vision", "感知", "robot"),
        ("lidar", "属于", "sensor"),
        ("camera", "属于", "sensor"),
        ("motor", "属于", "actuator"),
        ("servo", "属于", "actuator"),
    ]
    return triples


# ═══════════════════════════════════════════════════════════════════════════════
# Lifers KG Embedding Model (TransE-style)
# ═══════════════════════════════════════════════════════════════════════════════

class LifersKGEmbedding:
    """Lifers 知识图谱嵌入模型 — 实体和关系嵌入学习"""

    def __init__(self, n_entities: int, n_relations: int, dim: int = 128):
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.dim = dim
        rng = np.random.RandomState(42)
        # 实体嵌入 (归一化)
        self.entity_emb = rng.randn(n_entities, dim).astype(np.float32) * 0.01
        self.entity_emb = self.entity_emb / (np.linalg.norm(self.entity_emb, axis=1, keepdims=True) + 1e-8)
        # 关系嵌入
        self.rel_emb = rng.randn(n_relations, dim).astype(np.float32) * 0.01

    def score(self, head: int, rel: int, tail: int) -> float:
        """TransE评分: ||h + r - t||"""
        h = self.entity_emb[head]
        r = self.rel_emb[rel]
        t = self.entity_emb[tail]
        return float(np.linalg.norm(h + r - t))

    def predict_tail(self, head: int, rel: int, k: int = 10) -> List[Tuple[int, float]]:
        """预测尾实体"""
        h = self.entity_emb[head]
        r = self.rel_emb[rel]
        scores = np.linalg.norm(h + r - self.entity_emb, axis=1)
        # 排除自身
        scores[head] = float("inf")
        top_k = np.argsort(scores)[:k]
        return [(int(i), float(scores[i])) for i in top_k]

    def get_params(self) -> Dict[str, np.ndarray]:
        return {"entity_emb": self.entity_emb, "rel_emb": self.rel_emb}

    def set_params(self, params: Dict[str, np.ndarray]):
        for k, v in params.items():
            setattr(self, k, v.copy())


# ═══════════════════════════════════════════════════════════════════════════════
# Lifers KG Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class LifersKGTrainer:
    """Lifers 品牌化知识图谱训练器 — 翻译距离 + 负采样"""

    def __init__(self, dim: int = 128, lr: float = 1e-3, margin: float = 1.0):
        self.dim = dim
        self.lr = lr
        self.margin = margin
        self.model: Optional[LifersKGEmbedding] = None
        self._entity2id: Dict[str, int] = {}
        self._rel2id: Dict[str, int] = {}
        self._id2entity: Dict[int, str] = {}
        self._id2rel: Dict[int, str] = {}
        self._train_triples: List[Tuple[int, int, int]] = []
        self._loss_history: List[float] = []

    def prepare_data(self, triples: List[Tuple[str, str, str]]):
        """准备训练数据: 构建词表"""
        entities = set()
        relations = set()
        for h, r, t in triples:
            entities.add(h)
            entities.add(t)
            relations.add(r)

        self._entity2id = {e: i for i, e in enumerate(sorted(entities))}
        self._rel2id = {r: i for i, r in enumerate(sorted(relations))}
        self._id2entity = {i: e for e, i in self._entity2id.items()}
        self._id2rel = {i: r for r, i in self._rel2id.items()}

        self._train_triples = [
            (self._entity2id[h], self._rel2id[r], self._entity2id[t])
            for h, r, t in triples
        ]

        self.model = LifersKGEmbedding(
            n_entities=len(self._entity2id),
            n_relations=len(self._rel2id),
            dim=self.dim,
        )

    def train_epoch(self, batch_size: int = 64) -> Dict[str, float]:
        if self.model is None:
            return {"loss": 0.0, "mean_rank": 0.0}

        rng = np.random.RandomState()
        indices = list(range(len(self._train_triples)))
        rng.shuffle(indices)

        total_loss = 0.0
        total_rank = 0
        n_entities = self.model.n_entities
        n_batches = 0

        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            batch_loss = 0.0

            for idx in batch:
                h, r, t = self._train_triples[idx]
                h_emb = self.model.entity_emb[h]
                r_emb = self.model.rel_emb[r]
                t_emb = self.model.entity_emb[t]

                pos_score = np.linalg.norm(h_emb + r_emb - t_emb)

                # 硬负采样: 从多个随机负样本中选距离最小的
                n_neg = 8
                best_neg_score = float("inf")
                best_neg_t = -1
                for _ in range(n_neg):
                    neg_t = rng.randint(0, n_entities)
                    while neg_t == t:
                        neg_t = rng.randint(0, n_entities)
                    neg_score = np.linalg.norm(h_emb + r_emb - self.model.entity_emb[neg_t])
                    if neg_score < best_neg_score:
                        best_neg_score = neg_score
                        best_neg_t = neg_t

                neg_t_emb = self.model.entity_emb[best_neg_t]
                neg_score = best_neg_score

                loss = max(0.0, self.margin + pos_score - neg_score)
                batch_loss += loss

                # Mean Rank
                all_scores = np.linalg.norm(h_emb + r_emb - self.model.entity_emb, axis=1)
                rank = int(np.sum(all_scores < pos_score)) + 1
                total_rank += rank

                # 梯度更新
                if loss > 0:
                    d_pos = (h_emb + r_emb - t_emb) / (pos_score + 1e-8)
                    d_neg = (h_emb + r_emb - neg_t_emb) / (neg_score + 1e-8)

                    self.model.entity_emb[h] -= self.lr * (d_pos - d_neg)
                    self.model.rel_emb[r] -= self.lr * (d_pos - d_neg)
                    self.model.entity_emb[t] += self.lr * d_pos
                    self.model.entity_emb[best_neg_t] -= self.lr * d_neg

            n_batches += 1

        # 归一化嵌入
        self.model.entity_emb = self.model.entity_emb / (
            np.linalg.norm(self.model.entity_emb, axis=1, keepdims=True) + 1e-8
        )

        n = len(indices)
        avg_loss = total_loss / max(n, 1)
        mean_rank = total_rank / max(n, 1)
        self._loss_history.append(avg_loss)
        return {"loss": avg_loss, "mean_rank": mean_rank}


def train_lifers_kg(
    n_epochs: int = 100,
    dim: int = 128,
    save_path: Optional[Path] = None,
    verbose: bool = True,
) -> LifersKGTrainer:
    """Lifers KG 品牌化训练"""
    if save_path is None:
        save_path = ROOT / "weights" / "lifers_kg_embeddings.json"

    trainer = LifersKGTrainer(dim=dim)
    triples = _generate_kg_triples()
    trainer.prepare_data(triples)

    if verbose:
        print(f"[Lifers-KG] 训练数据: {len(triples)} 三元组  "
              f"entities={len(trainer._entity2id)}  relations={len(trainer._rel2id)}")

    best_loss = float("inf")
    for epoch in range(n_epochs):
        metrics = trainer.train_epoch()
        if (epoch + 1) % 20 == 0 and verbose:
            print(f"[Lifers-KG] epoch {epoch + 1}/{n_epochs}  "
                  f"loss={metrics['loss']:.4f}  mean_rank={metrics['mean_rank']:.1f}")

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            _save_kg_model(trainer, save_path)

    _save_kg_model(trainer, save_path)
    if verbose:
        print(f"[Lifers-KG] 训练完成 best_loss={best_loss:.4f} → {save_path}")
    return trainer


def _save_kg_model(trainer: LifersKGTrainer, path: Path):
    if trainer.model is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "brand": "Lifers Knowledge Graph Embeddings",
        "version": 1,
        "dim": trainer.dim,
        "n_entities": trainer.model.n_entities,
        "n_relations": trainer.model.n_relations,
        "entity2id": trainer._entity2id,
        "rel2id": trainer._rel2id,
        "id2entity": {str(k): v for k, v in trainer._id2entity.items()},
        "id2rel": {str(k): v for k, v in trainer._id2rel.items()},
        "entity_emb": trainer.model.entity_emb.tolist(),
        "rel_emb": trainer.model.rel_emb.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_kg_model(path: Path) -> LifersKGTrainer:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    trainer = LifersKGTrainer(dim=data["dim"])
    trainer._entity2id = data["entity2id"]
    trainer._rel2id = data["rel2id"]
    trainer._id2entity = {int(k): v for k, v in data["id2entity"].items()}
    trainer._id2rel = {int(k): v for k, v in data["id2rel"].items()}
    trainer.model = LifersKGEmbedding(
        n_entities=data["n_entities"],
        n_relations=data["n_relations"],
        dim=data["dim"],
    )
    trainer.model.entity_emb = np.array(data["entity_emb"], dtype=np.float32)
    trainer.model.rel_emb = np.array(data["rel_emb"], dtype=np.float32)
    return trainer


def main():
    epochs = int(os.environ.get("LIFERS_KG_EPOCHS", "100"))
    dim = int(os.environ.get("LIFERS_KG_DIM", "128"))
    out = ROOT / "weights" / "lifers_kg_embeddings.json"

    print(f"[Lifers-KG] 品牌化KG训练 epochs={epochs} dim={dim}")
    t0 = time.time()
    train_lifers_kg(n_epochs=epochs, dim=dim, save_path=out, verbose=True)
    print(f"[Lifers-KG] 完成 耗时={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
