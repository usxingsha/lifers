"""
Lifers KnowledgeGraph — 自给自足实体关系图谱
基于 NumPy 的嵌入式知识推理，无需外部依赖
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class Entity:
    id: str
    type: str  # person, object, concept, place, event, skill
    name: str
    props: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class Relation:
    src: str
    dst: str
    rel_type: str  # has, is_a, part_of, causes, uses, located_in, knows, before, after
    weight: float = 1.0
    props: Dict[str, Any] = field(default_factory=dict)


# ── Ontology ──────────────────────────────────────────────────────────────────

BUILTIN_ENTITY_TYPES = [
    "person", "robot", "object", "concept", "place", "event",
    "skill", "tool", "goal", "constraint", "memory",
]

BUILTIN_RELATION_TYPES = [
    "has", "is_a", "part_of", "causes", "uses", "located_in",
    "knows", "before", "after", "depends_on", "conflicts_with",
    "achieves", "requires", "produces",
]


# ── Graph Engine ──────────────────────────────────────────────────────────────

class KnowledgeGraph:
    """Entity-relation graph with numpy-powered similarity search."""

    def __init__(self, embed_dim: int = 128) -> None:
        self.dim = embed_dim
        self._entities: Dict[str, Entity] = {}
        self._relations: List[Relation] = []
        self._adj_out: Dict[str, List[Tuple[str, Relation]]] = defaultdict(list)
        self._adj_in: Dict[str, List[Tuple[str, Relation]]] = defaultdict(list)
        self._type_idx: Dict[str, List[str]] = defaultdict(list)
        self._rng = np.random.RandomState(42)

    # ── CRUD ──────────────────────────────────────────────────────────────

    def add_entity(self, entity: Entity) -> str:
        if entity.embedding is None:
            entity.embedding = self._random_embedding()
        self._entities[entity.id] = entity
        self._type_idx[entity.type].append(entity.id)
        return entity.id

    def add_relation(self, rel: Relation) -> None:
        if rel.src not in self._entities:
            self.add_entity(Entity(id=rel.src, type="concept", name=rel.src))
        if rel.dst not in self._entities:
            self.add_entity(Entity(id=rel.dst, type="concept", name=rel.dst))
        self._relations.append(rel)
        self._adj_out[rel.src].append((rel.dst, rel))
        self._adj_in[rel.dst].append((rel.src, rel))

    def remove_entity(self, entity_id: str) -> None:
        self._entities.pop(entity_id, None)
        self._relations = [r for r in self._relations if r.src != entity_id and r.dst != entity_id]
        self._adj_out.pop(entity_id, None)
        self._adj_in.pop(entity_id, None)
        for t, ids in self._type_idx.items():
            self._type_idx[t] = [i for i in ids if i != entity_id]

    # ── Query ─────────────────────────────────────────────────────────────

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(entity_id)

    def neighbors(self, entity_id: str, rel_type: Optional[str] = None, direction: str = "out") -> List[Tuple[str, Relation]]:
        adj = self._adj_out if direction == "out" else self._adj_in
        edges = adj.get(entity_id, [])
        if rel_type:
            edges = [(d, r) for d, r in edges if r.rel_type == rel_type]
        return edges

    def query(self, src: str, rel_type: Optional[str] = None) -> List[str]:
        """One-hop: ? --rel_type--> dst, starting from src."""
        return [dst for dst, r in self.neighbors(src, rel_type)]

    def path(self, src: str, dst: str, max_hops: int = 4) -> Optional[List[Tuple[str, str, str]]]:
        """BFS shortest path."""
        if src == dst:
            return []
        visited = {src}
        queue = [(src, [])]
        while queue:
            node, path_so_far = queue.pop(0)
            if len(path_so_far) >= max_hops:
                continue
            for nxt, rel in self.neighbors(node):
                step = (node, rel.rel_type, nxt)
                if nxt == dst:
                    return path_so_far + [step]
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path_so_far + [step]))
        return None

    # ── Semantic Search ───────────────────────────────────────────────────

    def search_similar(self, query_embedding: np.ndarray, k: int = 10, entity_type: Optional[str] = None) -> List[Tuple[Entity, float]]:
        candidates = list(self._entities.values())
        if entity_type:
            candidates = [e for e in candidates if e.type == entity_type]
        if not candidates:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)
        scores = []
        for e in candidates:
            if e.embedding is None:
                continue
            v = e.embedding / (np.linalg.norm(e.embedding) + 1e-8)
            scores.append((e, float(q @ v)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def search_by_name(self, name: str, top_k: int = 5) -> List[Entity]:
        name_lower = name.lower()
        scored = []
        for e in self._entities.values():
            if name_lower in e.name.lower():
                scored.append((e, len(e.name)))  # shorter name = better match
        scored.sort(key=lambda x: x[1])
        return [e for e, _ in scored[:top_k]]

    # ── Reasoning ─────────────────────────────────────────────────────────

    def infer(self, entity_id: str, max_hops: int = 3) -> List[Tuple[str, str, str, float]]:
        """Spread activation inference from a starting entity."""
        activated: Dict[str, float] = {entity_id: 1.0}
        results: List[Tuple[str, str, str, float]] = []
        decay = 0.6
        for hop in range(max_hops):
            new_activated: Dict[str, float] = defaultdict(float)
            for src, strength in activated.items():
                for dst, rel in self.neighbors(src):
                    s = strength * decay * rel.weight
                    new_activated[dst] = max(new_activated[dst], s)
                    if s > 0.1:
                        results.append((src, rel.rel_type, dst, s))
            activated = new_activated
        results.sort(key=lambda x: x[3], reverse=True)
        return results

    def analogy(self, a: str, b: str, c: str) -> List[Tuple[str, float]]:
        """a:b :: c:? via embedding arithmetic: b - a + c."""
        ea = self._entities.get(a)
        eb = self._entities.get(b)
        ec = self._entities.get(c)
        if not all([ea, eb, ec]) or any(e.embedding is None for e in [ea, eb, ec]):
            return []
        target = eb.embedding - ea.embedding + ec.embedding
        return [(e.id, float(s)) for e, s in self.search_similar(target, k=5) if e.id not in (a, b, c)]

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "dim": self.dim,
            "entities": {
                eid: {
                    "id": e.id, "type": e.type, "name": e.name,
                    "props": e.props,
                    "embedding": e.embedding.tolist() if e.embedding is not None else None,
                }
                for eid, e in self._entities.items()
            },
            "relations": [
                {"src": r.src, "dst": r.dst, "rel_type": r.rel_type, "weight": r.weight, "props": r.props}
                for r in self._relations
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "KnowledgeGraph":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        kg = cls(embed_dim=data["dim"])
        for eid, e_data in data["entities"].items():
            emb = np.array(e_data["embedding"], dtype=np.float32) if e_data["embedding"] else None
            kg._entities[eid] = Entity(
                id=e_data["id"], type=e_data["type"], name=e_data["name"],
                props=e_data.get("props", {}), embedding=emb,
            )
            kg._type_idx[e_data["type"]].append(eid)
        for r_data in data["relations"]:
            rel = Relation(**r_data)
            kg._relations.append(rel)
            kg._adj_out[rel.src].append((rel.dst, rel))
            kg._adj_in[rel.dst].append((rel.src, rel))
        return kg

    def _random_embedding(self) -> np.ndarray:
        return (self._rng.randn(self.dim) * 0.1).astype(np.float32)

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    @property
    def relation_count(self) -> int:
        return len(self._relations)


# ── World Ontology Builder ────────────────────────────────────────────────────

def build_world_ontology(kg: Optional[KnowledgeGraph] = None) -> KnowledgeGraph:
    """Seed a KnowledgeGraph with common-sense world ontology."""
    if kg is None:
        kg = KnowledgeGraph()

    # Core concepts
    concepts = [
        ("robot", "机器人"), ("human", "人类"), ("ai", "人工智能"),
        ("perception", "感知"), ("action", "行动"), ("reasoning", "推理"),
        ("memory", "记忆"), ("learning", "学习"), ("goal", "目标"),
        ("tool", "工具"), ("environment", "环境"), ("self", "自我"),
        ("time", "时间"), ("space", "空间"), ("knowledge", "知识"),
        ("language", "语言"), ("emotion", "情感"), ("plan", "计划"),
        ("sensor", "传感器"), ("actuator", "执行器"), ("body", "身体"),
        ("mind", "心智"), ("communication", "沟通"), ("safety", "安全"),
    ]
    for cid, cname in concepts:
        kg.add_entity(Entity(id=cid, type="concept", name=cname))

    # Relations forming basic ontology
    relations = [
        ("robot", "is_a", "ai"), ("ai", "part_of", "mind"),
        ("mind", "has", "perception"), ("mind", "has", "reasoning"),
        ("mind", "has", "memory"), ("mind", "has", "learning"),
        ("body", "has", "sensor"), ("body", "has", "actuator"),
        ("robot", "has", "body"), ("robot", "has", "mind"),
        ("human", "has", "mind"), ("human", "has", "body"),
        ("perception", "uses", "sensor"), ("action", "uses", "actuator"),
        ("reasoning", "produces", "plan"), ("plan", "requires", "goal"),
        ("plan", "requires", "knowledge"), ("learning", "requires", "memory"),
        ("communication", "uses", "language"), ("language", "is_a", "tool"),
        ("safety", "depends_on", "reasoning"), ("safety", "depends_on", "perception"),
        ("self", "has", "goal"), ("self", "located_in", "environment"),
        ("environment", "has", "space"), ("environment", "has", "time"),
        ("action", "causes", "event"), ("event", "located_in", "time"),
        ("event", "located_in", "space"),
    ]
    for src, rel, dst in relations:
        kg.add_relation(Relation(src=src, dst=dst, rel_type=rel))

    return kg
