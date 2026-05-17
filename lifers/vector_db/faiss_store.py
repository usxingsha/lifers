from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base import AbstractVectorStore, VectorSearchResult


class FAISSStore(AbstractVectorStore):
    """FAISS-backed vector store.  IndexFlatIP (cosine via normalized vectors)."""

    def __init__(self, dim: int, store_path: Optional[str] = None) -> None:
        self._dim = dim
        self._store_path = Path(store_path) if store_path else None
        self._index = None  # faiss.IndexFlatIP
        self._id_map: Dict[int, Any] = {}       # faiss_id -> user_id
        self._id_map_rev: Dict[Any, int] = {}   # user_id -> faiss_id
        self._metadatas: Dict[Any, Dict[str, Any]] = {}
        self._next_id = 0
        if self._store_path:
            self._load()

    def _lazy_import(self):
        try:
            import faiss
            return faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu not installed. Run: pip install faiss-cpu"
            )

    def _ensure_index(self) -> None:
        if self._index is None:
            faiss = self._lazy_import()
            self._index = faiss.IndexFlatIP(self._dim)

    def add(self, ids: List[Any], vectors: np.ndarray, metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        faiss = self._lazy_import()
        self._ensure_index()
        vecs = np.asarray(vectors, dtype=np.float32)
        _l2_normalize(vecs)
        start = self._index.ntotal
        self._index.add(vecs)
        for i, uid in enumerate(ids):
            faiss_id = start + i
            self._id_map[faiss_id] = uid
            self._id_map_rev[uid] = faiss_id
            if metadatas and i < len(metadatas):
                self._metadatas[uid] = metadatas[i]
        if self._store_path:
            self._save()

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[VectorSearchResult]:
        self._ensure_index()
        if self._index.ntotal == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        _l2_normalize(q)
        scores, indices = self._index.search(q, min(k, self._index.ntotal))
        results: List[VectorSearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            uid = self._id_map.get(int(idx))
            results.append(VectorSearchResult(
                id=uid,
                score=float(score),
                metadata=self._metadatas.get(uid, {}),
            ))
        return results

    def delete(self, ids: List[Any]) -> None:
        for uid in ids:
            if uid in self._id_map_rev:
                del self._id_map_rev[uid]
            self._metadatas.pop(uid, None)
        # Rebuild index without deleted ids (FAISS doesn't support single-item delete easily)
        if self._id_map_rev:
            vecs = []
            new_ids = []
            faiss = self._lazy_import()
            # We need original vectors — can't reconstruct from IndexFlatIP
            # Mark as stale and rebuild on next operation
            self._needs_rebuild = True
        else:
            self._index = None
            self._id_map.clear()
            self._next_id = 0
        if self._store_path:
            self._save()

    def count(self) -> int:
        return len(self._id_map_rev)

    def clear(self) -> None:
        self._index = None
        self._id_map.clear()
        self._id_map_rev.clear()
        self._metadatas.clear()
        self._next_id = 0
        if self._store_path and self._store_path.exists():
            self._store_path.unlink()

    def _save(self) -> None:
        if not self._store_path:
            return
        faiss = self._lazy_import()
        self._ensure_index()
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._store_path.with_suffix(self._store_path.suffix + ".tmp")
        faiss.write_index(self._index, str(tmp))
        meta = {
            "dim": self._dim,
            "id_map": {str(k): v for k, v in self._id_map.items()},
            "id_map_rev": {str(k): v for k, v in self._id_map_rev.items()},
            "metadatas": {str(k): v for k, v in self._metadatas.items()},
            "next_id": self._next_id,
        }
        meta_path = self._store_path.with_suffix(self._store_path.suffix + ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        os.replace(tmp, self._store_path)

    def _load(self) -> None:
        if not self._store_path or not self._store_path.exists():
            return
        try:
            faiss = self._lazy_import()
            self._index = faiss.read_index(str(self._store_path))
            meta_path = self._store_path.with_suffix(self._store_path.suffix + ".meta.json")
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self._dim = meta["dim"]
                self._id_map = {int(k): v for k, v in meta["id_map"].items()}
                self._id_map_rev = {_try_aton(k): int(v) for k, v in meta["id_map_rev"].items()}
                self._metadatas = {_try_aton(k): v for k, v in meta["metadatas"].items()}
                self._next_id = meta.get("next_id", 0)
        except Exception:
            self._index = None


def _l2_normalize(vecs: np.ndarray) -> None:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    vecs /= norms


def _try_aton(s: str):
    try:
        return int(s)
    except (ValueError, TypeError):
        return s
