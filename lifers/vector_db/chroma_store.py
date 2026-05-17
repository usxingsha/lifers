from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base import AbstractVectorStore, VectorSearchResult


class ChromaStore(AbstractVectorStore):
    """Chroma-backed vector store.  Uses persistent client if persist_path is given."""

    def __init__(self, collection_name: str = "lifers_memory", persist_path: Optional[str] = None) -> None:
        self._collection_name = collection_name
        self._persist_path = persist_path
        self._client = None
        self._collection = None

    def _lazy_connect(self):
        if self._collection is not None:
            return
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb not installed. Run: pip install chromadb"
            )
        if self._persist_path:
            self._client = chromadb.PersistentClient(path=self._persist_path)
        else:
            self._client = chromadb.Client()
        try:
            self._collection = self._client.get_collection(self._collection_name)
        except Exception:
            self._collection = self._client.create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def add(self, ids: List[Any], vectors: np.ndarray, metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        self._lazy_connect()
        str_ids = [str(uid) for uid in ids]
        vecs = vectors.astype(np.float32).tolist()
        metas = None
        if metadatas:
            metas = [{k: _safe_meta_val(v) for k, v in m.items()} for m in metadatas]
        self._collection.add(ids=str_ids, embeddings=vecs, metadatas=metas)

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[VectorSearchResult]:
        self._lazy_connect()
        n = self._collection.count()
        if n == 0:
            return []
        q = query_vector.astype(np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        result = self._collection.query(query_embeddings=q.tolist(), n_results=min(k, n))
        results: List[VectorSearchResult] = []
        ids_list = result.get("ids", [[]])[0]
        dists = result.get("distances", [[]])[0]
        metas = result.get("metadatas", [[]])[0] or []
        for i, uid in enumerate(ids_list):
            score = 1.0 - float(dists[i]) if dists and i < len(dists) else 0.0
            meta = metas[i] if metas and i < len(metas) else {}
            results.append(VectorSearchResult(id=_try_aton(uid), score=score, metadata=meta or {}))
        return results

    def delete(self, ids: List[Any]) -> None:
        self._lazy_connect()
        if not ids:
            return
        self._collection.delete(ids=[str(uid) for uid in ids])

    def count(self) -> int:
        self._lazy_connect()
        return self._collection.count()

    def clear(self) -> None:
        self._lazy_connect()
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = None
        if self._client:
            self._collection = self._client.create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )


def _safe_meta_val(v: Any) -> str:
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _try_aton(s: str):
    try:
        return int(s)
    except (ValueError, TypeError):
        return s
