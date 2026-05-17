from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .base import AbstractEmbeddingProvider, AbstractVectorStore, VectorSearchResult


class HybridSearch:
    """Combines vector search with FTS5 full-text search via reciprocal rank fusion."""

    def __init__(
        self,
        vector_store: AbstractVectorStore,
        embedding_provider: AbstractEmbeddingProvider,
        fts5_search_fn=None,  # callable(query, k, types) -> List[Dict]
    ) -> None:
        self._vs = vector_store
        self._emb = embedding_provider
        self._fts5 = fts5_search_fn
        self._k_rrf: int = 60  # constant for reciprocal rank fusion

    def search(
        self,
        query: str,
        k: int = 10,
        vector_weight: float = 0.6,
        fts_weight: float = 0.4,
        types: Optional[List[str]] = None,
    ) -> List[VectorSearchResult]:
        if not query.strip():
            return []

        # Vector search
        q_vec = self._emb.embed([query])[0]
        vec_results = self._vs.search(q_vec, k=min(k * 3, 30))

        # FTS5 search
        fts_results: List[Dict[str, Any]] = []
        if self._fts5:
            try:
                fts_results = self._fts5(query, k=min(k * 2, 20), types=types)
            except Exception:
                pass

        if not vec_results and not fts_results:
            return []

        if vec_results and not fts_results:
            return vec_results[:k]

        if fts_results and not vec_results:
            return [_fts_to_result(r) for r in fts_results[:k]]

        # Reciprocal rank fusion
        scores: Dict[str, float] = {}
        content_map: Dict[str, Any] = {}
        meta_map: Dict[str, Dict[str, Any]] = {}

        for rank, r in enumerate(vec_results):
            key = f"v_{r.id}"
            scores[key] = vector_weight / (self._k_rrf + rank + 1)
            content_map[key] = r.content
            meta_map[key] = r.metadata

        for rank, r in enumerate(fts_results):
            key = f"f_{r.get('id', '')}"
            s = fts_weight / (self._k_rrf + rank + 1)
            scores[key] = scores.get(key, 0.0) + s
            if key not in content_map:
                content_map[key] = r.get("content")
            if key not in meta_map:
                meta_map[key] = {k: r[k] for k in r if k not in ("id", "content")}

        # Sort by fused score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results: List[VectorSearchResult] = []
        for key, score in ranked[:k]:
            uid = key[2:]  # strip "v_" or "f_" prefix
            results.append(VectorSearchResult(
                id=_try_aton(uid),
                score=score,
                content=content_map.get(key),
                metadata=meta_map.get(key, {}),
            ))
        return results

    def embed_and_add(self, ids: List[Any], texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        vecs = self._emb.embed(texts)
        self._vs.add(ids, vecs, metadatas)

    @property
    def vector_store(self) -> AbstractVectorStore:
        return self._vs

    @property
    def embedding_provider(self) -> AbstractEmbeddingProvider:
        return self._emb


def _fts_to_result(r: Dict[str, Any]) -> VectorSearchResult:
    return VectorSearchResult(
        id=r.get("id"),
        score=r.get("importance", 0.5),
        content=r.get("content"),
        metadata={k: r[k] for k in r if k not in ("id", "content")},
    )


def _try_aton(s):
    try:
        return int(s)
    except (ValueError, TypeError):
        return s
