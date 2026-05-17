from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base import AbstractVectorStore, VectorSearchResult


class LanceDBStore(AbstractVectorStore):
    """LanceDB-backed vector store.  Serverless, columnar storage."""

    def __init__(self, uri: str = "memory/lancedb", table_name: str = "lifers_memory") -> None:
        self._uri = uri
        self._table_name = table_name
        self._db = None
        self._table = None

    def _lazy_connect(self):
        if self._table is not None:
            return
        try:
            import lancedb
        except ImportError:
            raise ImportError(
                "lancedb not installed. Run: pip install lancedb"
            )
        self._db = lancedb.connect(self._uri)
        try:
            self._table = self._db.open_table(self._table_name)
        except Exception:
            pass  # table will be created on first add

    def add(self, ids: List[Any], vectors: np.ndarray, metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        import lancedb
        self._lazy_connect()
        data: List[Dict[str, Any]] = []
        for i, uid in enumerate(ids):
            row = {
                "id": str(uid),
                "vector": vectors[i].astype(np.float32).tolist(),
            }
            if metadatas and i < len(metadatas):
                for k, v in metadatas[i].items():
                    row[k] = _safe_meta_val(v)
            data.append(row)
        if self._table is None:
            self._table = self._db.create_table(self._table_name, data)
        else:
            self._table.add(data)

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[VectorSearchResult]:
        self._lazy_connect()
        if self._table is None:
            return []
        q = query_vector.astype(np.float32).tolist()
        try:
            df = self._table.search(q).limit(k).to_pandas()
        except Exception:
            return []
        results: List[VectorSearchResult] = []
        for _, row in df.iterrows():
            meta = {k: row[k] for k in row.index if k not in ("id", "vector", "_distance")}
            results.append(VectorSearchResult(
                id=_try_aton(row.get("id", "")),
                score=1.0 - float(row.get("_distance", 0)),
                metadata=meta,
            ))
        return results

    def delete(self, ids: List[Any]) -> None:
        self._lazy_connect()
        if self._table is None or not ids:
            return
        str_ids = [str(uid) for uid in ids]
        try:
            self._table.delete(f"id IN ({','.join(repr(s) for s in str_ids)})")
        except Exception:
            pass

    def count(self) -> int:
        self._lazy_connect()
        if self._table is None:
            return 0
        try:
            return self._table.count_rows()
        except Exception:
            return len(self._table.to_pandas())

    def clear(self) -> None:
        self._lazy_connect()
        if self._db is not None:
            try:
                self._db.drop_table(self._table_name)
            except Exception:
                pass
            self._table = None


def _safe_meta_val(v: Any) -> str:
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _try_aton(s: str):
    try:
        return int(s)
    except (ValueError, TypeError):
        return s
