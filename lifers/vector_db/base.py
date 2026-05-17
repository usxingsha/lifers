from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class VectorSearchResult:
    id: Any
    score: float
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AbstractEmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return [len(texts), dim] float32 array."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        ...


class AbstractVectorStore(ABC):
    @abstractmethod
    def add(self, ids: List[Any], vectors: np.ndarray, metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        ...

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[VectorSearchResult]:
        ...

    @abstractmethod
    def delete(self, ids: List[Any]) -> None:
        ...

    @abstractmethod
    def count(self) -> int:
        ...

    def clear(self) -> None:
        raise NotImplementedError
