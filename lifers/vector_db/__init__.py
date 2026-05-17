from __future__ import annotations

from .base import (
    AbstractEmbeddingProvider,
    AbstractVectorStore,
    VectorSearchResult,
)
from .embeddings import (
    TFIDFEmbeddingProvider,
    SentenceTransformerProvider,
    create_embedding_provider,
)
from .hybrid import HybridSearch

__all__ = [
    "AbstractEmbeddingProvider",
    "AbstractVectorStore",
    "VectorSearchResult",
    "TFIDFEmbeddingProvider",
    "SentenceTransformerProvider",
    "create_embedding_provider",
    "HybridSearch",
    # Lazy-loaded backends
    "FAISSStore",
    "ChromaStore",
    "LanceDBStore",
]


def __getattr__(name: str):
    if name == "FAISSStore":
        from .faiss_store import FAISSStore
        return FAISSStore
    if name == "ChromaStore":
        from .chroma_store import ChromaStore
        return ChromaStore
    if name == "LanceDBStore":
        from .lancedb_store import LanceDBStore
        return LanceDBStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
