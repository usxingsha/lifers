from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from .base import AbstractEmbeddingProvider


# ── TF-IDF embedding (zero extra deps) ────────────────────────────────────────

class TFIDFEmbeddingProvider(AbstractEmbeddingProvider):
    """Lightweight TF-IDF vectorizer.  Suitable as a local fallback when no
    sentence-transformer model is installed."""

    def __init__(self, dim: int = 256, max_features: int = 8192) -> None:
        self._dim = dim
        self._max_features = max_features
        self._vocab: Dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None
        self._projection: Optional[np.ndarray] = None  # [max_features, dim]

    @property
    def dim(self) -> int:
        return self._dim

    def fit(self, corpus: List[str]) -> "TFIDFEmbeddingProvider":
        """Build vocabulary + IDF from a corpus, then create a random projection."""
        df = Counter()
        tokenized: List[List[str]] = []
        for text in corpus:
            tokens = _tokenize(text)
            tokenized.append(tokens)
            df.update(set(tokens))
        # Keep top max_features by document frequency
        top = [w for w, _ in df.most_common(self._max_features)]
        self._vocab = {w: i for i, w in enumerate(top)}
        n_docs = len(corpus) or 1
        self._idf = np.ones(len(self._vocab), dtype=np.float32)
        for w, idx in self._vocab.items():
            self._idf[idx] = np.log((n_docs + 1) / (df.get(w, 1) + 1)) + 1.0
        # Random projection to target dim
        rng = np.random.RandomState(42)
        self._projection = rng.randn(len(self._vocab), self._dim).astype(np.float32) * 0.02
        return self

    def fit_from_documents(self, documents: List[dict]) -> "TFIDFEmbeddingProvider":
        """Fit from memory documents (each doc has 'content' field)."""
        texts = [_doc_content(d) for d in documents]
        return self.fit(texts)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self._vocab or self._projection is None:
            return np.zeros((len(texts), self._dim), dtype=np.float32)
        tfidf = self._tfidf_matrix(texts)
        return (tfidf @ self._projection).astype(np.float32)

    def _tfidf_matrix(self, texts: List[str]) -> np.ndarray:
        m = np.zeros((len(texts), len(self._vocab)), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = _tokenize(text)
            if not tokens:
                continue
            counts = Counter(tokens)
            for tok, cnt in counts.items():
                idx = self._vocab.get(tok)
                if idx is not None:
                    m[i, idx] = cnt * self._idf[idx]
            norm = np.linalg.norm(m[i])
            if norm > 1e-8:
                m[i] /= norm
        return m

    def save(self, path: str) -> None:
        data = {
            "dim": self._dim,
            "max_features": self._max_features,
            "vocab": self._vocab,
            "idf": self._idf.tolist() if self._idf is not None else None,
            "projection": self._projection.tolist() if self._projection is not None else None,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "TFIDFEmbeddingProvider":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        provider = cls(dim=data["dim"], max_features=data["max_features"])
        provider._vocab = {str(k): int(v) for k, v in data["vocab"].items()}
        if data["idf"] is not None:
            provider._idf = np.array(data["idf"], dtype=np.float32)
        if data["projection"] is not None:
            provider._projection = np.array(data["projection"], dtype=np.float32)
        return provider


# ── Sentence-Transformer embedding (optional dep) ─────────────────────────────

class SentenceTransformerProvider(AbstractEmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._dim = 384  # default for all-MiniLM-L6-v2

    @property
    def dim(self) -> int:
        if self._model is not None:
            self._dim = self._model.get_sentence_embedding_dimension() or self._dim
        return self._dim

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name, device=self._device)
            self._dim = self._model.get_sentence_embedding_dimension() or self._dim
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

    def embed(self, texts: List[str]) -> np.ndarray:
        self._lazy_load()
        assert self._model is not None
        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.astype(np.float32)


# ── Factory ───────────────────────────────────────────────────────────────────

def create_embedding_provider(
    kind: str = "tfidf",
    dim: int = 256,
    model_name: str = "all-MiniLM-L6-v2",
    corpus: Optional[List[str]] = None,
    tfidf_path: Optional[str] = None,
) -> AbstractEmbeddingProvider:
    if kind == "sentence_transformer":
        return SentenceTransformerProvider(model_name=model_name)
    if kind == "tfidf_load" and tfidf_path:
        return TFIDFEmbeddingProvider.load(tfidf_path)
    provider = TFIDFEmbeddingProvider(dim=dim)
    if corpus:
        provider.fit(corpus)
    return provider


# ── helpers ───────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _doc_content(doc: dict) -> str:
    c = doc.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, (dict, list)):
        return json.dumps(c, ensure_ascii=False)
    return str(c) if c is not None else ""
