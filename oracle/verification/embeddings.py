"""Text embedders used by the verifier selector (and optionally as routing features).

All embedders return unit-norm vectors so that a dot product is a cosine similarity.
"""
from __future__ import annotations

import hashlib
import re
from typing import Optional, Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim) with unit-norm rows."""


_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]+|\d+")


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.where(n > 0, x / np.where(n > 0, n, 1.0), x)


class HashingEmbedder:
    """Hashed unigram + bigram bag of words.  No dependencies, no downloads.

    Good enough to separate task families whose requests use different
    vocabularies (patch a repo vs. book a flight).  Use
    :class:`SentenceTransformerEmbedder` for anything subtle.
    """

    def __init__(self, dim: int = 1024, bigrams: bool = True) -> None:
        self.dim, self.bigrams = int(dim), bigrams

    def _hash(self, s: str) -> int:
        return int(hashlib.blake2b(s.encode(), digest_size=4).hexdigest(), 16) % self.dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for i, t in enumerate(texts):
            toks = _TOKEN.findall(t.lower())
            for w in toks:
                out[i, self._hash(w)] += 1.0
            if self.bigrams:
                for a, b in zip(toks, toks[1:]):
                    out[i, self._hash(a + " " + b)] += 0.5
        return _unit(np.log1p(out))


class SentenceTransformerEmbedder:
    """Any sentence-transformers model, e.g. ``BAAI/bge-m3`` (the paper's choice)."""

    def __init__(self, model: str = "BAAI/bge-m3", device: Optional[str] = None, batch_size: int = 32) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install sentence-transformers") from e
        self.model = SentenceTransformer(model, device=device)
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return _unit(np.asarray(self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True)))


class OpenAIEmbedder:
    """Any OpenAI-compatible ``/v1/embeddings`` endpoint (vLLM, TEI, OpenAI)."""

    def __init__(self, url: str, model: str, api_key: str = "EMPTY", timeout_s: float = 60.0) -> None:
        self.url, self.model, self.api_key, self.timeout_s = url.rstrip("/"), model, api_key, timeout_s

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        import httpx

        r = httpx.post(f"{self.url}/embeddings", json={"model": self.model, "input": list(texts)}, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=self.timeout_s)
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        return _unit(np.asarray([d["embedding"] for d in data], dtype=float))


def make_embedder(spec: Optional[str] = None, **kw) -> Embedder:
    """``None``/``"hashing"`` -> HashingEmbedder; ``"st:BAAI/bge-m3"`` -> sentence-transformers;
    ``"openai:MODEL"`` with ``url=`` -> OpenAIEmbedder."""
    if spec in (None, "", "hashing"):
        return HashingEmbedder(**kw)
    if spec.startswith("st:"):
        return SentenceTransformerEmbedder(spec[3:], **kw)
    if spec.startswith("openai:"):
        return OpenAIEmbedder(model=spec[7:], **kw)
    raise ValueError(f"unknown embedder spec {spec!r}")
