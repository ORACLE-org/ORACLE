"""Feature extractors turn a program's initial request into a vector for
contextual learners (LinUCB, LinTS).  The default needs no dependencies."""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Optional, Protocol, Sequence

import numpy as np


class FeatureExtractor(Protocol):
    dim: int

    def __call__(self, prompt: str, metadata: Dict[str, Any]) -> np.ndarray: ...


_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]+|\d+|[^\sA-Za-z0-9_]")


class HashingFeatures:
    """Hashed bag of words plus a few length features, L2-normalised, with a bias term.

    Zero dependencies, deterministic, good enough for the bandit to learn which
    task families each model handles well.  Swap in :class:`EmbeddingFeatures`
    for a real sentence embedding.
    """

    def __init__(self, dim: int = 64, max_tokens: int = 512) -> None:
        self.dim = dim
        self.max_tokens = max_tokens

    def __call__(self, prompt: str, metadata: Dict[str, Any]) -> np.ndarray:
        x = np.zeros(self.dim, dtype=float)
        toks = _TOKEN.findall(prompt.lower())[: self.max_tokens]
        for t in toks:
            h = int(hashlib.blake2b(t.encode(), digest_size=4).hexdigest(), 16)
            x[h % (self.dim - 2)] += 1.0
        n = np.linalg.norm(x)
        if n > 0:
            x /= n
        x[-2] = min(len(prompt) / 4000.0, 1.0)  # length
        x[-1] = 1.0  # bias
        return x


class EmbeddingFeatures:
    """Use any :class:`oracle.verification.Embedder` as the feature vector (plus bias).

    Optional ``projection`` (``dim`` x ``embed_dim``) reduces the dimension; a
    random Gaussian projection is drawn with ``seed`` if ``dim`` is given.
    """

    def __init__(self, embedder, dim: Optional[int] = None, seed: int = 0) -> None:
        self.embedder = embedder
        self._proj = None
        probe = np.asarray(embedder.embed(["probe"])[0])
        if dim is not None and dim < probe.shape[0]:
            rng = np.random.default_rng(seed)
            self._proj = rng.standard_normal((dim, probe.shape[0])) / np.sqrt(dim)
            self.dim = dim + 1
        else:
            self.dim = probe.shape[0] + 1

    def __call__(self, prompt: str, metadata: Dict[str, Any]) -> np.ndarray:
        e = np.asarray(self.embedder.embed([prompt])[0], dtype=float)
        if self._proj is not None:
            e = self._proj @ e
        n = np.linalg.norm(e)
        if n > 0:
            e = e / n
        return np.concatenate([e, [1.0]])


class TaskTypeFeatures:
    """One-hot of the task type chosen by the verifier selector (plus bias).

    Pairs well with the per-task-type ``ucb`` selector or as extra context for
    LinUCB via :class:`ConcatFeatures`.
    """

    def __init__(self, task_types: Sequence[str]) -> None:
        self.task_types = list(task_types)
        self.dim = len(self.task_types) + 1

    def __call__(self, prompt: str, metadata: Dict[str, Any]) -> np.ndarray:
        x = np.zeros(self.dim)
        t = metadata.get("task_type")
        if t in self.task_types:
            x[self.task_types.index(t)] = 1.0
        x[-1] = 1.0
        return x


class ConcatFeatures:
    def __init__(self, *parts: FeatureExtractor) -> None:
        self.parts = parts
        self.dim = sum(p.dim for p in parts)

    def __call__(self, prompt: str, metadata: Dict[str, Any]) -> np.ndarray:
        return np.concatenate([p(prompt, metadata) for p in self.parts])
