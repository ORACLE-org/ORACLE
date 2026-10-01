"""Adaptive verification (paper Sec. 4.2.1, Eq. 2).

A task-type prototype ``mu_v`` is the mean embedding of a few exemplar requests
known to need verifier ``v``.  A new program is matched to the nearest
prototype and bound to that verifier for life.  Adding a verifier means adding
exemplars, not training anything.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..types import RoutingContext
from .embeddings import Embedder, HashingEmbedder


class VerifierSelector(ABC):
    @abstractmethod
    def select(self, ctx: RoutingContext) -> Tuple[str, str]:
        """Return ``(task_type, verifier_name)`` for the program's initial request."""

    def state(self) -> Dict[str, Any]:
        return {"type": type(self).__name__}


class FixedVerifierSelector(VerifierSelector):
    """Everyone gets the same verifier (the single-verifier baseline)."""

    def __init__(self, verifier: str, task_type: Optional[str] = None) -> None:
        self.verifier, self.task_type = verifier, task_type or verifier

    def select(self, ctx: RoutingContext) -> Tuple[str, str]:
        return self.task_type, self.verifier


class RuleVerifierSelector(VerifierSelector):
    """``fn(ctx) -> task_type``; verifier is ``mapping[task_type]`` (or the task type itself).

    Handy when the harness already knows the task family and passes it in
    ``ctx.metadata["task_type"]``::

        RuleVerifierSelector(lambda ctx: ctx.metadata["task_type"], {"swe": "patch_tests", "tau2": "state_check"})
    """

    def __init__(self, fn: Callable[[RoutingContext], str], mapping: Optional[Dict[str, str]] = None) -> None:
        self.fn, self.mapping = fn, mapping or {}

    def select(self, ctx: RoutingContext) -> Tuple[str, str]:
        t = self.fn(ctx)
        return t, self.mapping.get(t, t)


class PrototypeVerifierSelector(VerifierSelector):
    """Nearest-prototype selector over exemplar requests (Eq. 2).

    Args:
        exemplars: ``{task_type: [example request, ...]}``.
        verifier_of: ``{task_type: verifier_name}``; defaults to the identity.
        embedder: any :class:`Embedder`; the dependency-free hashing embedder by default.
        min_sim: if the best cosine similarity is below this, fall back to ``default``.
    """

    def __init__(
        self,
        exemplars: Dict[str, Sequence[str]],
        verifier_of: Optional[Dict[str, str]] = None,
        embedder: Optional[Embedder] = None,
        min_sim: float = -1.0,
        default: Optional[str] = None,
    ) -> None:
        if not exemplars:
            raise ValueError("need at least one task type with exemplars")
        self.embedder = embedder or HashingEmbedder()
        self.verifier_of = dict(verifier_of or {})
        self.min_sim, self.default = float(min_sim), default
        self.task_types: List[str] = []
        self.exemplars: Dict[str, List[str]] = {}
        self._mu: List[np.ndarray] = []
        for t, exs in exemplars.items():
            self.add_task_type(t, exs)
        self.counts: Dict[str, int] = {t: 0 for t in self.task_types}

    def add_task_type(self, task_type: str, exemplars: Sequence[str], verifier: Optional[str] = None) -> None:
        """Register (or extend) a task type from its exemplar requests."""
        exs = list(exemplars)
        if not exs:
            raise ValueError(f"task type {task_type!r} needs exemplars")
        if task_type in self.exemplars:
            self.exemplars[task_type].extend(exs)
            i = self.task_types.index(task_type)
            self._mu[i] = self._prototype(self.exemplars[task_type])
        else:
            self.task_types.append(task_type)
            self.exemplars[task_type] = exs
            self._mu.append(self._prototype(exs))
            if hasattr(self, "counts"):
                self.counts[task_type] = 0
        if verifier:
            self.verifier_of[task_type] = verifier

    def _prototype(self, exs: Sequence[str]) -> np.ndarray:
        e = np.asarray(self.embedder.embed(exs), dtype=float).mean(axis=0)
        n = np.linalg.norm(e)
        return e / n if n > 0 else e

    def similarities(self, prompt: str) -> Dict[str, float]:
        e = np.asarray(self.embedder.embed([prompt])[0], dtype=float)
        n = np.linalg.norm(e)
        if n > 0:
            e = e / n
        return {t: float(mu @ e) for t, mu in zip(self.task_types, self._mu)}

    def select(self, ctx: RoutingContext) -> Tuple[str, str]:
        sims = self.similarities(ctx.prompt)
        t = max(sims, key=sims.get)
        if sims[t] < self.min_sim and self.default is not None:
            t = self.default
        self.counts[t] = self.counts.get(t, 0) + 1
        ctx.metadata.setdefault("verifier_similarities", sims)
        return t, self.verifier_of.get(t, t)

    def state(self) -> Dict[str, Any]:
        return {"type": "prototype", "task_types": self.task_types, "exemplars": {t: len(v) for t, v in self.exemplars.items()}, "counts": self.counts, "verifier_of": self.verifier_of}
