"""Selectors that do not learn: useful as baselines and as plumbing."""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from ..types import RoutingContext
from .base import ModelSelector
from .registry import register_selector


@register_selector("fixed")
class Fixed(ModelSelector):
    """Always the same model (defaults to the first one)."""

    def __init__(self, models: Sequence[str], model: Optional[str] = None) -> None:
        super().__init__(models)
        self.model = model or self.models[0]
        self._check(self.model)

    def select(self, ctx: RoutingContext) -> str:
        self.n_select += 1
        return self.model

    def update(self, ctx: RoutingContext, model: str, reward: float) -> None:
        self._check(model)
        self.n_update += 1


@register_selector("round_robin")
class RoundRobin(ModelSelector):
    def __init__(self, models: Sequence[str]) -> None:
        super().__init__(models)
        self._i = 0

    def select(self, ctx: RoutingContext) -> str:
        self.n_select += 1
        m = self.models[self._i % len(self.models)]
        self._i += 1
        return m

    def update(self, ctx: RoutingContext, model: str, reward: float) -> None:
        self._check(model)
        self.n_update += 1


@register_selector("random")
class Random(ModelSelector):
    """Random mixing; ``weights`` gives the share of each model."""

    def __init__(self, models: Sequence[str], weights: Optional[Sequence[float]] = None, seed: int = 0) -> None:
        super().__init__(models)
        self.rng = np.random.default_rng(seed)
        w = np.ones(len(self.models)) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != (len(self.models),):
            raise ValueError(f"weights must have one entry per model ({len(self.models)}), got shape {w.shape}")
        if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
            raise ValueError("weights must be finite, non-negative and not all zero")
        self.p = w / w.sum()

    def select(self, ctx: RoutingContext) -> str:
        self.n_select += 1
        return self.models[int(self.rng.choice(len(self.models), p=self.p))]

    def update(self, ctx: RoutingContext, model: str, reward: float) -> None:
        self._check(model)
        self.n_update += 1


@register_selector("callable")
class CallableSelector(ModelSelector):
    """Wrap plain functions: ``select_fn(ctx) -> model`` and optional ``update_fn(ctx, model, reward)``.

    The quickest way to plug in an existing router without subclassing.
    """

    def __init__(
        self,
        models: Sequence[str],
        select_fn: Callable[[RoutingContext], str],
        update_fn: Optional[Callable[[RoutingContext, str, float], None]] = None,
        estimate_fn: Optional[Callable[[RoutingContext, str], Optional[float]]] = None,
    ) -> None:
        super().__init__(models)
        self.select_fn, self.update_fn, self.estimate_fn = select_fn, update_fn, estimate_fn

    def select(self, ctx: RoutingContext) -> str:
        self.n_select += 1
        m = self.select_fn(ctx)
        self._check(m)
        return m

    def update(self, ctx: RoutingContext, model: str, reward: float) -> None:
        self._check(model)
        self.n_update += 1
        if self.update_fn:
            self.update_fn(ctx, model, reward)

    def estimate(self, ctx: RoutingContext, model: str) -> Optional[float]:
        return self.estimate_fn(ctx, model) if self.estimate_fn else None
