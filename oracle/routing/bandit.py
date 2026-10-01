"""Bandit selectors. ``linucb`` is ORACLE's default."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..types import RoutingContext
from .base import ModelSelector, finite_reward
from .registry import register_selector


def _ctx_features(ctx: RoutingContext, d: int) -> np.ndarray:
    if ctx.features is None:
        x = np.zeros(d)
        x[-1] = 1.0
        return x
    x = np.asarray(ctx.features, dtype=float)
    if x.shape != (d,):
        raise ValueError(f"feature shape {x.shape} != selector dim ({d},)")
    if not np.isfinite(x).all():
        raise ValueError("features must be finite (no NaN/inf)")
    return x


@register_selector("linucb")
class LinUCB(ModelSelector):
    """Disjoint LinUCB (Li et al. 2010): one ridge-regression arm per model.

    Args:
        models: candidate models.
        dim: feature dimension (match your ``FeatureExtractor.dim``).
        alpha_ucb: exploration width.  Called ``alpha_UCB`` in the paper's appendix
            to distinguish it from the reward's cost weight.
        lam: ridge regulariser.
        seed: tie-breaking RNG seed.
    """

    def __init__(self, models: Sequence[str], dim: int = 64, alpha_ucb: float = 1.0, lam: float = 1.0, seed: int = 0) -> None:
        super().__init__(models)
        self.d = int(dim)
        self.alpha_ucb = float(alpha_ucb)
        self.rng = np.random.default_rng(seed)
        k = len(self.models)
        self.A = np.stack([lam * np.eye(self.d) for _ in range(k)])
        self.b = np.zeros((k, self.d))
        self._Ainv = np.stack([np.eye(self.d) / lam for _ in range(k)])
        self.pulls = np.zeros(k, dtype=int)

    def _theta(self, j: int) -> np.ndarray:
        return self._Ainv[j] @ self.b[j]

    def scores(self, ctx: RoutingContext) -> Dict[str, float]:
        x = _ctx_features(ctx, self.d)
        out = {}
        for j, m in enumerate(self.models):
            Ai = self._Ainv[j]
            out[m] = float(self._theta(j) @ x + self.alpha_ucb * math.sqrt(max(float(x @ Ai @ x), 0.0)))
        return out

    def select(self, ctx: RoutingContext) -> str:
        self.n_select += 1
        p = self.scores(ctx)
        vals = np.array([p[m] for m in self.models])
        best = np.flatnonzero(vals >= vals.max() - 1e-12)
        j = int(self.rng.choice(best))
        self.pulls[j] += 1
        return self.models[j]

    def update(self, ctx: RoutingContext, model: str, reward: float) -> None:
        self.n_update += 1
        j = self._check(model)
        x = _ctx_features(ctx, self.d)
        r = finite_reward(reward)
        self.A[j] += np.outer(x, x)
        self.b[j] += r * x
        self._Ainv[j] = np.linalg.inv(self.A[j])

    def estimate(self, ctx: RoutingContext, model: str) -> Optional[float]:
        j = self._check(model)
        if self.n_update == 0:
            return None
        return float(self._theta(j) @ _ctx_features(ctx, self.d))

    def state(self) -> Dict[str, Any]:
        s = super().state()
        s.update({"alpha_ucb": self.alpha_ucb, "dim": self.d, "pulls": {m: int(self.pulls[j]) for j, m in enumerate(self.models)}})
        return s


@register_selector("lints")
class LinTS(LinUCB):
    """Linear Thompson sampling with the same ridge statistics as LinUCB."""

    def __init__(self, models: Sequence[str], dim: int = 64, v: float = 0.5, lam: float = 1.0, seed: int = 0) -> None:
        super().__init__(models, dim=dim, alpha_ucb=0.0, lam=lam, seed=seed)
        self.v = float(v)

    def scores(self, ctx: RoutingContext) -> Dict[str, float]:
        x = _ctx_features(ctx, self.d)
        out = {}
        for j, m in enumerate(self.models):
            th = self.rng.multivariate_normal(self._theta(j), self.v ** 2 * self._Ainv[j])
            out[m] = float(th @ x)
        return out


@register_selector("ucb")
class TypeUCB(ModelSelector):
    """UCB1 over (task type, model) cells: the non-contextual learner.

    Uses ``ctx.task_type`` (set by the verifier selector) as the context; falls
    back to a single global cell when no task type is known.

    Args:
        delta: confidence level; width is ``sqrt(2 ln(1/delta) / n)``.
        prior: optional ``{(task_type, model): (n, mean_reward)}`` warm start.
    """

    def __init__(self, models: Sequence[str], delta: float = 0.1, contextual: bool = True, prior: Optional[Dict] = None, seed: int = 0) -> None:
        super().__init__(models)
        self.delta = float(delta)
        self.contextual = contextual
        self.rng = np.random.default_rng(seed)
        self.n: Dict[tuple, int] = {}
        self.s: Dict[tuple, float] = {}
        for (g, m), (n, mean) in (prior or {}).items():
            self.n[(g, m)] = int(n)
            self.s[(g, m)] = float(mean) * int(n)

    def _cell(self, ctx: RoutingContext) -> str:
        return (ctx.task_type or "_") if self.contextual else "_"

    def _index(self, g: str, m: str) -> float:
        n = self.n.get((g, m), 0)
        if n == 0:
            return float("inf")
        return self.s[(g, m)] / n + math.sqrt(2.0 * math.log(1.0 / self.delta) / n)

    def select(self, ctx: RoutingContext) -> str:
        self.n_select += 1
        g = self._cell(ctx)
        vals = np.array([self._index(g, m) for m in self.models])
        best = np.flatnonzero(vals >= vals.max() - 1e-12)
        return self.models[int(self.rng.choice(best))]

    def update(self, ctx: RoutingContext, model: str, reward: float) -> None:
        self.n_update += 1
        self._check(model)
        r = finite_reward(reward)
        k = (self._cell(ctx), model)
        self.n[k] = self.n.get(k, 0) + 1
        self.s[k] = self.s.get(k, 0.0) + r

    def estimate(self, ctx: RoutingContext, model: str) -> Optional[float]:
        k = (self._cell(ctx), model)
        n = self.n.get(k, 0)
        return self.s[k] / n if n else None

    def state(self) -> Dict[str, Any]:
        s = super().state()
        s["cells"] = {f"{g}/{m}": {"n": n, "mean": round(self.s[(g, m)] / n, 4)} for (g, m), n in sorted(self.n.items()) if n}
        return s


@register_selector("epsilon_greedy")
class EpsilonGreedy(TypeUCB):
    """Greedy on the per-cell mean with probability ``1 - epsilon``, random otherwise."""

    def __init__(self, models: Sequence[str], epsilon: float = 0.1, contextual: bool = True, seed: int = 0) -> None:
        super().__init__(models, contextual=contextual, seed=seed)
        self.epsilon = float(epsilon)

    def select(self, ctx: RoutingContext) -> str:
        self.n_select += 1
        if self.rng.random() < self.epsilon:
            return self.models[int(self.rng.integers(len(self.models)))]
        g = self._cell(ctx)
        vals = np.array([(self.s[(g, m)] / self.n[(g, m)]) if self.n.get((g, m), 0) else float("inf") for m in self.models])
        best = np.flatnonzero(vals >= vals.max() - 1e-12)
        return self.models[int(self.rng.choice(best))]
