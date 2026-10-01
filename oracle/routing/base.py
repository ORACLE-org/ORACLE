"""The model-selector interface: the pluggable "router backend".

Every routing policy ORACLE can use (the default LinUCB bandit, ACRouter,
RouteLLM, a fixed model, your own) implements :class:`ModelSelector`.  The
rest of the system only ever calls three methods:

* :meth:`ModelSelector.select` picks a model for a new program;
* :meth:`ModelSelector.update` feeds back the delayed reward once the program's
  verifier has run;
* :meth:`ModelSelector.estimate` (optional) returns the selector's current
  accuracy estimate for a model so the DISC scheduler can price a diversion.

Write a new one in a few lines::

    from oracle.routing import ModelSelector, register_selector

    @register_selector("cheapest_first")
    class CheapestFirst(ModelSelector):
        def select(self, ctx):
            return self.models[-1]            # models are listed strong -> weak
        def update(self, ctx, model, reward):
            pass
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence

from ..types import RoutingContext


class ModelSelector(ABC):
    """Base class for routing policies.

    Args:
        models: Names of the candidate models, in a fixed order.  By convention
            the list is ordered from strongest (most expensive) to weakest.
    """

    name: str = "selector"

    def __init__(self, models: Sequence[str]) -> None:
        if not models:
            raise ValueError("a selector needs at least one model")
        self.models = list(models)
        self.index = {m: i for i, m in enumerate(self.models)}
        self.n_select = 0
        self.n_update = 0

    # --- the contract -------------------------------------------------------
    @abstractmethod
    def select(self, ctx: RoutingContext) -> str:
        """Return the model for a new program."""

    @abstractmethod
    def update(self, ctx: RoutingContext, model: str, reward: float) -> None:
        """Delayed feedback for a completed program routed to ``model``."""

    def estimate(self, ctx: RoutingContext, model: str) -> Optional[float]:
        """Current estimate of the reward/accuracy of ``model`` for this context.

        Used by DISC's dispatch rule (paper Eq. 4). Return ``None`` if the
        policy has no estimate, in which case DISC never diverts.
        """
        return None

    # --- optional niceties --------------------------------------------------
    def state(self) -> Dict[str, Any]:
        """Serialisable view for the dashboard and for checkpoints."""
        return {"name": self.name, "models": self.models, "n_select": self.n_select, "n_update": self.n_update}

    def save(self, path: str) -> None:
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "ModelSelector":
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)

    def _check(self, model: str) -> int:
        if model not in self.index:
            raise KeyError(f"unknown model {model!r}; known: {self.models}")
        return self.index[model]
