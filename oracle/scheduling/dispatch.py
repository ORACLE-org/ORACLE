"""The dispatch rule (paper Eq. 4).

Given the router's proposed model and the forecast admission wait of every
backend, divert the program to a backend with a shorter wait only when the
reward gained from the shorter wait exceeds the reward lost from lower accuracy:

    beta * (w_proposed - w_alt) / w0  >  (1 - beta) * (a_hat_proposed - a_hat_alt)

``beta`` trades accuracy for wait, ``w0`` normalises the wait, and ``a_hat`` is
the model selector's current accuracy estimate.  Without estimates no
diversion happens.  ``beta = 0`` disables diversion; ``beta = 1`` is pure
least-wait dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple


@dataclass
class DispatchPolicy:
    beta: float = 0.5
    w0: float = 60.0
    """Reference wait in seconds (the wait at which the wait term equals ``beta``)."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if self.w0 <= 0:
            raise ValueError("w0 must be positive")

    def gain(self, w_from: float, w_to: float) -> float:
        return self.beta * (w_from - w_to) / self.w0

    def loss(self, a_from: Optional[float], a_to: Optional[float]) -> Optional[float]:
        if a_from is None or a_to is None:
            return None
        return (1.0 - self.beta) * (a_from - a_to)

    def choose(self, proposed: str, waits: Mapping[str, float], estimates: Mapping[str, Optional[float]]) -> Tuple[str, Dict[str, float]]:
        """Return ``(model, detail)``: the backend to dispatch to and the per-candidate margins."""
        if self.beta <= 0.0 or proposed not in waits:
            return proposed, {}
        w_p = waits[proposed]
        best, best_margin, detail = proposed, 0.0, {}
        for m, w in waits.items():
            if m == proposed or w >= w_p:
                continue
            if self.beta >= 1.0:
                margin = w_p - w
            else:
                loss = self.loss(estimates.get(proposed), estimates.get(m))
                if loss is None:
                    continue
                margin = self.gain(w_p, w) - loss
            detail[m] = round(margin, 4)
            if margin > best_margin:
                best, best_margin = m, margin
        return best, detail
