"""Paper Eq. 3: the delayed reward of a completed program.

    r = (1 - alpha) * a  -  alpha * min(cost / cost_ref, 1)

``a`` is the verifier's score in [0, 1], ``cost`` the program's final cost and
``cost_ref`` a reference cost that normalises it.  ``alpha = 0`` learns on
accuracy alone (the paper's live-loop setting).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Reward:
    alpha: float = 0.0
    cost_ref: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if not self.cost_ref > 0:
            raise ValueError("cost_ref must be positive")

    def __call__(self, score: float, cost: float) -> float:
        """Return the reward.  Raises ``ValueError`` for a NaN/inf score, or cost when ``alpha > 0``,
        so that a broken verifier can never feed a NaN into the model selector."""
        score = float(score)
        if not math.isfinite(score):
            raise ValueError(f"score must be finite, got {score!r}")
        score = min(max(score, 0.0), 1.0)
        if self.alpha == 0.0:
            return score  # the cost does not enter; a missing or NaN cost must not poison the reward
        cost = float(cost)
        if not math.isfinite(cost):
            raise ValueError(f"cost must be finite, got {cost!r}")
        cost_n = min(max(cost, 0.0) / self.cost_ref, 1.0)
        return (1.0 - self.alpha) * score - self.alpha * cost_n
