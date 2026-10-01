"""Delayed feedback (paper Sec. 4.2.2).

``FeedbackLoop.submit`` is called the moment a program finishes.  It returns at
once; a background task runs the bound verifier, turns (score, cost) into the
reward of Eq. 3 and updates the model selector in arrival order.  Programs may
therefore complete and be verified in a different order from their dispatch.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional

from ..routing.base import ModelSelector
from ..routing.reward import Reward
from ..types import Binding, ProgramOutcome
from .base import Verifier

logger = logging.getLogger(__name__)

FeedbackCallback = Callable[[Binding, float, float], Optional[Awaitable[None]]]


class FeedbackLoop:
    """Run verifiers asynchronously and feed rewards to the selector.

    Args:
        selector: the model selector to update.
        verifiers: ``{name: Verifier}``; a binding whose verifier name is missing
            falls back to ``outcome.success`` (``ReportedVerifier`` semantics).
        reward: the reward function (Eq. 3).
        max_concurrency: how many verifiers may run at once.
        on_reward: optional callback ``(binding, score, reward)`` after each update.

    A verifier that raises, or returns a NaN/inf score, counts as a failed
    verification: it is logged, ``failed`` is incremented and the selector is
    *not* updated (a NaN reward would corrupt a learner's statistics for good).
    """

    def __init__(self, selector: ModelSelector, verifiers: Optional[Dict[str, Verifier]] = None, reward: Optional[Reward] = None, max_concurrency: int = 8, on_reward: Optional[FeedbackCallback] = None, history: int = 500) -> None:
        self.selector = selector
        self.verifiers = dict(verifiers or {})
        self.reward = reward or Reward()
        self.on_reward = on_reward
        self._sem = asyncio.Semaphore(max_concurrency)
        self._tasks: set = set()
        self._update_lock = asyncio.Lock()
        self.pending = 0
        self.completed = 0
        self.failed = 0
        self.history: Deque[Dict[str, Any]] = deque(maxlen=history)

    def submit(self, binding: Binding, outcome: ProgramOutcome) -> "asyncio.Task[float]":
        """Schedule verification + update.  Returns the task (await it in tests)."""
        self.pending += 1
        t = asyncio.create_task(self._run(binding, outcome))
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        return t

    async def _run(self, binding: Binding, outcome: ProgramOutcome) -> float:
        t0 = time.time()
        try:
            async with self._sem:
                v = self.verifiers.get(binding.verifier or "")
                if v is None:
                    if outcome.success is None:
                        raise KeyError(f"no verifier {binding.verifier!r} and no reported success for {binding.program_id}")
                    score = float(outcome.success)
                else:
                    score = float(await v.verify(outcome))
            if not math.isfinite(score):
                raise ValueError(f"verifier {binding.verifier!r} returned a non-finite score {score!r} for program {binding.program_id}")
            score = min(max(score, 0.0), 1.0)
            r = self.reward(score, outcome.cost)  # raises on a non-finite cost when the cost term is active
            try:
                cost: Optional[float] = float(outcome.cost)
                cost = cost if math.isfinite(cost) else None  # a NaN would make the dashboard's JSON invalid
            except (TypeError, ValueError):
                cost = None
            async with self._update_lock:  # updates are applied in arrival order
                if binding.context is not None:
                    self.selector.update(binding.context, binding.model, r)
            binding.score, binding.reward = score, r
            self.completed += 1
            self.history.append({"program_id": binding.program_id, "model": binding.model, "verifier": binding.verifier, "task_type": binding.context.task_type if binding.context else None, "score": score, "cost": cost, "reward": r, "verify_s": round(time.time() - t0, 3), "t": time.time()})
            if self.on_reward is not None:
                res = self.on_reward(binding, score, r)
                if asyncio.iscoroutine(res):
                    await res
            return r
        except Exception:
            self.failed += 1
            logger.exception("verification failed for program %s", binding.program_id)
            return float("nan")
        finally:
            self.pending -= 1

    async def drain(self) -> None:
        """Wait for every in-flight verification (tests, shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    def state(self) -> Dict[str, Any]:
        per_model: Dict[str, List[float]] = {}
        for h in self.history:
            per_model.setdefault(h["model"], []).append(h["score"])
        return {
            "pending": self.pending,
            "completed": self.completed,
            "failed": self.failed,
            "verifiers": sorted(self.verifiers),
            "recent_accuracy": {m: round(sum(s) / len(s), 3) for m, s in per_model.items()},
            "recent": list(self.history)[-20:],
        }
