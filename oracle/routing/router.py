"""The router: lookup table + model selector + verifier selector + delayed feedback.

This is the "routing mechanism alone" entry point.  It knows nothing about
GPUs or KV caches; pair it with :class:`oracle.scheduling.DISC` through
:class:`oracle.Oracle` when you also want admission control.

Typical use::

    router = Router(models=["qwen3.6-27b", "qwen3.5-9b"], selector="linucb",
                    verifiers={"swe": my_patch_verifier, "tau2": my_state_verifier},
                    verifier_selector=PrototypeVerifierSelector(exemplars))
    b = router.bind("task-17", prompt=first_user_message)   # -> b.model, b.verifier
    ... run the agent on b.model ...
    router.complete("task-17", ProgramOutcome(program_id="task-17", cost=0.12, payload={...}))

``verifiers`` takes :class:`~oracle.verification.Verifier` instances or plain
``fn(outcome) -> score`` callables (wrapped in a ``CallableVerifier``).
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

from ..types import Binding, ProgramOutcome, RoutingContext
from ..verification.base import CallableVerifier, Verifier
from ..verification.feedback import FeedbackLoop
from ..verification.selector import FixedVerifierSelector, VerifierSelector
from .base import ModelSelector
from .features import FeatureExtractor, HashingFeatures
from .registry import make_selector
from .reward import Reward


def _as_verifiers(verifiers: Optional[Mapping[str, Union[Verifier, Callable[[ProgramOutcome], Any]]]]) -> Dict[str, Verifier]:
    out: Dict[str, Verifier] = {}
    for name, v in (verifiers or {}).items():
        if isinstance(v, Verifier):
            out[name] = v
        elif callable(v):
            out[name] = CallableVerifier(name, v)
        else:
            raise TypeError(f"verifier {name!r} must be a Verifier or a callable, got {type(v).__name__}")
    return out


class Router:
    def __init__(
        self,
        models: Sequence[str],
        selector: Union[str, ModelSelector] = "linucb",
        selector_kwargs: Optional[Dict[str, Any]] = None,
        verifiers: Optional[Mapping[str, Union[Verifier, Callable[[ProgramOutcome], Any]]]] = None,
        verifier_selector: Optional[VerifierSelector] = None,
        features: Optional[FeatureExtractor] = None,
        reward: Optional[Reward] = None,
        max_verify_concurrency: int = 8,
    ) -> None:
        self.models = list(models)
        if not self.models:
            raise ValueError("Router needs at least one model")
        if len(set(self.models)) != len(self.models):
            raise ValueError(f"duplicate model names: {self.models}")
        self.features = features if features is not None else HashingFeatures(dim=64)
        if isinstance(selector, str):
            kw = dict(selector_kwargs or {})
            if selector in ("linucb", "lints"):
                kw.setdefault("dim", self.features.dim)
            selector = make_selector(selector, self.models, **kw)
        self.selector: ModelSelector = selector
        self.verifiers = _as_verifiers(verifiers)
        if verifier_selector is None:
            verifier_selector = FixedVerifierSelector(next(iter(self.verifiers)) if self.verifiers else "reported")
        self.verifier_selector = verifier_selector
        self.reward = reward or Reward()
        self.feedback = FeedbackLoop(self.selector, self.verifiers, self.reward, max_concurrency=max_verify_concurrency)
        self.table: Dict[str, Binding] = {}
        self.finished: Dict[str, Binding] = {}
        self.max_finished = 1000

    # --- lookup table -------------------------------------------------------
    def context(self, program_id: str, prompt: str = "", metadata: Optional[Dict[str, Any]] = None) -> RoutingContext:
        """Build the routing context: verifier/task type first, then features (which may use the task type)."""
        ctx = RoutingContext(program_id=program_id, prompt=prompt, metadata=dict(metadata or {}))
        task_type, verifier = self.verifier_selector.select(ctx)
        ctx.task_type = task_type
        ctx.metadata["task_type"] = task_type
        ctx.metadata["verifier"] = verifier
        ctx.features = self.features(prompt, ctx.metadata)
        return ctx

    def _known(self, model: Any, who: str) -> str:
        if model not in self.models:
            raise ValueError(f"{who} chose {model!r}, which is not one of the router's models {self.models}")
        return model

    def propose(self, ctx: RoutingContext) -> str:
        """Ask the model selector for a model without recording a binding."""
        return self._known(self.selector.select(ctx), f"selector {self.selector.name!r}")

    async def apropose(self, ctx: RoutingContext) -> str:
        """Async :meth:`propose`: lets selectors that do I/O (``aselect``) run without blocking the event loop."""
        return self._known(await self.selector.aselect(ctx), f"selector {self.selector.name!r}")

    def bind(self, program_id: str, prompt: str = "", metadata: Optional[Dict[str, Any]] = None, model: Optional[str] = None) -> Binding:
        """Create the program's row (idempotent: a second call returns the existing row).

        ``model`` overrides the selector's choice (DISC uses this after a diversion).
        """
        if program_id in self.table:
            return self.table[program_id]
        if model is not None:
            self._known(model, "bind(model=...)")
        ctx = self.context(program_id, prompt, metadata)
        proposed = self.propose(ctx)
        b = Binding(program_id=program_id, model=model or proposed, verifier=ctx.metadata["verifier"], proposed_model=proposed, diverted=bool(model and model != proposed), context=ctx)
        self.table[program_id] = b
        return b

    def lookup(self, program_id: str) -> Optional[Binding]:
        return self.table.get(program_id)

    def estimates(self, ctx: RoutingContext) -> Dict[str, Optional[float]]:
        """Selector's current accuracy/reward estimate per model (for DISC)."""
        return {m: self.selector.estimate(ctx, m) for m in self.models}

    # --- completion -> delayed feedback -------------------------------------
    def complete(self, program_id: str, outcome: Optional[ProgramOutcome] = None, **fields) -> Optional["asyncio.Task[float]"]:
        """Mark a program finished, release its row and verify asynchronously.

        Must be called from a running event loop (verification is scheduled as
        a task); from synchronous code use :meth:`complete_sync`.  Returns the
        verification task, or ``None`` for an unknown program.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError("Router.complete() needs a running event loop: call it from async code, or use Router.complete_sync() from a plain script") from None
        b = self.table.pop(program_id, None)
        if b is None:
            return None
        if outcome is None:
            outcome = ProgramOutcome(program_id=program_id, **fields)
        if outcome.cost == 0.0 and b.cost:
            outcome.cost = b.cost
        b.completed_at = time.time()
        self.finished[program_id] = b
        if len(self.finished) > self.max_finished:
            self.finished.pop(next(iter(self.finished)))
        return self.feedback.submit(b, outcome)

    def complete_sync(self, program_id: str, outcome: Optional[ProgramOutcome] = None, **fields) -> Optional[float]:
        """Blocking variant for scripts without an event loop: verifies before returning."""

        async def _go():
            t = self.complete(program_id, outcome, **fields)
            return await t if t is not None else None

        return asyncio.run(_go())

    def release(self, program_id: str) -> bool:
        """Drop a program without feedback (client went away)."""
        return self.table.pop(program_id, None) is not None

    # --- introspection ------------------------------------------------------
    def state(self) -> Dict[str, Any]:
        per_model = {m: 0 for m in self.models}
        for b in self.table.values():
            per_model[b.model] = per_model.get(b.model, 0) + 1
        return {
            "models": self.models,
            "selector": self.selector.state(),
            "verifier_selector": self.verifier_selector.state(),
            "feedback": self.feedback.state(),
            "active": len(self.table),
            "active_per_model": per_model,
            "programs": [b.to_dict() for b in self.table.values()],
            "finished": [b.to_dict() for b in list(self.finished.values())[-50:]],
        }
