"""``Oracle``: the router and DISC wired together.

Use the pieces you need:

* routing only  ->  :class:`oracle.routing.Router`
* scheduling only -> :class:`oracle.scheduling.DISC`
* both            -> :class:`Oracle` (this module)

With both, the flow per program is the paper's Fig. 3: the router binds the
program to a model and a verifier, DISC reads the ledgers and may divert the
program to a backend with a shorter forecast wait, the gate admits it, and on
completion the slot is released before the verifier runs.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional

from .routing.router import Router
from .scheduling.disc import DISC
from .types import Binding, ProgramOutcome


class Oracle:
    def __init__(self, router: Router, scheduler: Optional[DISC] = None, backend_of: Optional[Mapping[str, str]] = None) -> None:
        """Args:
            router: the routing mechanism.
            scheduler: DISC; ``None`` means route only.
            backend_of: ``{model: backend name}`` when backend names differ from
                model names (defaults to the identity, one model per backend).
                Several models may share a backend (co-located models).
        """
        self.router = router
        self.scheduler = scheduler
        self.backend_of = dict(backend_of or {m: m for m in router.models})
        missing = [m for m in router.models if m not in self.backend_of]
        if missing:
            raise ValueError(f"backend_of names no backend for models {missing}")
        self.models_on: Dict[str, List[str]] = {}
        """``{backend: [models served there, in the router's order]}``."""
        for m in router.models:
            self.models_on.setdefault(self.backend_of[m], []).append(m)
        self.model_of = {b: ms[0] for b, ms in self.models_on.items()}
        """``{backend: first (strongest) model served there}``."""
        if scheduler is not None:
            missing_b = [b for b in self.models_on if b not in scheduler.ledgers]
            if missing_b:
                raise ValueError(f"DISC has no ledger for backends {missing_b}")
        self._inflight: Dict[str, asyncio.Event] = {}
        """Programs whose first request is being routed/admitted right now."""

    @classmethod
    def from_config(cls, cfg) -> "Oracle":
        from .config import build

        return build(cfg)

    async def start(self) -> None:
        if self.scheduler is not None:
            await self.scheduler.start()

    async def stop(self) -> None:
        if self.scheduler is not None:
            await self.scheduler.stop()
        await self.router.feedback.drain()

    # --- program boundary ---------------------------------------------------
    async def bind(self, program_id: str, prompt: str = "", metadata: Optional[Dict[str, Any]] = None, tokens: int = 0, disconnected: Optional[Callable[[], Awaitable[bool]]] = None, model: Optional[str] = None) -> Binding:
        """Route, dispatch and admit a new program.  Idempotent for a known program.

        ``model`` pins the program to that model (the server's ``force_model``):
        no selector, no diversion, but still admitted by DISC.

        Blocks while DISC holds the program at the gate; the returned binding's
        ``admission_wait_s`` says for how long.  A second request of the same
        program that arrives meanwhile waits for that admission instead of
        skipping the gate.
        """
        while True:
            ev = self._inflight.get(program_id)
            if ev is None:
                break
            await ev.wait()
        b = self.router.lookup(program_id)
        if b is not None:
            return b
        ev = self._inflight[program_id] = asyncio.Event()
        try:
            return await self._bind(program_id, prompt, metadata, tokens, disconnected, model)
        finally:
            self._inflight.pop(program_id, None)
            ev.set()

    async def _bind(self, program_id: str, prompt: str, metadata: Optional[Dict[str, Any]], tokens: int, disconnected: Optional[Callable[[], Awaitable[bool]]], model: Optional[str]) -> Binding:
        ctx = self.router.context(program_id, prompt, metadata)
        detail: Dict[str, Any] = {}
        if model is not None:
            if model not in self.router.models:
                raise ValueError(f"unknown model {model!r}; known: {self.router.models}")
            proposed = chosen = model
            detail = {"forced": True}
        else:
            proposed = chosen = await self.router.apropose(ctx)
            if self.scheduler is not None:
                est = self.router.estimates(ctx)
                backend, detail = self.scheduler.dispatch(self.backend_of[proposed], self._backend_estimates(est))
                if backend != self.backend_of[proposed]:
                    chosen = self._model_on(backend, est)
        b = Binding(program_id=program_id, model=chosen, verifier=ctx.metadata["verifier"], proposed_model=proposed, diverted=chosen != proposed, context=ctx)
        ctx.metadata["dispatch"] = detail
        self.router.table[program_id] = b
        if self.scheduler is not None:
            try:
                b.admission_wait_s = await self.scheduler.admit(program_id, self.backend_of[chosen], tokens, disconnected)
            except BaseException:
                self.router.table.pop(program_id, None)
                raise
        b.admitted_at = time.time()
        return b

    def _backend_estimates(self, est: Mapping[str, Optional[float]]) -> Dict[str, Optional[float]]:
        """Per-backend accuracy estimate for the dispatch rule: the best estimate among the models it serves."""
        out: Dict[str, Optional[float]] = {}
        for backend, ms in self.models_on.items():
            vals = [est[m] for m in ms if est.get(m) is not None]
            out[backend] = max(vals) if vals else None
        return out

    def _model_on(self, backend: str, est: Mapping[str, Optional[float]]) -> str:
        """The model a diverted program runs on: the best-estimated one on ``backend``, else the first listed there."""
        ms = self.models_on[backend]
        scored = [m for m in ms if est.get(m) is not None]
        return max(scored, key=lambda m: est[m]) if scored else ms[0]

    def lookup(self, program_id: str) -> Optional[Binding]:
        return self.router.lookup(program_id)

    def observe_request(self, program_id: str, prompt_tokens: int = 0, completion_tokens: int = 0, cost: float = 0.0) -> None:
        """Account one LLM request of the program (the proxy calls this after each response)."""
        b = self.router.lookup(program_id)
        if b is None:
            return
        b.requests += 1
        b.prompt_tokens += int(prompt_tokens)
        b.completion_tokens += int(completion_tokens)
        b.cost += float(cost)
        ctx_len = int(prompt_tokens) + int(completion_tokens)
        b.peak_context_tokens = max(b.peak_context_tokens, ctx_len)
        if self.scheduler is not None:
            self.scheduler.update(program_id, ctx_len)

    def complete(self, program_id: str, outcome: Optional[ProgramOutcome] = None, **fields) -> Optional["asyncio.Task[float]"]:
        """Program finished: release its slot now, verify and learn later."""
        b = self.router.lookup(program_id)
        if self.scheduler is not None and b is not None:
            self.scheduler.release(program_id, b.peak_context_tokens or None)
        return self.router.complete(program_id, outcome, **fields)

    def release(self, program_id: str) -> bool:
        """Drop a program without feedback."""
        if self.scheduler is not None:
            self.scheduler.release(program_id)
        return self.router.release(program_id)

    def snapshot(self) -> Dict[str, Any]:
        return {"router": self.router.state(), "scheduler": None if self.scheduler is None else self.scheduler.snapshot(), "backend_of": self.backend_of, "t": time.time()}
