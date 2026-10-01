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
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

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
        """
        self.router = router
        self.scheduler = scheduler
        self.backend_of = dict(backend_of or {m: m for m in router.models})
        self.model_of = {b: m for m, b in self.backend_of.items()}
        if scheduler is not None:
            missing = [b for b in self.backend_of.values() if b not in scheduler.ledgers]
            if missing:
                raise ValueError(f"DISC has no ledger for backends {missing}")

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
    async def bind(self, program_id: str, prompt: str = "", metadata: Optional[Dict[str, Any]] = None, tokens: int = 0, disconnected: Optional[Callable[[], Awaitable[bool]]] = None) -> Binding:
        """Route, dispatch and admit a new program.  Idempotent for a known program.

        Blocks while DISC holds the program at the gate; the returned binding's
        ``admission_wait_s`` says for how long.
        """
        b = self.router.lookup(program_id)
        if b is not None:
            return b
        ctx = self.router.context(program_id, prompt, metadata)
        proposed = self.router.propose(ctx)
        model, detail = proposed, {}
        if self.scheduler is not None:
            est = self.router.estimates(ctx)
            backend, detail = self.scheduler.dispatch(self.backend_of[proposed], {self.backend_of[m]: e for m, e in est.items()})
            model = self.model_of[backend]
        b = Binding(program_id=program_id, model=model, verifier=ctx.metadata["verifier"], proposed_model=proposed, diverted=model != proposed, context=ctx)
        b.context.metadata["dispatch"] = detail
        self.router.table[program_id] = b
        if self.scheduler is not None:
            try:
                b.admission_wait_s = await self.scheduler.admit(program_id, self.backend_of[model], tokens, disconnected)
            except BaseException:
                self.router.table.pop(program_id, None)
                raise
        b.admitted_at = time.time()
        return b

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
