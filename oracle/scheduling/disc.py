"""DISC: the DIspatch SCheduler.  One ledger + gate per backend and the
dispatch rule on top.

DISC is independent of the router and of any serving stack.  Use it alone to
add task-boundary admission control to an existing proxy, or let
:class:`oracle.Oracle` wire it to the router.

Minimal standalone use::

    disc = DISC(backends={"gpu-a": 120_000, "gpu-b": 120_000})
    wait = await disc.admit("task-1", "gpu-a")      # blocks until gpu-a has room
    ... run the task ...
    disc.release("task-1", peak_tokens=31_000)      # frees the reservation, learns the peak
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Tuple, Union

from .admission import AdmissionGate, ClientGone
from .capacity import CapacitySource
from .dispatch import DispatchPolicy
from .ledger import ReservationLedger

logger = logging.getLogger(__name__)


class DISC:
    """Args:
        backends: ``{name: capacity_tokens}`` or ``{name: ReservationLedger}`` or
            ``{name: CapacitySource}`` (capacity is then polled).
        policy: the dispatch rule; ``None`` means never divert (admission only).
        ledger_kwargs: defaults for ledgers built from an integer capacity
            (``rho``, ``peak_prior``, ``per_program_overhead``, ...).
        poll_s: interval for polling capacity sources.
    """

    def __init__(
        self,
        backends: Mapping[str, Union[int, ReservationLedger, CapacitySource]],
        policy: Optional[DispatchPolicy] = DispatchPolicy(),
        ledger_kwargs: Optional[Dict[str, Any]] = None,
        poll_s: float = 2.0,
        max_wait_s: float = 0.0,
    ) -> None:
        if not backends:
            raise ValueError("DISC needs at least one backend")
        self.policy = policy
        self.ledgers: Dict[str, ReservationLedger] = {}
        self.gates: Dict[str, AdmissionGate] = {}
        self.sources: Dict[str, CapacitySource] = {}
        kw = dict(ledger_kwargs or {})
        for name, spec in backends.items():
            if isinstance(spec, ReservationLedger):
                ledger = spec
            elif isinstance(spec, CapacitySource):
                ledger = ReservationLedger(0, **kw)
                self.sources[name] = spec
            else:
                ledger = ReservationLedger(int(spec), **kw)
            self.ledgers[name] = ledger
            self.gates[name] = AdmissionGate(ledger, max_wait_s=max_wait_s)
        self.pending: Dict[str, int] = {n: 0 for n in self.ledgers}
        """Programs dispatched to a backend that have not reached its gate yet."""
        self.where: Dict[str, str] = {}
        self.poll_s = float(poll_s)
        self._poll_task: Optional[asyncio.Task] = None
        self.n_dispatch = 0
        self.n_diverted = 0

    # --- lifecycle ----------------------------------------------------------
    async def start(self) -> None:
        """Start polling capacity sources (no-op when none were given)."""
        if self.sources and self._poll_task is None:
            await self.refresh()
            self._poll_task = asyncio.create_task(self._poll())

    async def stop(self) -> None:
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

    async def refresh(self) -> None:
        for name, src in self.sources.items():
            try:
                cap, used = await src.fetch()
                self.ledgers[name].observe_usage(used, cap)
            except Exception as e:  # keep running on a flaky metrics endpoint
                logger.warning("capacity fetch failed for %s: %s", name, e)
        for g in self.gates.values():
            g.pump()

    async def _poll(self) -> None:
        while True:
            await asyncio.sleep(self.poll_s)
            await self.refresh()

    # --- dispatch -----------------------------------------------------------
    def waits(self) -> Dict[str, float]:
        """Forecast admission wait per backend for a program arriving now."""
        return {n: g.wait_forecast(self.pending[n]) for n, g in self.gates.items()}

    def dispatch(self, proposed: str, estimates: Optional[Mapping[str, Optional[float]]] = None) -> Tuple[str, Dict[str, Any]]:
        """Apply the dispatch rule once, at the program boundary.

        Returns ``(backend, detail)``; the caller then ``admit``s on that backend.
        """
        if proposed not in self.ledgers:
            raise KeyError(f"unknown backend {proposed!r}; known: {list(self.ledgers)}")
        waits = self.waits()
        chosen, margins = (proposed, {}) if self.policy is None else self.policy.choose(proposed, waits, estimates or {})
        self.n_dispatch += 1
        if chosen != proposed:
            self.n_diverted += 1
        self.pending[chosen] += 1
        return chosen, {"waits": {k: round(v, 2) for k, v in waits.items()}, "margins": margins, "diverted": chosen != proposed}

    async def admit(self, program_id: str, backend: str, tokens: int = 0, disconnected: Optional[Callable[[], Awaitable[bool]]] = None) -> float:
        """Block until ``backend`` has room for the program; returns seconds waited."""
        if backend not in self.gates:
            raise KeyError(f"unknown backend {backend!r}")
        if self.pending[backend] > 0:
            self.pending[backend] -= 1
        self.where[program_id] = backend
        try:
            return await self.gates[backend].admit(program_id, tokens, disconnected)
        except BaseException:
            self.where.pop(program_id, None)
            raise

    def update(self, program_id: str, tokens: int) -> None:
        """Record the program's current context length."""
        b = self.where.get(program_id)
        if b is not None:
            self.ledgers[b].update(program_id, tokens)

    def release(self, program_id: str, peak_tokens: Optional[int] = None) -> bool:
        """Program finished (or vanished): free its reservation."""
        b = self.where.pop(program_id, None)
        if b is None:
            return False
        return self.gates[b].release(program_id, peak_tokens)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "policy": None if self.policy is None else {"beta": self.policy.beta, "w0": self.policy.w0},
            "dispatched": self.n_dispatch,
            "diverted": self.n_diverted,
            "backends": {n: {**g.snapshot(), "pending": self.pending[n]} for n, g in self.gates.items()},
        }


__all__ = ["DISC", "ClientGone"]
