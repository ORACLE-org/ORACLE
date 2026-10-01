"""Admission control: an async FIFO gate in front of each ledger.

``await gate.admit(pid)`` blocks until the ledger has room, reserves the
program's expected peak, and returns the seconds waited.  Waiters are
re-checked on every release and on a timer, so a stuck client cannot wedge the
queue.  Pure asyncio; the only shared state is the ledger.
"""
from __future__ import annotations

import asyncio
import time
from collections import Counter, deque
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional

from .ledger import ReservationLedger


class ClientGone(Exception):
    """Raised from :meth:`AdmissionGate.admit` when the waiting client disconnected."""


class AdmissionGate:
    def __init__(self, ledger: ReservationLedger, poll_s: float = 1.0, max_wait_s: float = 0.0, clock: Callable[[], float] = time.time) -> None:
        if not float(poll_s) > 0:
            raise ValueError("poll_s must be positive")
        self.ledger = ledger
        self.poll_s = float(poll_s)
        self.max_wait_s = float(max_wait_s)  # 0 = wait forever
        self.clock = clock
        self.queue: List[Dict[str, Any]] = []
        self.waits: Deque[float] = deque(maxlen=10_000)  # bounded: the server runs for weeks
        self.stats = Counter()

    @property
    def queued(self) -> int:
        return len(self.queue)

    def wait_forecast(self, extra_ahead: int = 0) -> float:
        """Forecast wait for a program arriving now (``extra_ahead``: programs bound but not yet here)."""
        return self.ledger.wait_forecast(len(self.queue) + extra_ahead)

    async def admit(self, program_id: str, tokens: int = 0, disconnected: Optional[Callable[[], Awaitable[bool]]] = None) -> float:
        """Block until the program may start; reserve its peak; return the wait in seconds."""
        ent = {"pid": program_id, "t": self.clock(), "ev": asyncio.Event(), "tokens": int(tokens), "admitted": False}
        self.queue.append(ent)
        self.stats["requests"] += 1
        self.pump()
        try:
            while not ent["ev"].is_set():
                if disconnected is not None and await disconnected():
                    raise ClientGone(program_id)
                if self.max_wait_s and self.clock() - ent["t"] >= self.max_wait_s:
                    self.stats["forced"] += 1
                    self._grant(ent)
                    break
                await self._wait(ent["ev"], self.poll_s)
                if not ent["ev"].is_set():  # timer, not a grant: re-check the ledger (capacity may have been refreshed)
                    self.pump()
        except BaseException:
            if ent in self.queue:
                self.queue.remove(ent)
            if ent["admitted"]:  # granted in the same instant: give the slot back and wake the next waiter now
                self.ledger.release(program_id, learn=False)
                self.pump()
            self.stats["gone"] += 1
            raise
        waited = self.clock() - ent["t"]
        self.waits.append(waited)
        return waited

    @staticmethod
    async def _wait(ev: asyncio.Event, timeout: float) -> None:
        """``ev.wait()`` bounded by ``timeout``.  Unlike ``asyncio.wait_for`` on Python < 3.12 it never swallows a
        cancellation that arrives in the same instant as the grant, so a cancelled waiter always gives its slot back."""
        waiter = asyncio.ensure_future(ev.wait())
        try:
            await asyncio.wait({waiter}, timeout=timeout)
        finally:
            waiter.cancel()

    def _grant(self, ent: Dict[str, Any]) -> None:
        if ent in self.queue:
            self.queue.remove(ent)
        ent["admitted"] = True
        self.ledger.reserve(ent["pid"], ent["tokens"])
        self.stats["admitted"] += 1
        ent["ev"].set()

    def pump(self) -> None:
        """Strict FIFO: admit from the head while the next program fits."""
        while self.queue and self.ledger.fits():
            self._grant(self.queue[0])

    def release(self, program_id: str, peak_tokens: Optional[int] = None) -> bool:
        """Program finished: free its reservation and wake the queue."""
        ok = self.ledger.release(program_id, peak_tokens)
        for e in [e for e in self.queue if e["pid"] == program_id]:  # released while still waiting
            self.queue.remove(e)
            e["ev"].set()
        self.pump()
        return ok

    def snapshot(self) -> Dict[str, Any]:
        w = sorted(self.waits)
        q = (lambda p: w[min(len(w) - 1, int(p * len(w)))]) if w else (lambda p: 0.0)
        return {"queued": len(self.queue), "wait_forecast_s": round(self.wait_forecast(), 1), "wait_p50_s": round(q(0.5), 1), "wait_p95_s": round(q(0.95), 1), "wait_max_s": round(w[-1], 1) if w else 0.0, **{k: v for k, v in self.stats.items()}, "ledger": self.ledger.snapshot()}
