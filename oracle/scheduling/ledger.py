"""The reservation ledger (paper Sec. 4.3): one per backend.

When a program is admitted the ledger reserves its *expected peak* KV
footprint; the reservation is released when the program finishes.  A program
is admitted only when the unreserved capacity can hold one more expected peak.
The expected peak ``c_hat`` starts from a prior and is refined online from the
peaks of finished programs.  Nothing here is async or HTTP-aware, so the same
class can sit inside any scheduler or a unit test.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Any, Callable, Dict, Optional


class ReservationLedger:
    """Per-backend ledger of admitted programs.

    Args:
        capacity_tokens: KV-cache capacity in tokens (update later via :attr:`capacity_tokens`).
        rho: fraction of the capacity the ledger may commit (headroom for decode growth).
        peak_prior: prior expected peak context of a program (tokens).
        prior_weight: pseudo-count of the prior; the estimate moves after this many finished programs.
        floor: an admitted program counts at least ``floor * c_hat`` even while its context is small.
        per_program_overhead: fixed tokens charged per running program (hybrid-attention state, decode buffer).
        duration_prior_s: prior mean run time of a program (seconds), used by the wait forecast
            until enough programs have finished.
        max_programs: optional hard cap on admitted programs (``0`` = no cap).
    """

    def __init__(
        self,
        capacity_tokens: int = 0,
        rho: float = 0.85,
        peak_prior: float = 20000.0,
        prior_weight: int = 5,
        floor: float = 1.0,
        per_program_overhead: int = 0,
        duration_prior_s: float = 30.0,
        max_programs: int = 0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.capacity_tokens = int(capacity_tokens)
        self.rho = float(rho)
        self.peak_prior, self.prior_weight = float(peak_prior), int(prior_weight)
        self.floor = float(floor)
        self.overhead = int(per_program_overhead)
        self.duration_prior_s = float(duration_prior_s)
        self.max_programs = int(max_programs)
        self.clock = clock
        self.admitted: Dict[str, Dict[str, Any]] = {}  # pid -> {"tokens": current context, "peak": peak so far, "t": admit time}
        self.finished_peak_sum, self.finished_n = 0.0, 0
        self.dur_sum, self.dur_n = 0.0, 0
        self.measured_usage: Optional[float] = None
        self.stats = Counter()

    # --- learned quantities -------------------------------------------------
    @property
    def c_hat(self) -> float:
        """Expected peak context of a program (tokens)."""
        return (self.prior_weight * self.peak_prior + self.finished_peak_sum) / (self.prior_weight + self.finished_n)

    @property
    def d_hat(self) -> float:
        """Expected run time of a program (seconds)."""
        return (self.prior_weight * self.duration_prior_s + self.dur_sum) / (self.prior_weight + self.dur_n)

    @property
    def budget(self) -> float:
        return self.rho * self.capacity_tokens

    # --- reservations -------------------------------------------------------
    def reserved(self) -> float:
        """Tokens the ledger is committed to for the admitted programs."""
        c = self.floor * self.c_hat
        return sum(max(p["tokens"], c) + self.overhead for p in self.admitted.values())

    def fits(self) -> bool:
        """May one more program start now?"""
        if self.max_programs and len(self.admitted) >= self.max_programs:
            return False
        if not self.admitted:
            return True  # an empty backend always admits one program
        if self.capacity_tokens <= 0:
            return True  # capacity unknown: do not block
        return self.reserved() + self.c_hat + self.overhead <= self.budget

    def free_slots(self) -> int:
        """How many more programs the ledger would admit right now."""
        if self.max_programs:
            return max(0, self.max_programs - len(self.admitted))
        if self.capacity_tokens <= 0:
            return 1 if not self.admitted else 10 ** 6
        per = self.c_hat + self.overhead
        room = self.budget - self.reserved()
        k = int(room // per) if per > 0 else 0
        return max(k, 1) if not self.admitted else max(k, 0)

    def reserve(self, program_id: str, tokens: int = 0) -> None:
        """Admit ``program_id``: reserve its expected peak from this instant."""
        self.admitted[program_id] = {"tokens": int(tokens), "peak": int(tokens), "t": self.clock()}
        self.stats["reserved"] += 1

    def update(self, program_id: str, tokens: int) -> None:
        """Record the program's current context length (after each request)."""
        p = self.admitted.get(program_id)
        if p is not None:
            p["tokens"] = int(tokens)
            p["peak"] = max(p["peak"], int(tokens))

    def release(self, program_id: str, peak_tokens: Optional[int] = None) -> bool:
        """The program finished: free its reservation and learn from its peak and duration."""
        p = self.admitted.pop(program_id, None)
        if p is None:
            return False
        peak = int(peak_tokens) if peak_tokens is not None else p["peak"]
        if peak > 0:
            self.finished_peak_sum += peak
            self.finished_n += 1
        self.dur_sum += self.clock() - p["t"]
        self.dur_n += 1
        self.stats["released"] += 1
        return True

    def observe_usage(self, used_fraction: Optional[float], capacity_tokens: Optional[int] = None) -> None:
        """Feed a measurement from a :class:`CapacitySource`."""
        if capacity_tokens:
            self.capacity_tokens = int(capacity_tokens)
        if used_fraction is not None:
            self.measured_usage = float(used_fraction)

    # --- forecast -----------------------------------------------------------
    def wait_forecast(self, queued: int = 0) -> float:
        """Forecast admission wait (s) of one more program with ``queued`` programs ahead of it.

        0 if it would start now; otherwise the drain estimate
        ``(queued - free_slots + 1) * d_hat / admitted``.  Every quantity is measured.
        """
        k = self.free_slots()
        if queued < k:
            return 0.0
        return (queued - k + 1) * self.d_hat / max(len(self.admitted), 1)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "capacity_tokens": self.capacity_tokens,
            "rho": self.rho,
            "admitted": len(self.admitted),
            "reserved_tokens": int(self.reserved()),
            "budget_tokens": int(self.budget),
            "free_slots": self.free_slots(),
            "c_hat": round(self.c_hat, 1),
            "d_hat": round(self.d_hat, 1),
            "finished": self.finished_n,
            "measured_usage": self.measured_usage,
            "programs": {pid: {"tokens": p["tokens"], "peak": p["peak"], "age_s": round(self.clock() - p["t"], 1)} for pid, p in self.admitted.items()},
            **{k: v for k, v in self.stats.items()},
        }
