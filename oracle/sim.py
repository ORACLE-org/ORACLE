"""A small synthetic workload for demos, examples and tests.

Task families have a hidden per-model success probability and cost, so a
learning selector should discover that the cheap model is enough for some
families and route the rest to the strong model.  No GPUs involved.
"""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .core import Oracle
from .routing.router import Router
from .types import ProgramOutcome

FAMILIES: Dict[str, Dict] = {
    "swe": {"prompts": ["fix the failing test in {repo} so that pytest passes", "patch the bug in {repo}: the CLI crashes on empty input", "implement the missing function in {repo} described in the issue"], "success": {"strong": 0.62, "weak": 0.28}, "cost": {"strong": 0.14, "weak": 0.05}, "peak": 60000},
    "tau2": {"prompts": ["book a flight from {a} to {b} for next Tuesday and add a checked bag", "change my hotel reservation in {a} to a later check-in", "cancel the order and refund the customer, then confirm by email"], "success": {"strong": 0.78, "weak": 0.70}, "cost": {"strong": 0.06, "weak": 0.02}, "peak": 20000},
    "terminal": {"prompts": ["in the container, compile {repo} and run its benchmark", "set up a cron job that rotates logs in {repo}", "find and kill the process holding port 8080 in the sandbox"], "success": {"strong": 0.55, "weak": 0.40}, "cost": {"strong": 0.10, "weak": 0.04}, "peak": 40000},
}
_FILL = {"repo": ["repo-a", "proj-b", "lib-c"], "a": ["SFO", "JFK", "ORD"], "b": ["LHR", "CDG", "NRT"]}


@dataclass
class SimTask:
    program_id: str
    family: str
    prompt: str


@dataclass
class SimResult:
    programs: int = 0
    success: float = 0.0
    cost: float = 0.0
    per_model: Dict[str, int] = field(default_factory=dict)
    diverted: int = 0
    waits: List[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.success / self.programs if self.programs else 0.0

    def summary(self) -> str:
        w = sorted(self.waits)
        p95 = w[int(0.95 * (len(w) - 1))] if w else 0.0
        return f"programs={self.programs} accuracy={100 * self.accuracy:.1f}% cost=${self.cost:.2f} per_model={self.per_model} diverted={self.diverted} wait_p95={p95:.2f}s"


def make_tasks(n: int, seed: int = 0) -> List[SimTask]:
    rng = random.Random(seed)
    out = []
    fams = list(FAMILIES)
    for i in range(n):
        f = rng.choice(fams)
        p = rng.choice(FAMILIES[f]["prompts"]).format(**{k: rng.choice(v) for k, v in _FILL.items()})
        out.append(SimTask(f"sim-{i}", f, p))
    return out


def exemplars() -> Dict[str, List[str]]:
    return {f: [p.format(repo="x", a="SFO", b="LHR") for p in d["prompts"]] for f, d in FAMILIES.items()}


def family_of(task: SimTask) -> str:
    return task.family


async def run(system, tasks: List[SimTask], concurrency: int = 8, seed: int = 0, time_scale: float = 0.0) -> SimResult:
    """Run the tasks through a :class:`Router` or an :class:`Oracle`.

    ``time_scale`` > 0 sleeps a scaled "execution time" per program so that DISC's
    admission gate has something to do.
    """
    rng = random.Random(seed)
    res = SimResult()
    sem = asyncio.Semaphore(concurrency)
    is_oracle = isinstance(system, Oracle)

    async def one(t: SimTask):
        async with sem:
            fam = FAMILIES[t.family]
            if is_oracle:
                b = await system.bind(t.program_id, t.prompt, {"family": t.family})
            else:
                b = system.bind(t.program_id, t.prompt, {"family": t.family})
            model = b.model
            succ = 1.0 if rng.random() < fam["success"][model] else 0.0
            cost = fam["cost"][model] * rng.uniform(0.7, 1.3)
            peak = int(fam["peak"] * rng.uniform(0.5, 1.5))
            if time_scale > 0:
                await asyncio.sleep(time_scale * rng.uniform(0.5, 1.5))
            if is_oracle:
                system.observe_request(t.program_id, peak, 200, cost)
            res.programs += 1
            res.success += succ
            res.cost += cost
            res.per_model[model] = res.per_model.get(model, 0) + 1
            res.diverted += int(b.diverted)
            res.waits.append(b.admission_wait_s)
            task = system.complete(t.program_id, ProgramOutcome(program_id=t.program_id, cost=cost, success=succ, payload={"family": t.family}))
            if task is not None:
                await task

    await asyncio.gather(*(one(t) for t in tasks))
    return res
