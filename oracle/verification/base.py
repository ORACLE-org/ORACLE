"""Verifiers score a finished program.  They run *off the critical path*: the
program's slot is released first, the verifier runs afterwards, and the model
selector receives the reward whenever it arrives (paper Sec. 4.2.2)."""
from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, Optional, Union

from ..types import ProgramOutcome


class Verifier(ABC):
    """Return a score in [0, 1] for a completed program."""

    name: str = "verifier"

    @abstractmethod
    async def verify(self, outcome: ProgramOutcome) -> float: ...

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "type": type(self).__name__}


ScoreFn = Callable[[ProgramOutcome], Union[float, Awaitable[float]]]


class CallableVerifier(Verifier):
    """Wrap a sync or async function ``fn(outcome) -> float``."""

    def __init__(self, name: str, fn: ScoreFn) -> None:
        self.name, self.fn = name, fn

    async def verify(self, outcome: ProgramOutcome) -> float:
        r = self.fn(outcome)
        if inspect.isawaitable(r):
            r = await r
        return float(r)


class ReportedVerifier(Verifier):
    """Trust the score the harness put in ``outcome.success`` (``default`` if absent).

    The right choice when the agent framework already grades its own tasks.
    """

    def __init__(self, name: str = "reported", default: float = 0.0) -> None:
        self.name, self.default = name, default

    async def verify(self, outcome: ProgramOutcome) -> float:
        return float(outcome.success) if outcome.success is not None else self.default


class CommandVerifier(Verifier):
    """Run a shell command; exit code 0 means success.

    The command is a template with ``{program_id}`` and any ``outcome.payload``
    keys, e.g. ``"docker exec {container} pytest -q"``.  Set ``parse_score`` to
    read a float from the last line of stdout instead of using the exit code.
    """

    def __init__(self, name: str, command: str, timeout_s: float = 1800.0, parse_score: bool = False, cwd: Optional[str] = None) -> None:
        self.name, self.command, self.timeout_s, self.parse_score, self.cwd = name, command, timeout_s, parse_score, cwd

    async def verify(self, outcome: ProgramOutcome) -> float:
        cmd = self.command.format(program_id=outcome.program_id, **outcome.payload)
        proc = await asyncio.create_subprocess_shell(cmd, cwd=self.cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=self.timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            return 0.0
        if self.parse_score:
            try:
                return float(out.decode().strip().splitlines()[-1])
            except Exception:
                return 0.0
        return 1.0 if proc.returncode == 0 else 0.0


class HTTPVerifier(Verifier):
    """POST the outcome as JSON to ``url`` and read ``{"score": float}`` back.

    Lets you keep a verifier (a patch sandbox, a state checker) as its own service.
    """

    def __init__(self, name: str, url: str, timeout_s: float = 1800.0, headers: Optional[Dict[str, str]] = None) -> None:
        self.name, self.url, self.timeout_s, self.headers = name, url, timeout_s, headers or {}

    async def verify(self, outcome: ProgramOutcome) -> float:
        import httpx

        body = {"program_id": outcome.program_id, "cost": outcome.cost, "success": outcome.success, "payload": outcome.payload}
        async with httpx.AsyncClient(timeout=self.timeout_s) as c:
            r = await c.post(self.url, json=body, headers=self.headers)
            r.raise_for_status()
            return float(r.json()["score"])


class LLMJudgeVerifier(Verifier):
    """Ask an OpenAI-compatible model to grade the program's final answer.

    ``outcome.payload`` should contain ``task`` and ``answer`` (and optionally
    ``reference``).  The judge is asked for a number in [0, 1].
    """

    PROMPT = (
        "Grade the answer to the task on a scale from 0 to 1, where 1 is fully correct.\n"
        "Task:\n{task}\n\nAnswer:\n{answer}\n\n{reference}Reply with only the number."
    )

    def __init__(self, name: str, url: str, model: str, api_key: str = "EMPTY", timeout_s: float = 120.0) -> None:
        self.name, self.url, self.model, self.api_key, self.timeout_s = name, url.rstrip("/"), model, api_key, timeout_s

    async def verify(self, outcome: ProgramOutcome) -> float:
        import httpx

        p = outcome.payload
        ref = f"Reference answer:\n{p['reference']}\n\n" if p.get("reference") else ""
        prompt = self.PROMPT.format(task=p.get("task", ""), answer=p.get("answer", ""), reference=ref)
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 8, "temperature": 0}
        async with httpx.AsyncClient(timeout=self.timeout_s) as c:
            r = await c.post(f"{self.url}/chat/completions", json=body, headers={"Authorization": f"Bearer {self.api_key}"})
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
        import re

        m = re.search(r"[01](?:\.\d+)?", text)
        return min(max(float(m.group()), 0.0), 1.0) if m else 0.0
