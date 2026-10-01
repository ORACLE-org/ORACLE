"""Shared data types.

The whole library is organised around one idea from the paper: the *program*
(one agentic task, many LLM requests) is the unit of routing, concurrency and
feedback.  ORACLE keeps a lookup table that binds every program to a model and
a verifier once, at its first request:

    R = <ID, P, M, V>

``Binding`` below is one row of that table.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class RoutingContext:
    """What a model selector sees when it picks a model for a new program."""

    program_id: str
    prompt: str = ""
    """Text of the program's initial request (first user message)."""
    features: Optional[np.ndarray] = None
    """Feature vector for contextual learners. Filled by a ``FeatureExtractor``."""
    task_type: Optional[str] = None
    """Task type inferred by the verifier selector (e.g. ``"swe"``)."""
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Anything else the caller wants to pass along (user id, budget, ...)."""


@dataclass
class ProgramOutcome:
    """What the harness reports when a program finishes."""

    program_id: str
    cost: float = 0.0
    """Final cost of the program in the unit of ``Reward.cost_ref`` (dollars by default)."""
    success: Optional[float] = None
    """Score in [0, 1] if the harness already knows it. ``None`` lets the bound verifier decide."""
    payload: Dict[str, Any] = field(default_factory=dict)
    """Whatever the verifier needs: final answer, repo path, container id, environment state, ..."""
    wall_time_s: Optional[float] = None


@dataclass
class Binding:
    """One row of the lookup table: program -> (model, verifier, backend)."""

    program_id: str
    model: str
    """Model (and backend) the program is dispatched to."""
    verifier: Optional[str] = None
    """Name of the verifier that will score the program when it completes."""
    proposed_model: Optional[str] = None
    """What the model selector proposed before DISC's dispatch decision."""
    diverted: bool = False
    """True if DISC moved the program to a backend with a shorter forecast wait."""
    context: Optional[RoutingContext] = None
    created_at: float = field(default_factory=time.time)
    admitted_at: Optional[float] = None
    admission_wait_s: float = 0.0
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    """Running cost accumulated by the proxy (when it has prices)."""
    peak_context_tokens: int = 0
    completed_at: Optional[float] = None
    reward: Optional[float] = None
    score: Optional[float] = None

    @property
    def backend(self) -> str:
        return self.model

    def to_dict(self) -> Dict[str, Any]:
        return {
            "program_id": self.program_id,
            "model": self.model,
            "verifier": self.verifier,
            "proposed_model": self.proposed_model,
            "diverted": self.diverted,
            "task_type": self.context.task_type if self.context else None,
            "created_at": self.created_at,
            "admitted_at": self.admitted_at,
            "admission_wait_s": round(self.admission_wait_s, 3),
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost": round(self.cost, 6),
            "peak_context_tokens": self.peak_context_tokens,
            "completed_at": self.completed_at,
            "score": self.score,
            "reward": self.reward,
        }
