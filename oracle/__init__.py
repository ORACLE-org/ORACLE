"""ORACLE: concurrency-aware routing and scheduling for agentic programs.

Two independent pieces:

* the router (:mod:`oracle.routing` with :mod:`oracle.verification`) - bind
  each program to a model with a pluggable selector (LinUCB by default,
  ACRouter, RouteLLM, your own), choose a type-matched verifier per program,
  run it off the critical path and learn from the delayed verdict;
* the scheduler (:mod:`oracle.scheduling`) - DISC: a reservation ledger +
  admission gate per backend and a dispatch rule that trades a little
  accuracy for a shorter wait.

:class:`Oracle` composes them; ``oracle serve`` exposes the result as an
OpenAI-compatible proxy with a dashboard.
"""
from .types import Binding, ProgramOutcome, RoutingContext
from .routing import Router, ModelSelector, register_selector, make_selector, available_selectors, Reward
from .verification import Verifier, CallableVerifier, PrototypeVerifierSelector, FeedbackLoop
from .scheduling import DISC, ReservationLedger, AdmissionGate, DispatchPolicy
from .core import Oracle
from .config import OracleConfig

__version__ = "0.1.0"
__all__ = [
    "Binding", "ProgramOutcome", "RoutingContext",
    "Router", "ModelSelector", "register_selector", "make_selector", "available_selectors", "Reward",
    "Verifier", "CallableVerifier", "PrototypeVerifierSelector", "FeedbackLoop",
    "DISC", "ReservationLedger", "AdmissionGate", "DispatchPolicy",
    "Oracle", "OracleConfig", "__version__",
]
