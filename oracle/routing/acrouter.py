"""ACRouter-style selector (Zhou et al. 2026, "Agent-as-a-Router").

ACRouter keeps a memory of past tasks (embedding, chosen model, reward) and,
for a new task, retrieves the ``k`` most similar entries with similarity at
least ``min_sim`` and lets an orchestrator LLM choose the model given that
evidence.  This adapter reproduces that loop with ORACLE's program-level
feedback and works in two modes:

* ``orchestrator_url`` set: the released prompt format is sent to any
  OpenAI-compatible endpoint (vLLM, OpenAI, ...) and its answer is parsed;
* no endpoint: the *memory rule* is used, i.e. the model with the best mean
  reward among the retrieved neighbours (the first model when memory is empty
  or no neighbour passes the similarity floor).  Because the rule never tries
  a model it has no evidence for, ``explore`` (default 0.1) picks such a model
  with that probability; set it to 0 for the pure rule.

Only the memory rule is exercised in the unit tests; the LLM path is a thin
HTTP call.  Inside ORACLE the reward is the paper's delayed reward, so the
memory stores verified outcomes rather than the orchestrator's own guess.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..types import RoutingContext
from .base import ModelSelector
from .registry import register_selector

_PROMPT = """You are a routing orchestrator. Choose which model should solve the task.
Available models (strongest first): {models}
Cost weight: a cheaper model is preferred when past evidence shows it succeeds.

Similar past tasks and their outcomes:
{memory}

New task:
{task}

Answer with the model name only."""


@register_selector("acrouter")
class ACRouterSelector(ModelSelector):
    def __init__(
        self,
        models: Sequence[str],
        embedder=None,
        k: int = 10,
        min_sim: float = 0.5,
        orchestrator_url: Optional[str] = None,
        orchestrator_model: str = "",
        api_key: str = "EMPTY",
        timeout_s: float = 30.0,
        max_memory: int = 10000,
        explore: float = 0.1,
        seed: int = 0,
    ) -> None:
        super().__init__(models)
        if embedder is None:
            from ..verification.embeddings import HashingEmbedder

            embedder = HashingEmbedder()
        self.embedder = embedder
        self.k, self.min_sim = int(k), float(min_sim)
        self.orchestrator_url = orchestrator_url.rstrip("/") if orchestrator_url else None
        self.orchestrator_model, self.api_key, self.timeout_s = orchestrator_model, api_key, timeout_s
        self.max_memory = max_memory
        self.explore = float(explore)
        self.rng = np.random.default_rng(seed)
        self._emb: List[np.ndarray] = []
        self._mem: List[Dict[str, Any]] = []
        self.llm_calls = 0
        self.llm_failures = 0

    # --- memory -------------------------------------------------------------
    def _embed(self, ctx: RoutingContext) -> np.ndarray:
        if "acrouter_embedding" in ctx.metadata:
            return ctx.metadata["acrouter_embedding"]
        e = np.asarray(self.embedder.embed([ctx.prompt])[0], dtype=float)
        n = np.linalg.norm(e)
        ctx.metadata["acrouter_embedding"] = e / n if n > 0 else e
        return ctx.metadata["acrouter_embedding"]

    def neighbours(self, ctx: RoutingContext) -> List[Dict[str, Any]]:
        if not self._mem:
            return []
        e = self._embed(ctx)
        sims = np.stack(self._emb) @ e
        order = np.argsort(-sims)[: self.k]
        return [dict(self._mem[i], sim=float(sims[i])) for i in order if sims[i] >= self.min_sim]

    def memory_rule(self, nb: List[Dict[str, Any]]) -> str:
        if not nb:
            return self.models[0]
        means = {}
        for m in self.models:
            rs = [n["reward"] for n in nb if n["model"] == m]
            if rs:
                means[m] = sum(rs) / len(rs)
        if not means:
            return self.models[0]
        untried = [m for m in self.models if m not in means]
        if untried and self.explore > 0 and self.rng.random() < self.explore:
            return untried[int(self.rng.integers(len(untried)))]
        best = max(means.values())
        # ties go to the cheapest (last) model that reaches the best mean
        return [m for m in self.models if means.get(m) == best][-1]

    # --- orchestrator LLM ---------------------------------------------------
    def _ask_llm(self, ctx: RoutingContext, nb: List[Dict[str, Any]]) -> Optional[str]:
        import httpx

        mem = "\n".join(f"- [{n['model']}] reward={n['reward']:.2f} sim={n['sim']:.2f}: {n['prompt'][:200]}" for n in nb) or "(none)"
        body = {
            "model": self.orchestrator_model,
            "messages": [{"role": "user", "content": _PROMPT.format(models=", ".join(self.models), memory=mem, task=ctx.prompt[:2000])}],
            "max_tokens": 16,
            "temperature": 0,
        }
        self.llm_calls += 1
        try:
            r = httpx.post(f"{self.orchestrator_url}/chat/completions", json=body, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=self.timeout_s)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
        except Exception:
            self.llm_failures += 1
            return None
        for m in sorted(self.models, key=len, reverse=True):
            if re.search(re.escape(m), text, re.IGNORECASE):
                return m
        return None

    # --- contract -----------------------------------------------------------
    def select(self, ctx: RoutingContext) -> str:
        self.n_select += 1
        nb = self.neighbours(ctx)
        if self.orchestrator_url:
            m = self._ask_llm(ctx, nb)
            if m is not None:
                return m
        return self.memory_rule(nb)

    def update(self, ctx: RoutingContext, model: str, reward: float) -> None:
        self.n_update += 1
        self._check(model)
        self._emb.append(self._embed(ctx))
        self._mem.append({"prompt": ctx.prompt, "model": model, "reward": float(reward)})
        if len(self._mem) > self.max_memory:
            self._emb.pop(0)
            self._mem.pop(0)

    def estimate(self, ctx: RoutingContext, model: str) -> Optional[float]:
        rs = [n["reward"] for n in self.neighbours(ctx) if n["model"] == model]
        return sum(rs) / len(rs) if rs else None

    def state(self) -> Dict[str, Any]:
        s = super().state()
        s.update({"memory": len(self._mem), "k": self.k, "min_sim": self.min_sim, "orchestrator": self.orchestrator_url, "llm_calls": self.llm_calls, "llm_failures": self.llm_failures})
        return s
