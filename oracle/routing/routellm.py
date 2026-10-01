"""RouteLLM adapter (Ong et al. 2024): a pre-trained strong/weak router.

Requires ``pip install routellm``.  RouteLLM does not learn online, so
``update`` only counts; ORACLE's verifiers still run and the dashboard still
shows the realised accuracy of each model.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from ..types import RoutingContext
from .base import ModelSelector
from .registry import register_selector


@register_selector("routellm")
class RouteLLMSelector(ModelSelector):
    def __init__(self, models: Sequence[str], router: str = "mf", threshold: float = 0.11593, strong: Optional[str] = None, weak: Optional[str] = None, **controller_kwargs) -> None:
        super().__init__(models)
        if len(self.models) != 2 and (strong is None or weak is None):
            raise ValueError("RouteLLM routes between two models; pass strong= and weak= when more are configured")
        self.strong = strong or self.models[0]
        self.weak = weak or self.models[-1]
        self.router, self.threshold = router, float(threshold)
        try:
            from routellm.controller import Controller
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install routellm   (see https://github.com/lm-sys/RouteLLM)") from e
        self.controller = Controller(routers=[router], strong_model=self.strong, weak_model=self.weak, **controller_kwargs)

    def select(self, ctx: RoutingContext) -> str:
        self.n_select += 1
        return self.controller.route(prompt=ctx.prompt, router=self.router, threshold=self.threshold)

    def update(self, ctx: RoutingContext, model: str, reward: float) -> None:
        self.n_update += 1

    def state(self) -> Dict[str, Any]:
        s = super().state()
        s.update({"router": self.router, "threshold": self.threshold, "strong": self.strong, "weak": self.weak})
        return s
