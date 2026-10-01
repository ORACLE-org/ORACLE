"""ORACLE's router as a LiteLLM custom routing strategy.

LiteLLM's ``Router`` picks a deployment per request; this strategy makes the
pick once per program (``metadata["program_id"]``) through an ORACLE
:class:`~oracle.routing.Router`, so every request of a program lands on the
same model and verified outcomes flow back as delayed feedback::

    import litellm
    from oracle.routing import Router
    from oracle.integrations.litellm import OracleRoutingStrategy

    router = Router(["gpt-strong", "gpt-weak"], "linucb", verifiers=...)
    lr = litellm.Router(model_list=[{"model_name": "gpt-strong", ...}, {"model_name": "gpt-weak", ...}])
    lr.set_custom_routing_strategy(OracleRoutingStrategy(router))

    await lr.acompletion(model="auto", messages=msgs, metadata={"program_id": "task-1"})
    ...
    router.complete("task-1", ProgramOutcome(program_id="task-1", success=1.0, cost=0.12))
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..routing.router import Router
from ..server.fields import first_user_text

try:  # pragma: no cover - optional dependency
    from litellm.router import CustomRoutingStrategyBase
except Exception:  # noqa: BLE001

    class CustomRoutingStrategyBase:  # type: ignore[no-redef]
        pass


class OracleRoutingStrategy(CustomRoutingStrategyBase):
    def __init__(self, router: Router, program_id_key: str = "program_id") -> None:
        self.router = router
        self.key = program_id_key

    def _pick(self, healthy: List[Dict[str, Any]], messages, request_kwargs) -> Dict[str, Any]:
        meta = (request_kwargs or {}).get("metadata") or {}
        pid = str(meta.get(self.key) or (request_kwargs or {}).get(self.key) or "default")
        b = self.router.lookup(pid) or self.router.bind(pid, first_user_text({"messages": messages or []}), dict(meta))
        for d in healthy:
            if d.get("model_name") == b.model:
                return d
        raise ValueError(f"no healthy LiteLLM deployment named {b.model!r}; have {[d.get('model_name') for d in healthy]}")

    async def async_get_available_deployment(self, model: str, messages: Optional[List[Dict[str, str]]] = None, input=None, specific_deployment: Optional[bool] = False, request_kwargs: Optional[Dict] = None):
        healthy = self._healthy(model)
        return self._pick(healthy, messages, request_kwargs)

    def get_available_deployment(self, model: str, messages: Optional[List[Dict[str, str]]] = None, input=None, specific_deployment: Optional[bool] = False, request_kwargs: Optional[Dict] = None):
        healthy = self._healthy(model)
        return self._pick(healthy, messages, request_kwargs)

    def _healthy(self, model: str) -> List[Dict[str, Any]]:
        lr = getattr(self, "router_instance", None) or getattr(self, "litellm_router", None)
        deployments = getattr(lr, "healthy_deployments", None) or getattr(lr, "model_list", None) or []
        return [d for d in deployments if d.get("model_name") in self.router.models]
