"""Where a ledger learns a backend's KV-cache capacity and current usage.

A :class:`CapacitySource` answers two questions for one backend: how many
tokens fit in its KV cache, and what fraction is in use right now.  DISC only
needs the first; the second is shown on the dashboard and lets the ledger
sanity-check its reservations.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class CapacitySource(ABC):
    @abstractmethod
    async def fetch(self) -> Tuple[Optional[int], Optional[float]]:
        """Return ``(capacity_tokens, used_fraction)``; either may be ``None`` if unknown."""


class StaticCapacity(CapacitySource):
    """A fixed capacity you set by hand (tokens)."""

    def __init__(self, capacity_tokens: int) -> None:
        self.capacity_tokens = int(capacity_tokens)

    async def fetch(self) -> Tuple[Optional[int], Optional[float]]:
        return self.capacity_tokens, None


def _strip_v1(url: str) -> str:
    url = url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


class VLLMCapacity(CapacitySource):
    """Parse vLLM's Prometheus ``/metrics``: ``vllm:cache_config_info`` and ``vllm:gpu_cache_usage_perc``."""

    def __init__(self, url: str, timeout_s: float = 5.0) -> None:
        self.url, self.timeout_s = _strip_v1(url), timeout_s

    @staticmethod
    def parse(text: str) -> Tuple[Optional[int], Optional[float]]:
        cap = None
        m = re.search(r"vllm:cache_config_info\{([^}]+)\}", text)
        if m:
            bs = re.search(r'block_size="(\d+)"', m.group(1))
            nb = re.search(r'num_gpu_blocks="(\d+)"', m.group(1))
            if bs and nb:
                cap = int(bs.group(1)) * int(nb.group(1))
        u = re.search(r"vllm:(?:gpu_cache_usage_perc|kv_cache_usage_perc)(?:\{[^}]*\})?\s+([\d.eE+-]+)", text)
        used = float(u.group(1)) if u else None
        return cap, used

    async def fetch(self) -> Tuple[Optional[int], Optional[float]]:
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout_s) as c:
            r = await c.get(f"{self.url}/metrics")
            r.raise_for_status()
        return self.parse(r.text)


class SGLangCapacity(CapacitySource):
    """SGLang: capacity from ``/get_server_info`` (``max_total_num_tokens``), usage from ``sglang:token_usage``."""

    def __init__(self, url: str, timeout_s: float = 5.0) -> None:
        self.url, self.timeout_s = _strip_v1(url), timeout_s

    @staticmethod
    def parse_metrics(text: str) -> Optional[float]:
        u = re.search(r"sglang:token_usage\{[^}]*\}\s+([\d.eE+-]+)", text)
        return float(u.group(1)) if u else None

    async def fetch(self) -> Tuple[Optional[int], Optional[float]]:
        import httpx

        cap, used = None, None
        async with httpx.AsyncClient(timeout=self.timeout_s) as c:
            try:
                r = await c.get(f"{self.url}/get_server_info")
                if r.status_code == 200:
                    data = r.json()
                    cap = data.get("max_total_num_tokens") or (data.get("internal_states") or [{}])[0].get("max_total_num_tokens")
                    cap = int(cap) if cap else None
            except Exception:
                pass
            try:
                r = await c.get(f"{self.url}/metrics")
                if r.status_code == 200:
                    used = self.parse_metrics(r.text)
            except Exception:
                pass
        return cap, used


class AutoCapacity(CapacitySource):
    """Detect the engine behind ``url``: vLLM's ``/metrics`` first, then SGLang's ``/get_server_info``.

    The first source that reports a capacity is remembered (``detected``) and
    used from then on.
    """

    def __init__(self, url: str, timeout_s: float = 5.0) -> None:
        self.url, self.timeout_s = _strip_v1(url), timeout_s
        self._sources = {"vllm": VLLMCapacity(url, timeout_s), "sglang": SGLangCapacity(url, timeout_s)}
        self.detected: Optional[str] = None

    async def fetch(self) -> Tuple[Optional[int], Optional[float]]:
        for kind in ([self.detected] if self.detected else ["vllm", "sglang"]):
            try:
                cap, used = await self._sources[kind].fetch()
            except Exception:
                continue
            if cap:
                self.detected = kind
                return cap, used
        return None, None


def make_capacity_source(kind: str, url: Optional[str] = None, capacity_tokens: Optional[int] = None) -> CapacitySource:
    """``static`` needs ``capacity_tokens``; ``vllm``/``sglang`` need ``url``; ``auto`` takes ``capacity_tokens`` if given, else detects the engine at ``url``."""
    if kind == "static" or (kind == "auto" and capacity_tokens):
        if not capacity_tokens or int(capacity_tokens) <= 0:
            raise ValueError("static capacity needs a positive capacity_tokens")
        return StaticCapacity(capacity_tokens)
    if kind not in ("auto", "vllm", "sglang"):
        raise ValueError(f"unknown capacity source {kind!r} (auto | static | vllm | sglang)")
    if not url:
        raise ValueError(f"{kind} capacity needs the backend url (or set capacity_tokens for a static capacity)")
    return {"auto": AutoCapacity, "vllm": VLLMCapacity, "sglang": SGLangCapacity}[kind](url)
