"""ORACLE's request fields: reading and validating them from an OpenAI-style request body.

Shared by the proxy (:mod:`oracle.server.app`) and the integrations; this
module has no FastAPI dependency so that the integrations stay importable
without the ``server`` extra.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

OUR_KEYS = {"program_id", "program_done", "program_success", "program_payload", "program_metadata", "force_model"}
"""Fields ORACLE consumes; they are stripped before the request is forwarded."""


def get_field(payload: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """A field at the top level of the body or inside ``extra_body`` (what the OpenAI SDK's ``extra_body=`` produces)."""
    if key in payload:
        return payload[key]
    eb = payload.get("extra_body")
    if isinstance(eb, dict) and key in eb:
        return eb[key]
    return default


def get_program_id(payload: Mapping[str, Any], headers: Optional[Mapping[str, str]] = None, header_name: str = "X-Program-ID") -> str:
    """The program id: body, ``extra_body``, then the ``X-Program-ID``/``X-Session-ID`` headers; ``"default"`` if absent."""
    for src in (payload, payload.get("extra_body")):
        if isinstance(src, Mapping):
            v = src.get("program_id")
            if v is not None and str(v) != "":
                return str(v)
    if headers is not None:
        for h in (header_name, "X-Session-ID"):
            v = headers.get(h) or headers.get(h.lower())
            if v:
                return str(v)
    return "default"


def first_user_text(payload: Mapping[str, Any]) -> str:
    """The program's initial request: first user message (text parts only), else the ``prompt`` field, else ``""``."""
    msgs = payload.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if not isinstance(m, dict) or m.get("role") != "user":
                continue
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return "\n".join(str(p.get("text", "")) for p in c if isinstance(p, dict) and p.get("type") == "text")
    p = payload.get("prompt")
    if isinstance(p, str):
        return p
    if isinstance(p, list) and p and isinstance(p[0], str):
        return p[0]
    return ""


def strip_fields(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """The body without ORACLE's fields (top level and ``extra_body``)."""
    out = {k: v for k, v in payload.items() if k not in OUR_KEYS}
    eb = out.get("extra_body")
    if isinstance(eb, dict):
        eb = {k: v for k, v in eb.items() if k not in OUR_KEYS}
        if eb:
            out["extra_body"] = eb
        else:
            out.pop("extra_body")
    return out


def parse_success(value: Any, field: str = "program_success") -> Optional[float]:
    """``None`` stays ``None``; otherwise a finite number in [0, 1].  Raises ``ValueError``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(value) and 0.0 <= value <= 1.0:
        return float(value)
    raise ValueError(f"{field} must be a number in [0, 1] or null, got {value!r}")


def parse_object(value: Any, field: str) -> Dict[str, Any]:
    """``None`` -> ``{}``; a JSON object is returned as a dict; anything else raises ``ValueError``."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    raise ValueError(f"{field} must be a JSON object, got {type(value).__name__}")


def parse_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes")


def program_fields(payload: Mapping[str, Any], done_key: str = "program_done") -> Dict[str, Any]:
    """Validate ORACLE's fields of one request.

    Returns ``{"metadata", "payload", "success", "done", "force"}``; raises
    ``ValueError`` with a message fit for a 400 response.
    """
    force = get_field(payload, "force_model")
    if force is not None and (not isinstance(force, str) or not force):
        raise ValueError(f"force_model must be a model name, got {force!r}")
    return {
        "metadata": parse_object(get_field(payload, "program_metadata"), "program_metadata"),
        "payload": parse_object(get_field(payload, "program_payload"), "program_payload"),
        "success": parse_success(get_field(payload, "program_success")),
        "done": parse_flag(get_field(payload, done_key)),
        "force": force,
    }


def parse_cost(value: Any, field: str = "cost") -> float:
    """``None`` -> 0.0; otherwise a finite non-negative number.  Raises ``ValueError``."""
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be a non-negative number, got {value!r}")
    return float(value)
