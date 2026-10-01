"""The ORACLE server: an OpenAI-compatible proxy with the router and/or DISC in front.

Clients change one thing: send a ``program_id``
(in the body, in ``extra_body`` or in the ``X-Program-ID`` header).  The first
request of a program binds it; later requests reuse the binding.  Mark the
last request with ``program_done: true`` (optionally with ``program_success``
and ``program_payload``), or call ``POST /programs/{id}/complete`` from the
harness once the task has been graded.

Routes
------
POST /v1/chat/completions, /v1/completions   proxy (routed + scheduled)
GET  /v1/models                              the configured models
GET  /programs, /programs/{id}               bindings
POST /programs/{id}/complete                 report outcome -> release slot -> verify -> learn
POST /programs/{id}/release                  drop without feedback
GET  /state                                  everything the dashboard shows (JSON)
GET  /health, GET /ui

Malformed input (a body that is not a JSON object, a ``program_success``
outside [0, 1], a ``program_metadata`` that is not an object, an unknown
``force_model``) is answered with 400 before anything is bound or forwarded.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ..config import ModelConfig, ServerConfig
from ..core import Oracle
from ..scheduling.admission import ClientGone
from ..types import ProgramOutcome
from .fields import first_user_text, get_program_id, parse_cost, parse_object, parse_success, program_fields, strip_fields
from .proxy import forward

logger = logging.getLogger(__name__)


def _json_default(o: Any):
    """Let plugin selectors put numpy values (or anything printable) in their ``state()`` without breaking ``/state``."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=str)
    return str(o)


class _JSON(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(content, ensure_ascii=False, allow_nan=False, separators=(",", ":"), default=_json_default).encode("utf-8")


async def _json_object(request: Request, required: bool = True) -> Dict[str, Any]:
    """The request body as a dict; 400 if it is not JSON or not an object (an empty body is ``{}`` unless ``required``)."""
    raw = await request.body()
    if not raw.strip():
        if required:
            raise HTTPException(status_code=400, detail="request body must be a JSON object")
        return {}
    try:
        body = json.loads(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")
    return body


def _outcome(program_id: str, body: Dict[str, Any]) -> ProgramOutcome:
    try:
        wall = body.get("wall_time_s")
        return ProgramOutcome(program_id=program_id, cost=parse_cost(body.get("cost")), success=parse_success(body.get("success"), "success"), payload=parse_object(body.get("payload"), "payload"), wall_time_s=None if wall is None else parse_cost(wall, "wall_time_s"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def create_app(oracle: Oracle, models: List[ModelConfig], server: Optional[ServerConfig] = None, client: Optional[httpx.AsyncClient] = None) -> FastAPI:
    server = server or ServerConfig()
    by_name = {m.name: m for m in models}
    missing = [m for m in oracle.router.models if m not in by_name]
    if missing:
        raise ValueError(f"no backend configured for models {missing}")
    no_url = [m.name for m in models if not m.url]
    if no_url:
        raise ValueError(f"models {no_url} have no backend url (an OpenAI-compatible base URL ending in /v1)")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = client or httpx.AsyncClient(timeout=server.request_timeout_s, limits=httpx.Limits(max_connections=None, max_keepalive_connections=None))
        await oracle.start()
        try:
            yield
        finally:
            await oracle.stop()
            if client is None:
                await app.state.client.aclose()

    app = FastAPI(title="ORACLE", lifespan=lifespan)
    app.state.oracle = oracle

    async def _proxy(request: Request, path: str):
        payload = await _json_object(request)
        pid = get_program_id(payload, request.headers, server.program_id_header)
        try:
            f = program_fields(payload, server.done_key)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        force = f["force"]
        if force is not None and force not in by_name:
            raise HTTPException(status_code=400, detail=f"unknown force_model {force!r}; known: {list(by_name)}")
        binding = oracle.lookup(pid)
        if binding is None:
            try:
                binding = await oracle.bind(pid, first_user_text(payload), f["metadata"], disconnected=request.is_disconnected, model=force)
            except ClientGone:
                raise HTTPException(status_code=499, detail="client disconnected while waiting for admission")
        m = by_name[binding.model]
        fwd = strip_fields(payload)
        fwd["model"] = m.served_name or m.name
        headers = {"Authorization": f"Bearer {m.api_key}"} if m.api_key else {}
        done, success, extra_payload = f["done"], f["success"], f["payload"]

        async def on_usage(pt: int, ct: int) -> None:
            cost = (pt * m.price_in + ct * m.price_out) / 1e6
            oracle.observe_request(pid, pt, ct, cost)

        async def on_done() -> None:
            if done:
                oracle.complete(pid, ProgramOutcome(program_id=pid, success=success, payload=dict(extra_payload)))

        return await forward(request.app.state.client, m.url.rstrip("/"), path, fwd, headers, on_usage, on_done)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _proxy(request, "/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _proxy(request, "/completions")

    @app.get("/v1/models")
    async def list_models():
        return _JSON({"object": "list", "data": [{"id": m.name, "object": "model", "owned_by": "oracle", "backend": m.url} for m in models]})

    @app.get("/programs")
    async def list_programs():
        return _JSON({pid: b.to_dict() for pid, b in oracle.router.table.items()})

    @app.get("/programs/{program_id}")
    async def get_program(program_id: str):
        b = oracle.lookup(program_id) or oracle.router.finished.get(program_id)
        if b is None:
            raise HTTPException(status_code=404, detail="unknown program")
        return _JSON(b.to_dict())

    def _complete(program_id: str, body: Dict[str, Any]):
        outcome = _outcome(program_id, body)  # validate before touching the program
        if oracle.lookup(program_id) is None:
            raise HTTPException(status_code=404, detail="unknown or already completed program")
        oracle.complete(program_id, outcome)
        return _JSON({"program_id": program_id, "completed": True, "pending_verifications": oracle.router.feedback.pending})

    @app.post("/programs/complete")
    async def complete_program_body(request: Request):
        body = await _json_object(request)
        pid = body.get("program_id")
        if pid is None or str(pid) == "":
            raise HTTPException(status_code=400, detail="program_id is required")
        return _complete(str(pid), body)

    @app.post("/programs/{program_id}/complete")
    async def complete_program(program_id: str, request: Request):
        return _complete(program_id, await _json_object(request, required=False))

    @app.post("/programs/{program_id}/release")
    async def release_program(program_id: str):
        return _JSON({"program_id": program_id, "released": oracle.release(program_id)})

    @app.get("/state")
    async def state():
        return _JSON(oracle.snapshot())

    @app.get("/health")
    async def health():
        s = oracle.snapshot()
        return _JSON({"status": "ok", "router": oracle.router.selector.name, "scheduler": oracle.scheduler is not None, "models": oracle.router.models, "active_programs": s["router"]["active"], "pending_verifications": s["router"]["feedback"]["pending"]})

    if server.ui:
        html_path = os.path.join(os.path.dirname(__file__), "ui", "index.html")

        @app.get("/ui", response_class=HTMLResponse)
        async def ui():
            with open(html_path, encoding="utf-8") as f:
                return HTMLResponse(f.read())

        @app.get("/", response_class=HTMLResponse)
        async def root():
            return HTMLResponse('<meta http-equiv="refresh" content="0; url=/ui">')

    return app


__all__ = ["create_app", "get_program_id", "first_user_text"]
