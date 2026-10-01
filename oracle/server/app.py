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
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Mapping, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ..config import ModelConfig, ServerConfig
from ..core import Oracle
from ..scheduling.admission import ClientGone
from ..types import ProgramOutcome
from .proxy import forward

logger = logging.getLogger(__name__)

_OUR_KEYS = {"program_id", "program_done", "program_success", "program_payload", "program_metadata", "force_model"}


def get_program_id(payload: Dict[str, Any], headers: Optional[Mapping[str, str]] = None, header_name: str = "X-Program-ID") -> str:
    if "program_id" in payload:
        return str(payload["program_id"])
    eb = payload.get("extra_body")
    if isinstance(eb, dict) and "program_id" in eb:
        return str(eb["program_id"])
    if headers is not None:
        for h in (header_name, "X-Session-ID"):
            v = headers.get(h) or headers.get(h.lower())
            if v:
                return str(v)
    return "default"


def _field(payload: Dict[str, Any], key: str, default=None):
    if key in payload:
        return payload[key]
    eb = payload.get("extra_body")
    if isinstance(eb, dict) and key in eb:
        return eb[key]
    return default


def first_user_text(payload: Dict[str, Any]) -> str:
    """The program's initial request: first user message (text parts only), else the prompt field."""
    for m in payload.get("messages") or []:
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    p = payload.get("prompt")
    if isinstance(p, str):
        return p
    if isinstance(p, list) and p and isinstance(p[0], str):
        return p[0]
    return ""


def _strip(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in payload.items() if k not in _OUR_KEYS}
    eb = out.get("extra_body")
    if isinstance(eb, dict):
        eb = {k: v for k, v in eb.items() if k not in _OUR_KEYS}
        if eb:
            out["extra_body"] = eb
        else:
            out.pop("extra_body")
    return out


def create_app(oracle: Oracle, models: List[ModelConfig], server: Optional[ServerConfig] = None, client: Optional[httpx.AsyncClient] = None) -> FastAPI:
    server = server or ServerConfig()
    by_name = {m.name: m for m in models}
    missing = [m for m in oracle.router.models if m not in by_name]
    if missing:
        raise ValueError(f"no backend URL configured for models {missing}")

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
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON") from exc
        pid = get_program_id(payload, request.headers, server.program_id_header)
        meta = _field(payload, "program_metadata") or {}
        force = _field(payload, "force_model")
        binding = oracle.lookup(pid)
        if binding is None:
            if force:
                if force not in by_name:
                    raise HTTPException(status_code=400, detail=f"unknown force_model {force!r}")
                oracle.router.bind(pid, first_user_text(payload), meta, model=force)
                if oracle.scheduler is not None:
                    try:
                        oracle.lookup(pid).admission_wait_s = await oracle.scheduler.admit(pid, oracle.backend_of[force], disconnected=request.is_disconnected)
                    except ClientGone:
                        oracle.router.release(pid)
                        raise HTTPException(status_code=499, detail="client disconnected while waiting for admission")
                binding = oracle.lookup(pid)
            else:
                try:
                    binding = await oracle.bind(pid, first_user_text(payload), meta, disconnected=request.is_disconnected)
                except ClientGone:
                    raise HTTPException(status_code=499, detail="client disconnected while waiting for admission")
        m = by_name[binding.model]
        fwd = _strip(payload)
        fwd["model"] = m.served_name or m.name
        headers = {"Authorization": f"Bearer {m.api_key}"} if m.api_key else {}
        done = bool(_field(payload, server.done_key, False))
        success = _field(payload, "program_success")
        extra_payload = _field(payload, "program_payload") or {}

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
        return JSONResponse({"object": "list", "data": [{"id": m.name, "object": "model", "owned_by": "oracle", "backend": m.url} for m in models]})

    @app.get("/programs")
    async def list_programs():
        return JSONResponse({pid: b.to_dict() for pid, b in oracle.router.table.items()})

    @app.get("/programs/{program_id}")
    async def get_program(program_id: str):
        b = oracle.lookup(program_id) or oracle.router.finished.get(program_id)
        if b is None:
            raise HTTPException(status_code=404, detail="unknown program")
        return JSONResponse(b.to_dict())

    @app.post("/programs/{program_id}/complete")
    async def complete_program(program_id: str, request: Request):
        try:
            body = await request.json() if await request.body() else {}
        except Exception:
            body = {}
        if oracle.lookup(program_id) is None:
            raise HTTPException(status_code=404, detail="unknown or already completed program")
        outcome = ProgramOutcome(program_id=program_id, cost=float(body.get("cost") or 0.0), success=body.get("success"), payload=body.get("payload") or {}, wall_time_s=body.get("wall_time_s"))
        oracle.complete(program_id, outcome)
        return JSONResponse({"program_id": program_id, "completed": True, "pending_verifications": oracle.router.feedback.pending})

    @app.post("/programs/complete")
    async def complete_program_body(request: Request):
        body = await request.json()
        pid = body.get("program_id")
        if not pid or oracle.lookup(pid) is None:
            raise HTTPException(status_code=404, detail="unknown or already completed program")
        oracle.complete(pid, ProgramOutcome(program_id=pid, cost=float(body.get("cost") or 0.0), success=body.get("success"), payload=body.get("payload") or {}))
        return JSONResponse({"program_id": pid, "completed": True})

    @app.post("/programs/{program_id}/release")
    async def release_program(program_id: str):
        return JSONResponse({"program_id": program_id, "released": oracle.release(program_id)})

    @app.get("/state")
    async def state():
        return JSONResponse(oracle.snapshot())

    @app.get("/health")
    async def health():
        s = oracle.snapshot()
        return JSONResponse({"status": "ok", "router": oracle.router.selector.name, "scheduler": oracle.scheduler is not None, "models": oracle.router.models, "active_programs": s["router"]["active"], "pending_verifications": s["router"]["feedback"]["pending"]})

    if server.ui:
        html_path = os.path.join(os.path.dirname(__file__), "ui", "index.html")

        @app.get("/ui", response_class=HTMLResponse)
        async def ui():
            with open(html_path) as f:
                return HTMLResponse(f.read())

        @app.get("/", response_class=HTMLResponse)
        async def root():
            return HTMLResponse('<meta http-equiv="refresh" content="0; url=/ui">')

    return app


__all__ = ["create_app", "get_program_id", "first_user_text"]
