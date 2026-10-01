"""DISC as an ASGI middleware: admission control in front of any OpenAI-compatible server.

Mount it on your own FastAPI/Starlette app (an SGLang router written in
Python, a vLLM-based proxy, ...) and every program is held at the gate until
its backend has room, released when the client marks the program done or calls
``POST /programs/{id}/release``::

    from oracle.scheduling import DISC
    from oracle.integrations.fastapi import DISCMiddleware

    app.add_middleware(DISCMiddleware, disc=DISC({"local": 120_000}), backend_of=lambda payload: "local")

A client disconnect while waiting frees the queue slot.  For a full proxy with
routing, run ``oracle serve`` instead.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional

from ..scheduling.admission import ClientGone
from ..scheduling.disc import DISC
from ..server.app import get_program_id


class DISCMiddleware:
    def __init__(self, app, disc: DISC, backend_of: Optional[Callable[[Dict[str, Any]], str]] = None, paths=("/v1/chat/completions", "/v1/completions"), done_key: str = "program_done", release_path: str = "/programs/release") -> None:
        self.app, self.disc, self.paths, self.done_key, self.release_path = app, disc, set(paths), done_key, release_path
        self.backend_of = backend_of or (lambda payload: next(iter(disc.ledgers)))

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST" or scope["path"] not in self.paths | {self.release_path}:
            return await self.app(scope, receive, send)
        body = b""
        more = True
        while more:
            msg = await receive()
            body += msg.get("body", b"")
            more = msg.get("more_body", False)
        try:
            payload = json.loads(body or b"{}")
        except ValueError:
            payload = {}
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        pid = get_program_id(payload, headers)

        if scope["path"] == self.release_path:
            self.disc.release(pid)
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": json.dumps({"program_id": pid, "released": True}).encode()})
            return

        if pid not in self.disc.where:
            backend, _ = self.disc.dispatch(self.backend_of(payload))
            try:
                await self.disc.admit(pid, backend)
            except ClientGone:
                await send({"type": "http.response.start", "status": 499, "headers": []})
                await send({"type": "http.response.body", "body": b""})
                return

        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        done = bool(payload.get(self.done_key) or (payload.get("extra_body") or {}).get(self.done_key))
        try:
            await self.app(scope, replay, send)
        finally:
            if done:
                self.disc.release(pid)
