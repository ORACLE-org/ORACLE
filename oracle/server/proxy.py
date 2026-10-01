"""Forward one OpenAI-compatible request to a backend and report its token usage.

Streaming responses are passed through unchanged; ``stream_options.include_usage``
is switched on so the final chunk carries the usage block.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Optional, Tuple

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

UsageCallback = Callable[[int, int], Awaitable[None]]  # (prompt_tokens, completion_tokens)


def _usage(obj: Any) -> Optional[Tuple[int, int]]:
    """``(prompt_tokens, completion_tokens)`` from a response object, ``None`` if it carries no usable usage block."""
    if not isinstance(obj, dict):
        return None
    u = obj.get("usage")
    if not isinstance(u, dict) or not u:
        return None
    try:
        return int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
    except (TypeError, ValueError):
        return None


def _error_body(text: bytes) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return {"error": text.decode(errors="replace")}


async def forward(client: httpx.AsyncClient, url: str, path: str, payload: Dict[str, Any], headers: Dict[str, str], on_usage: UsageCallback, on_done: Optional[Callable[[], Awaitable[None]]] = None):
    """Send ``payload`` to ``url + path``.  Calls ``on_usage`` once the usage is known
    and ``on_done`` when the response has fully left (streaming: after the last chunk)."""
    stream = bool(payload.get("stream"))
    if stream:
        so = dict(payload.get("stream_options") or {})
        so["include_usage"] = True
        payload["stream_options"] = so
    if not stream:
        try:
            r = await client.post(f"{url}{path}", json=payload, headers=headers)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"backend error: {e}") from e
        try:
            body = r.json()
        except ValueError:
            body = {"error": r.text}
        if r.status_code == 200:
            u = _usage(body)
            if u:
                await on_usage(*u)
        if on_done is not None:
            await on_done()
        return JSONResponse(body, status_code=r.status_code)

    req = client.build_request("POST", f"{url}{path}", json=payload, headers=headers)
    try:
        resp = await client.send(req, stream=True)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"backend error: {e}") from e
    if resp.status_code != 200:
        text = await resp.aread()
        await resp.aclose()
        if on_done is not None:
            await on_done()
        return JSONResponse(_error_body(text), status_code=resp.status_code)

    async def gen() -> AsyncIterator[bytes]:
        usage: Optional[Tuple[int, int]] = None
        buf = b""
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if line.startswith(b"data:") and line != b"data: [DONE]":
                        try:
                            u = _usage(json.loads(line[5:].strip()))
                        except ValueError:  # not JSON (a comment, a partial line): keep streaming
                            u = None
                        if u:
                            usage = u
        finally:
            await resp.aclose()
            if usage:
                await on_usage(*usage)
            if on_done is not None:
                await on_done()

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
