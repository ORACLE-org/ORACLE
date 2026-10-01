"""The proxy against a fake OpenAI-compatible backend (no network)."""
import json

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from oracle import Oracle, Router, DISC
from oracle.config import ModelConfig, ServerConfig
from oracle.server import create_app
from oracle.verification import ReportedVerifier


def fake_backend(name: str) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        p = await req.json()
        n = sum(len(m.get("content", "")) for m in p["messages"]) // 4 + 1
        if p.get("stream"):
            async def gen():
                yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                yield ('data: ' + json.dumps({"choices": [], "usage": {"prompt_tokens": n, "completion_tokens": 7}}) + "\n\n").encode()
                yield b"data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        return JSONResponse({"id": "x", "model": p["model"], "choices": [{"message": {"role": "assistant", "content": f"from {name}"}}], "usage": {"prompt_tokens": n, "completion_tokens": 5}})

    return app


class Mux(httpx.AsyncBaseTransport):
    def __init__(self, apps):
        self.t = {host: httpx.ASGITransport(app=a) for host, a in apps.items()}

    async def handle_async_request(self, request):
        return await self.t[request.url.host].handle_async_request(request)


@pytest.fixture
def system():
    router = Router(["strong", "weak"], "round_robin", verifiers={"rep": ReportedVerifier("rep")})
    disc = DISC({"strong": 100_000, "weak": 100_000}, policy=None, ledger_kwargs={"peak_prior": 10_000})
    o = Oracle(router, disc)
    models = [ModelConfig("strong", "http://strong/v1", served_name="big", price_in=1e6, price_out=1e6), ModelConfig("weak", "http://weak/v1", served_name="small")]
    client = httpx.AsyncClient(transport=Mux({"strong": fake_backend("strong"), "weak": fake_backend("weak")}))
    app = create_app(o, models, ServerConfig(), client=client)
    return app, o


def test_proxy_routes_and_completes(run, system):
    app, o = system

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            async with app.router.lifespan_context(app):
                r = await c.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "fix the test"}], "program_id": "p1"})
                assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "from strong"
                b = o.lookup("p1")
                assert b.model == "strong" and b.requests == 1 and b.cost == pytest.approx(4 + 5)  # price 1e6/1M = 1 $/token
                r = await c.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "again"}], "extra_body": {"program_id": "p1"}})
                assert r.json()["choices"][0]["message"]["content"] == "from strong"  # sticky
                r = await c.post("/v1/chat/completions", headers={"X-Program-ID": "p2"}, json={"model": "auto", "messages": [{"role": "user", "content": "book flight"}], "stream": True})
                assert r.status_code == 200 and b"[DONE]" in r.content
                assert o.lookup("p2").model == "weak" and o.lookup("p2").completion_tokens == 7
                r = await c.post("/programs/p1/complete", json={"success": 1.0, "cost": 0.5})
                assert r.json()["completed"] and o.lookup("p1") is None
                await o.router.feedback.drain()
                assert o.router.feedback.completed == 1 and o.router.finished["p1"].score == 1.0
                r = await c.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "last"}], "program_id": "p2", "program_done": True, "program_success": 0.0})
                assert r.status_code == 200
                await o.router.feedback.drain()
                assert o.lookup("p2") is None and o.router.feedback.completed == 2
                assert "p2" not in o.scheduler.where
                s = (await c.get("/state")).json()
                assert s["router"]["active"] == 0 and s["scheduler"]["dispatched"] == 2
                assert (await c.get("/health")).json()["status"] == "ok"
                assert "ORACLE" in (await c.get("/ui")).text
                assert (await c.post("/programs/zzz/complete", json={})).status_code == 404
                r = await c.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "x"}], "program_id": "p3", "force_model": "weak"})
                assert r.json()["choices"][0]["message"]["content"] == "from weak"

    run(go())


def test_disc_middleware(run):
    from oracle.integrations.fastapi import DISCMiddleware

    disc = DISC({"local": 100_000}, policy=None, ledger_kwargs={"peak_prior": 10_000})
    app = fake_backend("local")
    app.add_middleware(DISCMiddleware, disc=disc)

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "program_id": "q1"})
            assert r.status_code == 200 and "q1" in disc.where
            r = await c.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "bye"}], "program_id": "q1", "program_done": True})
            assert r.status_code == 200 and "q1" not in disc.where and disc.ledgers["local"].stats["released"] == 1

    run(go())
