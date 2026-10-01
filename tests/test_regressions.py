"""Regression tests for the bugs fixed while hardening the first release.

Each test names the failure it guards against: NaN rewards poisoning a learner,
co-located models bound to the wrong model, the proxy answering 500 (and leaking
a binding) on malformed input, a second request skipping the admission gate,
cancellation at the gate, config edge cases, shell injection through verifier
payloads, and so on.
"""
import asyncio
import math
import subprocess
import sys

import httpx
import numpy as np
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from oracle import DISC, AdmissionGate, DispatchPolicy, Oracle, ProgramOutcome, ReservationLedger, Reward, Router, RoutingContext
from oracle.config import ModelConfig, OracleConfig, build
from oracle.routing import CallableSelector, HashingFeatures, LinUCB, Random, TypeUCB
from oracle.scheduling.capacity import AutoCapacity, VLLMCapacity, make_capacity_source
from oracle.server import create_app
from oracle.server.fields import first_user_text, get_program_id
from oracle.verification import CallableVerifier, CommandVerifier, LLMJudgeVerifier, ReportedVerifier


# --- feedback / rewards ------------------------------------------------------
def test_nan_score_is_a_failed_verification_not_a_poisoned_selector(run):
    async def go():
        r = Router(["a", "b"], "linucb", verifiers={"v": CallableVerifier("v", lambda o: float("nan"))})
        r.bind("p", "hello")
        assert math.isnan(await r.complete("p", ProgramOutcome("p")))
        assert r.feedback.failed == 1 and r.feedback.completed == 0 and r.selector.n_update == 0
        assert r.bind("q", "hello").model in ("a", "b")  # the selector still works afterwards

    run(go())


def test_learners_refuse_non_finite_rewards_and_features():
    x = np.array([1.0, 0.0, 1.0])
    for sel in (LinUCB(["a", "b"], dim=3), TypeUCB(["a", "b"])):
        with pytest.raises(ValueError):
            sel.update(RoutingContext("p", features=x), "a", float("nan"))
    with pytest.raises(ValueError):
        LinUCB(["a"], dim=2).select(RoutingContext("p", features=np.array([float("inf"), 1.0])))


def test_reward_rejects_nan_and_ignores_cost_when_alpha_is_zero():
    assert Reward()(1.0, float("nan")) == 1.0
    with pytest.raises(ValueError):
        Reward()(float("nan"), 0.0)
    with pytest.raises(ValueError):
        Reward(alpha=0.5)(1.0, float("inf"))
    with pytest.raises(ValueError):
        Reward(cost_ref=float("nan"))


def test_plain_functions_are_accepted_as_verifiers(run):
    async def async_score(o):
        return 1.0

    async def go():
        r = Router(["a"], "fixed", verifiers={"swe": lambda o: 0.5, "tau2": async_score})
        r.bind("p", "x")
        assert await r.complete("p", ProgramOutcome("p")) == 0.5 and r.feedback.failed == 0

    run(go())
    with pytest.raises(TypeError):
        Router(["a"], "fixed", verifiers={"x": 42})


# --- router -------------------------------------------------------------------
def test_complete_outside_event_loop_is_a_clear_error_and_keeps_state():
    r = Router(["a"], "fixed")
    r.bind("p", "x")
    with pytest.raises(RuntimeError, match="complete_sync"):
        r.complete("p", ProgramOutcome("p", success=1.0))
    assert r.lookup("p") is not None and "p" not in r.finished  # nothing was lost
    assert r.complete_sync("p", ProgramOutcome("p", success=1.0)) == 1.0


def test_router_validates_models_and_selector_choice():
    with pytest.raises(ValueError):
        Router([], "fixed")
    with pytest.raises(ValueError):
        Router(["a", "a"], "fixed")
    with pytest.raises(ValueError):
        Router(["a"], "fixed").bind("p", "x", model="zzz")
    bad = Router(["a", "b"], CallableSelector(["a", "b", "c"], lambda ctx: "c"))
    with pytest.raises(ValueError, match="not one of the router's models"):
        bad.bind("p", "x")


def test_selector_argument_validation():
    with pytest.raises(ValueError):
        Random(["a", "b"], weights=[1])
    with pytest.raises(ValueError):
        Random(["a", "b"], weights=[0, 0])
    with pytest.raises(ValueError):
        HashingFeatures(dim=2)


# --- Oracle: router + DISC ----------------------------------------------------
def test_shared_backend_keeps_the_proposed_model(run):
    async def go():
        r = Router(["m1", "m2"], "fixed", verifiers={"rep": ReportedVerifier("rep")})
        o = Oracle(r, DISC({"gpu": 100_000}, policy=None), backend_of={"m1": "gpu", "m2": "gpu"})
        b = await o.bind("p", "x")
        assert (b.proposed_model, b.model, b.diverted) == ("m1", "m1", False)
        assert o.models_on == {"gpu": ["m1", "m2"]}
        with pytest.raises(ValueError):
            Oracle(r, None, backend_of={"m1": "gpu"})  # m2 has no backend

    run(go())


def test_diversion_to_a_shared_backend_picks_its_best_model(run):
    async def go():
        est = {"m1": 0.9, "m2": 0.5, "m3": 0.7}
        sel = CallableSelector(["m1", "m2", "m3"], lambda ctx: "m1", estimate_fn=lambda ctx, m: est[m])
        r = Router(["m1", "m2", "m3"], sel, verifiers={"rep": ReportedVerifier("rep")})
        disc = DISC({"a": 100_000, "b": 100_000}, policy=DispatchPolicy(beta=0.9, w0=1.0), ledger_kwargs={"peak_prior": 90_000, "prior_weight": 100, "duration_prior_s": 5.0})
        o = Oracle(r, disc, backend_of={"m1": "a", "m2": "b", "m3": "b"})
        assert (await o.bind("hog", "x")).model == "m1"  # fills backend a
        b = await o.bind("p", "x")
        assert b.diverted and b.proposed_model == "m1" and b.model == "m3" and o.scheduler.where["p"] == "b"

    run(go())


def test_second_request_waits_for_the_first_admission(run):
    async def go():
        r = Router(["a"], "fixed", verifiers={"rep": ReportedVerifier("rep")})
        o = Oracle(r, DISC({"a": 100_000}, policy=None, ledger_kwargs={"peak_prior": 90_000, "prior_weight": 100}))
        await o.bind("hog", "x")
        t1 = asyncio.create_task(o.bind("p", "x"))
        await asyncio.sleep(0.02)
        t2 = asyncio.create_task(o.bind("p", "x"))
        await asyncio.sleep(0.02)
        assert not t1.done() and not t2.done()  # the second request does not skip the gate
        o.complete("hog")
        b1, b2 = await t1, await t2
        assert b1 is b2 and b1.admitted_at is not None and "p" in o.scheduler.where
        await o.router.feedback.drain()

    run(go())


def test_forced_model_is_scheduled_and_not_diverted(run):
    async def go():
        r = Router(["a", "b"], "fixed", verifiers={"rep": ReportedVerifier("rep")})
        o = Oracle(r, DISC({"a": 100_000, "b": 100_000}, policy=DispatchPolicy(beta=1.0)))
        b = await o.bind("p", "x", model="b")
        assert b.model == b.proposed_model == "b" and not b.diverted and b.admitted_at is not None and o.scheduler.where["p"] == "b"
        assert b.context.metadata["dispatch"] == {"forced": True}
        with pytest.raises(ValueError):
            await o.bind("q", "x", model="zzz")
        assert o.lookup("q") is None

    run(go())


# --- scheduling ---------------------------------------------------------------
def test_ledger_prior_weight_zero_unknown_capacity_and_bad_args():
    led = ReservationLedger(100_000, prior_weight=0, peak_prior=20_000, duration_prior_s=3.0)
    assert led.c_hat == 20_000 and led.d_hat == 3.0
    led.reserve("a")
    led.release("a", peak_tokens=30_000)
    assert led.c_hat == 30_000
    unknown = ReservationLedger(0)
    unknown.reserve("a")
    assert unknown.fits() and unknown.wait_forecast(5) == 0.0
    unknown.observe_usage(float("nan"))
    assert unknown.measured_usage is None
    with pytest.raises(ValueError):
        ReservationLedger(1, rho=0)


def test_gate_gives_the_slot_back_when_cancelled_at_the_grant(run):
    async def go():
        led = ReservationLedger(100_000, rho=1.0, peak_prior=50_000, prior_weight=100)
        g = AdmissionGate(led, poll_s=5.0)
        await g.admit("p1")
        await g.admit("p2")
        t3 = asyncio.create_task(g.admit("p3"))
        t4 = asyncio.create_task(g.admit("p4"))
        await asyncio.sleep(0.01)
        g.release("p1")  # grants p3 in this instant ...
        t3.cancel()  # ... but p3 is cancelled before it ever resumed
        with pytest.raises(asyncio.CancelledError):
            await t3
        await asyncio.sleep(0.01)
        assert "p3" not in led.admitted and t4.done() and "p4" in led.admitted  # slot handed to p4 at once, no poll wait
        assert led.dur_n == 1 and led.stats["released"] == 2  # the cancelled grant taught the ledger nothing

    run(go())


def test_disc_default_policy_is_not_shared_between_instances():
    d1 = DISC({"a": 1})
    d1.policy.beta = 0.9
    assert DISC({"a": 1}).policy.beta == 0.5 and DISC({"a": 1}, policy=None).policy is None


def test_auto_capacity_detects_the_engine(run):
    async def go():
        auto = AutoCapacity("http://x/v1")
        calls = []

        async def vllm_fail():
            calls.append("vllm")
            raise httpx.ConnectError("no")

        async def sglang_ok():
            calls.append("sglang")
            return 4096, 0.1

        auto._sources["vllm"].fetch = vllm_fail
        auto._sources["sglang"].fetch = sglang_ok
        assert await auto.fetch() == (4096, 0.1) and auto.detected == "sglang"
        assert await auto.fetch() == (4096, 0.1) and calls == ["vllm", "sglang", "sglang"]

    run(go())
    assert VLLMCapacity.parse("vllm:gpu_cache_usage_perc 0.5\n") == (None, 0.5)
    with pytest.raises(ValueError):
        make_capacity_source("vllm")
    with pytest.raises(ValueError):
        make_capacity_source("nope", "http://x")


# --- config -------------------------------------------------------------------
def test_config_edge_cases():
    cfg = OracleConfig.from_dict({"models": [{"name": "a", "url": "http://a/v1"}], "router": None, "scheduler": {"enabled": False}})
    assert cfg.router.selector == "linucb" and build(cfg).scheduler is None
    with pytest.raises(ValueError, match="duplicate"):
        OracleConfig.from_dict({"models": [{"name": "a"}, {"name": "a"}]})
    with pytest.raises(ValueError, match="router"):
        OracleConfig.from_dict({"router": {"selectr": "ucb"}})
    with pytest.raises(ValueError, match="capacity_tokens"):
        build(OracleConfig.from_dict({"models": [{"name": "a", "url": "http://a/v1", "capacity": "static"}]}))
    o = build(OracleConfig.from_dict({"models": [{"name": "a", "url": "http://a/v1"}, {"name": "b", "url": "http://b/v1"}], "router": {"enabled": False}}))
    assert o.router.selector.name == "fixed"
    o = build(OracleConfig.from_dict({"models": [{"name": "a", "url": "http://a/v1"}]}))
    assert isinstance(o.scheduler.sources["a"], AutoCapacity)


# --- verifiers ----------------------------------------------------------------
def test_command_verifier_quotes_client_values(run):
    hostile = ProgramOutcome("p", payload={"name": "x; false"})
    v = CommandVerifier("t", "echo {name}")
    assert "'x; false'" in v.render(hostile)
    assert run(v.verify(hostile)) == 1.0  # the injected `false` never runs
    assert run(CommandVerifier("t", "echo {name}", quote=False).verify(hostile)) == 0.0
    assert run(CommandVerifier("t", "exit 1").verify(ProgramOutcome("p"))) == 0.0
    assert run(CommandVerifier("t", "sleep 5", timeout_s=0.1).verify(ProgramOutcome("p"))) == 0.0


def test_llm_judge_score_parsing():
    p = LLMJudgeVerifier.parse_score
    assert p("0.85") == 0.85 and p("Score: 1") == 1.0 and p("1.0\n") == 1.0 and p("I'd say 0.7.") == 0.7
    assert p("10") == 0.0 and p("1.5") == 0.0 and p("70/100") == 0.0 and p("") == 0.0


# --- server -------------------------------------------------------------------
def test_request_field_helpers():
    assert get_program_id({"program_id": None, "extra_body": {"program_id": "e"}}) == "e"
    assert get_program_id({"program_id": ""}, {"x-program-id": "h"}) == "h"
    assert get_program_id({}) == "default"
    assert first_user_text({"messages": ["hi", {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image_url"}]}]}) == "a"
    assert first_user_text({"messages": "hi", "prompt": ["p"]}) == "p"
    assert first_user_text({"messages": 5}) == ""


def test_integrations_import_without_fastapi():
    code = "import sys; sys.modules['fastapi'] = None; import oracle.server, oracle.integrations.litellm, oracle.integrations.fastapi; print(oracle.server.get_program_id({'program_id': 'x'}))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "x"


def _backend(name: str, stream_prefix: bytes = b"", body=None) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        p = await req.json()
        if body is not None:
            return JSONResponse(body)
        if p.get("stream"):
            async def gen():
                yield stream_prefix + b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":7}}\n\ndata: [DONE]\n\n'

            return StreamingResponse(gen(), media_type="text/event-stream")
        return JSONResponse({"choices": [{"message": {"role": "assistant", "content": f"from {name}"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 5}})

    return app


class _Mux(httpx.AsyncBaseTransport):
    def __init__(self, apps):
        self.t = {host: httpx.ASGITransport(app=a) for host, a in apps.items()}

    async def handle_async_request(self, request):
        return await self.t[request.url.host].handle_async_request(request)


def _system(**backend_kw):
    router = Router(["strong", "weak"], "round_robin", verifiers={"rep": ReportedVerifier("rep")})
    o = Oracle(router, DISC({"strong": 100_000, "weak": 100_000}, policy=None))
    models = [ModelConfig("strong", "http://strong/v1"), ModelConfig("weak", "http://weak/v1")]
    client = httpx.AsyncClient(transport=_Mux({"strong": _backend("strong", **backend_kw), "weak": _backend("weak", **backend_kw)}))
    return create_app(o, models, client=client), o


MSGS = [{"role": "user", "content": "hi"}]
JSON_HDR = {"content-type": "application/json"}


def test_create_app_requires_backend_urls():
    o = Oracle(Router(["strong", "weak"], "fixed"))
    with pytest.raises(ValueError, match="no backend url"):
        create_app(o, [ModelConfig("strong"), ModelConfig("weak", "http://w/v1")])


def test_proxy_rejects_malformed_input_with_400(run):
    app, o = _system()

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c, app.router.lifespan_context(app):
            assert (await c.post("/v1/chat/completions", content=b"[]", headers=JSON_HDR)).status_code == 400
            assert (await c.post("/v1/chat/completions", content=b"{bad", headers=JSON_HDR)).status_code == 400
            for bad in ({"program_metadata": "x"}, {"program_payload": [1], "program_done": True}, {"program_success": "abc"}, {"program_success": 2}, {"force_model": "nope"}, {"force_model": 3}):
                r = await c.post("/v1/chat/completions", json={"messages": MSGS, "program_id": "p", **bad})
                assert r.status_code == 400 and r.json()["detail"], bad
            assert o.lookup("p") is None  # a rejected request binds nothing
            r = await c.post("/v1/chat/completions", json={"messages": ["hi"], "program_id": None})
            assert r.status_code == 200 and o.lookup("default") is not None  # odd messages are the backend's business; a null id means no id
            assert (await c.post("/programs/complete", content=b"{bad", headers=JSON_HDR)).status_code == 400
            assert (await c.post("/programs/complete", json={"cost": 1})).status_code == 400  # no program_id
            assert (await c.post("/programs/default/complete", json={"success": "abc"})).status_code == 400
            assert (await c.post("/programs/default/complete", json={"cost": -1})).status_code == 400
            assert o.lookup("default") is not None and o.router.feedback.failed == 0
            assert (await c.post("/programs/default/complete", content=b"")).status_code == 200
            assert (await c.post("/programs/default/complete", content=b"")).status_code == 404

    run(go())


def test_force_model_is_admitted_and_sticky(run):
    app, o = _system()

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c, app.router.lifespan_context(app):
            r = await c.post("/v1/chat/completions", json={"messages": MSGS, "program_id": "f", "force_model": "weak"})
            assert r.json()["choices"][0]["message"]["content"] == "from weak"
            b = o.lookup("f")
            assert b.model == "weak" and not b.diverted and b.admitted_at is not None and o.scheduler.where["f"] == "weak"
            r = await c.post("/v1/chat/completions", json={"messages": MSGS, "program_id": "f", "force_model": "strong"})
            assert r.json()["choices"][0]["message"]["content"] == "from weak"  # later requests keep the binding

    run(go())


def test_streaming_usage_survives_non_object_lines(run):
    app, o = _system(stream_prefix=b"data: 123\n\n: keepalive\n\n")

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c, app.router.lifespan_context(app):
            r = await c.post("/v1/chat/completions", json={"messages": MSGS, "program_id": "s", "stream": True})
            assert r.status_code == 200 and b"[DONE]" in r.content and o.lookup("s").completion_tokens == 7

    run(go())


def test_backend_non_object_body_is_passed_through(run):
    app, o = _system(body=[1, 2])

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c, app.router.lifespan_context(app):
            r = await c.post("/v1/chat/completions", json={"messages": MSGS, "program_id": "l"})
            assert r.status_code == 200 and r.json() == [1, 2] and o.lookup("l").requests == 0

    run(go())


def test_state_tolerates_numpy_and_dashboard_escapes_client_strings(run):
    app, o = _system()
    o.router.selector.state = lambda: {"name": "rr", "weird": np.float64(1.5), "arr": np.zeros(2)}

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c, app.router.lifespan_context(app):
            s = (await c.get("/state")).json()
            assert s["router"]["selector"]["weird"] == 1.5 and s["router"]["selector"]["arr"] == [0.0, 0.0]
            html = (await c.get("/ui")).text
            assert "const esc=" in html and "${esc(p.program_id)}" in html and "${h.program_id}" not in html

    run(go())


# --- CLI ----------------------------------------------------------------------
def test_cli_demo_and_selectors(capsys):
    from oracle.__main__ import main

    assert main(["selectors"]) == 0 and "linucb" in capsys.readouterr().out
    assert main(["demo", "--programs", "30", "--time-scale", "0", "--selectors", "fixed,ucb"]) == 0
    assert "accuracy=" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="nope"):
        main(["demo", "--programs", "5", "--selectors", "nope"])
