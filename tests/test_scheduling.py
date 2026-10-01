import asyncio

import pytest

from oracle import DISC, ReservationLedger, AdmissionGate, DispatchPolicy
from oracle.scheduling.capacity import VLLMCapacity, SGLangCapacity


def test_ledger_reserves_expected_peak_and_learns():
    clock = [0.0]
    led = ReservationLedger(capacity_tokens=100_000, rho=0.8, peak_prior=20_000, prior_weight=1, duration_prior_s=0.0, clock=lambda: clock[0])
    assert led.fits() and led.free_slots() == 4  # 80k / 20k
    for i in range(4):
        led.reserve(f"p{i}", tokens=1000)
    assert not led.fits() and led.free_slots() == 0
    assert led.reserved() == 4 * 20_000  # floor at c_hat even though contexts are small
    led.update("p0", 50_000)
    assert led.reserved() == 50_000 + 3 * 20_000
    clock[0] = 10.0
    led.release("p0", peak_tokens=60_000)
    assert led.c_hat == (20_000 + 60_000) / 2 and led.d_hat == 5.0
    assert led.release("nope") is False


def test_wait_forecast():
    led = ReservationLedger(100_000, rho=1.0, peak_prior=50_000, prior_weight=1, duration_prior_s=10.0)
    assert led.wait_forecast(0) == 0.0
    led.reserve("a")
    led.reserve("b")
    assert led.free_slots() == 0
    assert led.wait_forecast(0) == pytest.approx(1 * 10.0 / 2)
    assert led.wait_forecast(3) == pytest.approx(4 * 10.0 / 2)


def test_gate_fifo_and_release(run):
    async def go():
        led = ReservationLedger(100_000, rho=1.0, peak_prior=50_000, prior_weight=100, max_programs=0)
        g = AdmissionGate(led, poll_s=0.01)
        w1 = await g.admit("p1")
        w2 = await g.admit("p2")
        assert w1 < 0.01 and w2 < 0.01 and not led.fits()
        t3 = asyncio.create_task(g.admit("p3"))
        t4 = asyncio.create_task(g.admit("p4"))
        await asyncio.sleep(0.03)
        assert g.queued == 2 and not t3.done()
        g.release("p1")
        await asyncio.sleep(0.03)
        assert t3.done() and not t4.done()  # FIFO
        g.release("p2")
        await t4
        assert g.stats["admitted"] == 4
        t4b = asyncio.create_task(g.admit("p5"))
        await asyncio.sleep(0.02)
        t4b.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t4b
        assert g.queued == 0

    run(go())


def test_dispatch_rule():
    pol = DispatchPolicy(beta=0.5, w0=10.0)
    waits = {"strong": 20.0, "weak": 0.0}
    # gain = 0.5 * 20/10 = 1.0 ; loss = 0.5 * (0.8 - 0.6) = 0.1 -> divert
    assert pol.choose("strong", waits, {"strong": 0.8, "weak": 0.6})[0] == "weak"
    # no estimates -> never divert
    assert pol.choose("strong", waits, {})[0] == "strong"
    # accuracy gap too large
    assert pol.choose("strong", {"strong": 2.0, "weak": 0.0}, {"strong": 0.9, "weak": 0.2})[0] == "strong"
    assert DispatchPolicy(beta=0.0).choose("strong", waits, {"strong": 0.8, "weak": 0.6})[0] == "strong"
    assert DispatchPolicy(beta=1.0).choose("strong", waits, {})[0] == "weak"
    with pytest.raises(ValueError):
        DispatchPolicy(beta=2)


def test_disc_end_to_end(run):
    async def go():
        disc = DISC({"a": 100_000, "b": 100_000}, policy=DispatchPolicy(beta=0.9, w0=1.0), ledger_kwargs={"peak_prior": 50_000, "prior_weight": 100, "duration_prior_s": 5.0, "rho": 1.0})
        for i in range(2):
            b, _ = disc.dispatch("a", {"a": 0.7, "b": 0.6})
            await disc.admit(f"p{i}", b)
        assert disc.where == {"p0": "a", "p1": "a"}
        b, d = disc.dispatch("a", {"a": 0.7, "b": 0.6})
        assert b == "b" and d["diverted"] and disc.pending["b"] == 1
        await disc.admit("p2", b)
        assert disc.pending["b"] == 0 and disc.n_diverted == 1
        disc.update("p2", 70_000)
        assert disc.release("p2", 70_000) and disc.ledgers["b"].finished_n == 1
        assert disc.release("p2") is False
        s = disc.snapshot()
        assert s["backends"]["a"]["ledger"]["admitted"] == 2 and s["diverted"] == 1

    run(go())


def test_capacity_parsers():
    text = 'vllm:cache_config_info{block_size="16",num_gpu_blocks="1000"} 1.0\nvllm:gpu_cache_usage_perc{model="x"} 0.42\n'
    assert VLLMCapacity.parse(text) == (16_000, 0.42)
    assert SGLangCapacity.parse_metrics('sglang:token_usage{model="x"} 0.31\n') == 0.31
