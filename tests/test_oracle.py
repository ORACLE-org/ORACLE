import pytest

from oracle import Oracle, Router, DISC, DispatchPolicy, ProgramOutcome, PrototypeVerifierSelector
from oracle.verification import ReportedVerifier
from oracle import sim


def test_oracle_binding_and_feedback(run):
    async def go():
        router = Router(["strong", "weak"], "ucb", verifiers={f: ReportedVerifier(f) for f in sim.FAMILIES}, verifier_selector=PrototypeVerifierSelector(sim.exemplars()))
        disc = DISC({"strong": 100_000, "weak": 100_000}, policy=DispatchPolicy(beta=0.5, w0=1.0), ledger_kwargs={"peak_prior": 20_000})
        o = Oracle(router, disc)
        b = await o.bind("t1", "fix the failing test in repo x")
        assert b.verifier == "swe" and b.model in ("strong", "weak") and o.lookup("t1") is b
        assert await o.bind("t1", "ignored") is b
        o.observe_request("t1", 3000, 100, 0.01)
        assert b.requests == 1 and b.peak_context_tokens == 3100 and disc.ledgers[b.model].admitted["t1"]["tokens"] == 3100
        t = o.complete("t1", ProgramOutcome("t1", success=1.0))
        assert o.lookup("t1") is None and "t1" not in disc.where
        assert await t == 1.0 and b.cost == pytest.approx(0.01)
        assert o.release("nope") is False

    run(go())


def test_sim_learning_beats_fixed_weak(run):
    async def go():
        tasks = sim.make_tasks(400, seed=1)
        out = {}
        for name in ("fixed", "linucb"):
            r = Router(["strong", "weak"], name, verifiers={f: ReportedVerifier(f) for f in sim.FAMILIES}, verifier_selector=PrototypeVerifierSelector(sim.exemplars()), selector_kwargs={"model": "weak"} if name == "fixed" else {})
            out[name] = await sim.run(r, tasks, concurrency=8, seed=1)
        assert out["linucb"].accuracy > out["fixed"].accuracy
        assert out["linucb"].per_model.get("strong", 0) > 0 and out["linucb"].per_model.get("weak", 0) > 0

    run(go())


def test_config_build(tmp_path):
    from oracle.config import OracleConfig, build

    cfg = OracleConfig.from_dict({
        "models": [{"name": "s", "url": "http://x/v1", "capacity_tokens": 1000}, {"name": "w", "url": "http://y/v1", "capacity_tokens": 1000}],
        "router": {"selector": "ucb", "features": "task_type"},
        "verification": {"verifiers": [{"name": "a", "type": "reported", "exemplars": ["fix the test"]}, {"name": "b", "type": "reported", "exemplars": ["book a flight"]}]},
        "scheduler": {"beta": 0.3},
    })
    o = build(cfg)
    assert o.router.selector.name == "ucb" and set(o.scheduler.ledgers) == {"s", "w"} and o.scheduler.policy.beta == 0.3
    p = tmp_path / "c.yaml"
    p.write_text("models:\n  - name: a\n    url: http://a/v1\n    api_key: ${ORACLE_TEST_KEY}\nscheduler: {enabled: false}\n")
    import os

    os.environ["ORACLE_TEST_KEY"] = "sekret"
    c2 = OracleConfig.load(str(p))
    assert c2.models[0].api_key == "sekret" and build(c2).scheduler is None
