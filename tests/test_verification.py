import asyncio

from oracle import Router, ProgramOutcome, RoutingContext
from oracle.verification import PrototypeVerifierSelector, CallableVerifier, ReportedVerifier, CommandVerifier, RuleVerifierSelector, HashingEmbedder


EX = {"swe": ["fix the failing test in the repo", "patch the bug in module so pytest passes", "implement the function from the issue"], "tau2": ["book a flight from SFO to JFK", "change my hotel reservation", "cancel the order and refund"]}


def test_prototype_selector_matches_family():
    vs = PrototypeVerifierSelector(EX, {"swe": "tests", "tau2": "state"})
    assert vs.select(RoutingContext("p", "please patch the failing test in repo foo")) == ("swe", "tests")
    assert vs.select(RoutingContext("p", "book me a flight to Paris and a hotel")) == ("tau2", "state")
    vs.add_task_type("terminal", ["compile the project in the container and run the benchmark"], verifier="container")
    assert vs.select(RoutingContext("p", "compile and run the benchmark in the container"))[1] == "container"


def test_rule_selector():
    vs = RuleVerifierSelector(lambda c: c.metadata["kind"], {"a": "va"})
    assert vs.select(RoutingContext("p", metadata={"kind": "a"})) == ("a", "va")
    assert vs.select(RoutingContext("p", metadata={"kind": "b"})) == ("b", "b")


def test_delayed_feedback_updates_selector_out_of_order(run):
    async def go():
        slow = CallableVerifier("slow", lambda o: _sleep_then(0.05, 1.0))
        fast = CallableVerifier("fast", lambda o: _sleep_then(0.0, 0.0))
        vs = RuleVerifierSelector(lambda c: c.metadata["v"])
        r = Router(["a", "b"], "ucb", verifiers={"slow": slow, "fast": fast}, verifier_selector=vs)
        r.bind("p1", "x", {"v": "slow"})
        r.bind("p2", "x", {"v": "fast"})
        t1 = r.complete("p1", ProgramOutcome("p1", cost=0.0))
        t2 = r.complete("p2", ProgramOutcome("p2", cost=0.0))
        assert r.lookup("p1") is None and r.feedback.pending == 2
        r2, r1 = await t2, await t1
        assert (r1, r2) == (1.0, 0.0)
        assert r.selector.n_update == 2 and r.feedback.completed == 2
        assert [h["program_id"] for h in r.feedback.history] == ["p2", "p1"]

    run(go())


async def _sleep_then(d, v):
    await asyncio.sleep(d)
    return v


def test_reward_and_reported(run):
    from oracle import Reward

    rw = Reward(alpha=0.5, cost_ref=2.0)
    assert rw(1.0, 1.0) == 0.5 * 1.0 - 0.5 * 0.5
    assert rw(1.0, 10.0) == 0.0  # cost clipped at 1
    r = Router(["a"], "fixed", verifiers={"rep": ReportedVerifier("rep")}, reward=rw)
    r.bind("p", "x")
    assert r.complete_sync("p", ProgramOutcome("p", cost=1.0, success=1.0)) == 0.25


def test_command_verifier(run):
    v = CommandVerifier("t", "python3 -c 'import sys; sys.exit(0 if {ok} else 1)'")
    assert run(v.verify(ProgramOutcome("p", payload={"ok": True}))) == 1.0
    assert run(v.verify(ProgramOutcome("p", payload={"ok": False}))) == 0.0


def test_embedder_unit_norm():
    e = HashingEmbedder(256).embed(["hello world", "hello world foo"])
    import numpy as np

    assert np.allclose(np.linalg.norm(e, axis=1), 1.0)
