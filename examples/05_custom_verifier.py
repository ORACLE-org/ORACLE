"""Verifiers: anything that scores a finished program, run off the critical path.

Run:  python examples/05_custom_verifier.py
"""
import asyncio

from oracle import Router, ProgramOutcome
from oracle.verification import CallableVerifier, CommandVerifier, PrototypeVerifierSelector


async def check_state(outcome: ProgramOutcome) -> float:
    """A tau2-style state checker: compare the environment's final state to the expected one."""
    await asyncio.sleep(0.01)  # pretend to query the environment
    return 1.0 if outcome.payload.get("final_state") == outcome.payload.get("expected_state") else 0.0


verifiers = {
    # patch verifier: run the tests; exit code 0 -> 1.0
    "tests": CommandVerifier("tests", "python3 -c 'import sys; sys.exit(0 if {expect_pass} else 1)'"),
    "state": CallableVerifier("state", check_state),
}
vsel = PrototypeVerifierSelector(
    {"swe": ["fix the failing test in the repo", "patch the bug so pytest passes"], "tau2": ["book a flight for the user", "update the hotel reservation"]},
    verifier_of={"swe": "tests", "tau2": "state"},
)
router = Router(["strong", "weak"], "ucb", verifiers=verifiers, verifier_selector=vsel)


async def main():
    b1 = router.bind("t1", "please fix the failing test in repo foo")
    b2 = router.bind("t2", "book a flight to Paris for next week")
    print("t1 ->", b1.model, "verified by", b1.verifier, "| t2 ->", b2.model, "verified by", b2.verifier)
    r1 = router.complete("t1", ProgramOutcome(program_id="t1", cost=0.1, payload={"expect_pass": True}))
    r2 = router.complete("t2", ProgramOutcome(program_id="t2", cost=0.02, payload={"final_state": "booked", "expected_state": "booked"}))
    print("rewards:", await r1, await r2)
    print("feedback:", router.feedback.state()["recent_accuracy"])


asyncio.run(main())
