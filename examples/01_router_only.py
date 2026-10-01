"""Routing only: bind programs to models, learn from verified outcomes.  No server, no GPUs.

Run:  python examples/01_router_only.py
"""
import asyncio

from oracle import Router, ProgramOutcome, CallableVerifier, PrototypeVerifierSelector
from oracle import sim

MODELS = ["strong", "weak"]  # strongest first

# One verifier per task family. Here the harness reports success itself; in real use
# "swe" would run the repo's tests and "tau2" would check the environment state.
verifiers = {fam: CallableVerifier(fam, lambda o: float(o.success)) for fam in sim.FAMILIES}

# Adaptive verification: a few example requests per family build the prototypes.
vsel = PrototypeVerifierSelector(sim.exemplars())

router = Router(MODELS, selector="linucb", verifiers=verifiers, verifier_selector=vsel)


async def main():
    tasks = sim.make_tasks(300)
    res = await sim.run(router, tasks, concurrency=16)
    print(res.summary())
    print("selector state:", router.selector.state()["pulls"])
    print("verifier counts:", vsel.state()["counts"])

    # The three calls you would make from your own harness:
    b = router.bind("my-task", "fix the failing test in repo foo")
    print("bound", b.program_id, "->", b.model, "verifier", b.verifier)
    t = router.complete("my-task", ProgramOutcome(program_id="my-task", success=1.0, cost=0.11))
    print("reward", await t)


asyncio.run(main())
