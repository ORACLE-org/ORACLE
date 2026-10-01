"""Router + DISC together (the paper's default ORACLE), in-process.

Run:  python examples/03_router_and_disc.py
"""
import asyncio

from oracle import Oracle, Router, DISC, DispatchPolicy, PrototypeVerifierSelector
from oracle.verification import ReportedVerifier
from oracle import sim

MODELS = ["strong", "weak"]
router = Router(MODELS, "linucb", verifiers={f: ReportedVerifier(f) for f in sim.FAMILIES}, verifier_selector=PrototypeVerifierSelector(sim.exemplars()))
disc = DISC({m: 150_000 for m in MODELS}, policy=DispatchPolicy(beta=0.5, w0=1.0), ledger_kwargs={"peak_prior": 40_000})
oracle = Oracle(router, disc)


async def main():
    await oracle.start()
    res = await sim.run(oracle, sim.make_tasks(300), concurrency=24, time_scale=0.03)
    print(res.summary())
    s = oracle.snapshot()
    for name, b in s["scheduler"]["backends"].items():
        print(f"{name}: admitted now {b['ledger']['admitted']}, wait p95 {b['wait_p95_s']}s, learned peak {b['ledger']['c_hat']}")
    await oracle.stop()


asyncio.run(main())
