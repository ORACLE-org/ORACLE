"""DISC alone: task-boundary admission control over two backends.  No router.

Run:  python examples/02_scheduler_only.py
"""
import asyncio
import random

from oracle import DISC, DispatchPolicy

# Two backends with a 100k-token KV cache each.  Capacity can also come from
# VLLMCapacity(url) / SGLangCapacity(url) so it is read from the engine.
disc = DISC({"gpu-a": 100_000, "gpu-b": 100_000}, policy=DispatchPolicy(beta=0.5, w0=5.0), ledger_kwargs={"peak_prior": 30_000, "duration_prior_s": 0.1})


async def program(i: int, rng: random.Random):
    # The router (or you) proposes a backend; DISC may divert to the one with the shorter forecast wait
    # when the accuracy estimates say it is worth it.  Here both backends are equally good.
    backend, detail = disc.dispatch("gpu-a", estimates={"gpu-a": 0.6, "gpu-b": 0.6})
    waited = await disc.admit(f"p{i}", backend)  # blocks until the ledger has room
    peak = int(rng.uniform(15_000, 45_000))
    disc.update(f"p{i}", peak)  # report context growth as requests return
    await asyncio.sleep(rng.uniform(0.05, 0.15))  # the agent runs
    disc.release(f"p{i}", peak_tokens=peak)  # frees the reservation, learns the peak and duration
    return backend, waited


async def main():
    rng = random.Random(0)
    results = await asyncio.gather(*(program(i, rng) for i in range(40)))
    by = {}
    for b, w in results:
        by.setdefault(b, []).append(w)
    for b, ws in by.items():
        print(f"{b}: {len(ws)} programs, mean wait {sum(ws) / len(ws):.2f}s, max {max(ws):.2f}s")
    print("diverted:", disc.n_diverted, "of", disc.n_dispatch)
    print("learned expected peak:", {n: round(l.c_hat) for n, l in disc.ledgers.items()})


asyncio.run(main())
