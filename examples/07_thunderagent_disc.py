"""Run DISC inside ThunderAgent (pip install -e ThunderAgent first).

Launch ThunderAgent from this script instead of `thunderagent ...`; everything
else (clients, program_id, backends) stays the same.

    python examples/07_thunderagent_disc.py --backends http://localhost:8001,http://localhost:8002 --port 9000
"""
import argparse

import uvicorn
from ThunderAgent.config import Config, set_config
from ThunderAgent.scheduler import MultiBackendRouter

from oracle.integrations.thunderagent import install
from oracle.scheduling import DISC, VLLMCapacity

p = argparse.ArgumentParser()
p.add_argument("--backends", required=True)
p.add_argument("--port", type=int, default=9000)
p.add_argument("--rho", type=float, default=0.85)
args = p.parse_args()
urls = [u.strip() for u in args.backends.split(",")]

set_config(Config(backends=urls, router_mode="tr"))
from ThunderAgent import app as ta_app  # noqa: E402  (reads the config at import)

ta_router: MultiBackendRouter = ta_app.ta_router
disc = DISC({u: VLLMCapacity(u) for u in urls}, policy=None, ledger_kwargs={"rho": args.rho})  # admission only
install(ta_router, disc)


@ta_app.app.get("/disc")
async def disc_state():
    return disc.snapshot()


uvicorn.run(ta_app.app, host="0.0.0.0", port=args.port)
