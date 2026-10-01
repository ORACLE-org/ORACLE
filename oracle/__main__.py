"""``oracle`` command line.

    oracle serve --config oracle.yaml
    oracle serve --models strong=http://localhost:8001/v1 weak=http://localhost:8002/v1
    oracle serve ... --no-scheduler        # routing only
    oracle serve ... --router none         # scheduling only (DISC in front of your backends)
    oracle demo                            # synthetic workload, no GPUs
    oracle selectors                       # list pluggable router backends
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from typing import List

from .config import ModelConfig, OracleConfig, build


def _parse_models(specs: List[str]) -> List[ModelConfig]:
    out = []
    for s in specs:
        name, _, url = s.partition("=")
        if not url:
            raise SystemExit(f"--models expects name=url, got {s!r}")
        out.append(ModelConfig(name=name, url=url))
    return out


def cmd_serve(args: argparse.Namespace) -> int:
    cfg = OracleConfig.load(args.config) if args.config else OracleConfig()
    if args.models:
        cfg.models = _parse_models(args.models)
    if not cfg.models:
        raise SystemExit("no models: pass --config or --models name=url ...")
    if args.selector:
        cfg.router.selector = args.selector
    if args.no_scheduler:
        cfg.scheduler.enabled = False
    if args.beta is not None:
        cfg.scheduler.beta = args.beta
    if args.capacity_tokens:
        for m in cfg.models:
            m.capacity, m.capacity_tokens = "static", args.capacity_tokens
    if args.port:
        cfg.server.port = args.port
    if args.host:
        cfg.server.host = args.host
    if args.router == "none":
        cfg.router.selector = "fixed"  # scheduler-only: every program goes to its requested/first model
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required: pip install 'oracle-router[server]'", file=sys.stderr)
        return 1
    from .server.app import create_app

    oracle = build(cfg)
    app = create_app(oracle, cfg.models, cfg.server)
    mode = ("router" if cfg.router.selector != "fixed" or args.router != "none" else "") + (" + DISC" if cfg.scheduler.enabled else "")
    print(f"ORACLE {mode.strip() or 'proxy'} | selector={cfg.router.selector} | models={[m.name for m in cfg.models]} | dashboard http://{cfg.server.host}:{cfg.server.port}/ui")
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level=cfg.server.log_level)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from . import sim
    from .core import Oracle
    from .routing.router import Router
    from .scheduling.disc import DISC
    from .scheduling.dispatch import DispatchPolicy
    from .verification.base import ReportedVerifier
    from .verification.selector import PrototypeVerifierSelector

    models = ["strong", "weak"]
    verifiers = {f: ReportedVerifier(f) for f in sim.FAMILIES}
    tasks = sim.make_tasks(args.programs, seed=args.seed)

    async def go():
        for name in [n.strip() for n in args.selectors.split(",") if n.strip()]:
            try:
                router = Router(models, name, verifiers=verifiers, verifier_selector=PrototypeVerifierSelector(sim.exemplars()))
            except (KeyError, ImportError, ValueError) as e:  # unknown selector, missing optional dependency, bad kwargs
                raise SystemExit(f"demo: cannot build selector {name!r}: {e.args[0] if e.args else e}")
            system = router if args.no_scheduler else Oracle(router, DISC({m: args.capacity_tokens for m in models}, policy=DispatchPolicy(beta=args.beta, w0=args.w0)))
            r = await sim.run(system, tasks, concurrency=args.concurrency, seed=args.seed, time_scale=args.time_scale)
            print(f"{name:>15}: {r.summary()}")

    asyncio.run(go())
    return 0


def cmd_selectors(args: argparse.Namespace) -> int:
    from .routing.registry import available_selectors

    for n, c in sorted(available_selectors().items()):
        doc = (c.__doc__ or "").strip().splitlines()[0] if c.__doc__ else ""
        print(f"{n:>15}  {doc}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="oracle", description="ORACLE: concurrency-aware routing and scheduling for agentic programs")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the OpenAI-compatible proxy")
    s.add_argument("--config", "-c", help="oracle.yaml")
    s.add_argument("--models", nargs="*", help="name=url pairs (OpenAI-compatible base URLs ending in /v1)")
    s.add_argument("--selector", help="router backend: linucb (default), ucb, acrouter, routellm, fixed, ...")
    s.add_argument("--router", choices=["oracle", "none"], default="oracle", help="'none' = scheduling only")
    s.add_argument("--no-scheduler", action="store_true", help="routing only")
    s.add_argument("--beta", type=float, help="DISC accuracy/wait trade-off in [0,1]; 0 = admission only")
    s.add_argument("--capacity-tokens", type=int, help="static KV capacity per backend (skips metrics polling)")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.set_defaults(fn=cmd_serve)

    d = sub.add_parser("demo", help="run a synthetic workload through the router and DISC")
    d.add_argument("--programs", type=int, default=400)
    d.add_argument("--concurrency", type=int, default=16)
    d.add_argument("--selectors", default="fixed,random,ucb,linucb,acrouter")
    d.add_argument("--no-scheduler", action="store_true")
    d.add_argument("--capacity-tokens", type=int, default=200_000)
    d.add_argument("--beta", type=float, default=0.5)
    d.add_argument("--w0", type=float, default=30.0)
    d.add_argument("--time-scale", type=float, default=0.02, help="seconds of simulated execution per program")
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=cmd_demo)

    ls = sub.add_parser("selectors", help="list available router backends")
    ls.set_defaults(fn=cmd_selectors)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
