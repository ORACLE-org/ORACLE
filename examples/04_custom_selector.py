"""Plug in your own routing policy in a few lines.

Run:  python examples/04_custom_selector.py
"""
import asyncio

from oracle import Router, ModelSelector, register_selector, make_selector, CallableVerifier
from oracle.routing import CallableSelector
from oracle import sim


# Option A: subclass and register, so configs can say `selector: length_rule`.
@register_selector("length_rule")
class LengthRule(ModelSelector):
    """Long requests go to the strong model, short ones to the weak one."""

    def __init__(self, models, cutoff=80):
        super().__init__(models)
        self.cutoff = cutoff

    def select(self, ctx):
        return self.models[0] if len(ctx.prompt) > self.cutoff else self.models[-1]

    def update(self, ctx, model, reward):
        pass  # does not learn; ORACLE still verifies and shows accuracy per model


# Option B: wrap plain functions.
def pick(ctx):
    return "strong" if "test" in ctx.prompt else "weak"


async def main():
    verifiers = {f: CallableVerifier(f, lambda o: float(o.success)) for f in sim.FAMILIES}
    for sel in [make_selector("length_rule", ["strong", "weak"], cutoff=60), CallableSelector(["strong", "weak"], pick)]:
        router = Router(["strong", "weak"], selector=sel, verifiers=verifiers)
        res = await sim.run(router, sim.make_tasks(200))
        print(f"{type(sel).__name__:>16}: {res.summary()}")


asyncio.run(main())
