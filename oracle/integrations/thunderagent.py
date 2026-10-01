"""DISC inside ThunderAgent (https://github.com/ThunderAgent-org/ThunderAgent).

A program is admitted to a backend only when the backend's reservation ledger
has room for its expected peak, and its reservation is released when the
program is released.  Nothing in ThunderAgent is modified; ``install`` wraps
three of its router methods.

Usage (in your own launcher, after creating ThunderAgent's router)::

    from ThunderAgent.scheduler import MultiBackendRouter
    from oracle.scheduling import DISC, DispatchPolicy
    from oracle.integrations.thunderagent import install

    ta = MultiBackendRouter(backend_urls, ...)
    disc = DISC({url: VLLMCapacity(url) for url in backend_urls}, policy=None)   # admission only
    install(ta, disc)

With ``policy=DispatchPolicy(beta=...)`` and a ``propose`` callback, DISC also
chooses the backend at the program boundary (the host's own assignment is the
proposal by default).
"""
from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Optional

from ..scheduling.disc import DISC


def install(ta_router: Any, disc: DISC, propose: Optional[Callable[[str, Dict[str, Any]], str]] = None, estimates: Optional[Callable[[str, Dict[str, Any]], Dict[str, Optional[float]]]] = None) -> DISC:
    """Wrap ThunderAgent's ``MultiBackendRouter`` so every program passes DISC's gate at its first request.

    Args:
        ta_router: a ``ThunderAgent.scheduler.MultiBackendRouter`` instance.
        disc: a DISC whose backend names are ThunderAgent's backend URLs.
        propose: optional ``(program_id, payload) -> backend_url`` proposal (e.g. from an ORACLE Router);
            defaults to the host's own assignment.
        estimates: optional ``(program_id, payload) -> {backend_url: accuracy estimate}`` for the dispatch rule.
    """
    missing = [u for u in ta_router.backends if u not in disc.ledgers]
    if missing:
        raise ValueError(f"DISC has no ledger for ThunderAgent backends {missing}")

    orig_before = ta_router.update_program_before_request
    orig_after = ta_router.update_program_after_request
    orig_release = ta_router.release_program

    @functools.wraps(orig_before)
    async def before(program_id: str, program_state: Any, payload: Dict[str, Any]):
        if program_id not in disc.where:
            proposed = propose(program_id, payload) if propose else (program_state.backend_url or ta_router.get_backend_for_program(program_id) or next(iter(ta_router.backends)))
            backend, _ = disc.dispatch(proposed, estimates(program_id, payload) if estimates else None)
            program_state.backend_url = backend
            program_state.origin_backend = backend
            await disc.admit(program_id, backend, int(getattr(program_state, "total_tokens", 0) or 0))
        return await orig_before(program_id, program_state, payload)

    @functools.wraps(orig_after)
    def after(program_id: str, program_state: Any, total_tokens: int, *a, **kw):
        disc.update(program_id, int(total_tokens))
        return orig_after(program_id, program_state, total_tokens, *a, **kw)

    @functools.wraps(orig_release)
    async def release(program_id: str, *a, **kw):
        disc.release(program_id)
        return await orig_release(program_id, *a, **kw)

    ta_router.update_program_before_request = before
    ta_router.update_program_after_request = after
    ta_router.release_program = release
    ta_router.disc = disc
    return disc
