"""The OpenAI-compatible proxy.

``create_app`` needs the ``server`` extra (FastAPI, httpx); the request-field
helpers (``get_program_id``, ``first_user_text``) do not and are imported
eagerly so that the integrations work without it.
"""
from .fields import first_user_text, get_program_id

__all__ = ["create_app", "get_program_id", "first_user_text"]


def __getattr__(name: str):
    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
