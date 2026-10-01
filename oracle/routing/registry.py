"""Name -> selector class registry, so configs can say ``selector: linucb``.

Third-party packages can register selectors through the ``oracle.selectors``
entry-point group in their ``pyproject.toml``::

    [project.entry-points."oracle.selectors"]
    my_router = "my_pkg.routing:MySelector"
"""
from __future__ import annotations

from typing import Callable, Dict, Sequence, Type

from .base import ModelSelector

_REGISTRY: Dict[str, Type[ModelSelector]] = {}


def register_selector(name: str) -> Callable[[Type[ModelSelector]], Type[ModelSelector]]:
    def deco(cls: Type[ModelSelector]) -> Type[ModelSelector]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def _load_entry_points() -> None:
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        return
    try:
        eps = entry_points(group="oracle.selectors")
    except TypeError:  # python < 3.10 style
        eps = entry_points().get("oracle.selectors", [])  # type: ignore[attr-defined]
    for ep in eps:
        if ep.name not in _REGISTRY:
            try:
                _REGISTRY[ep.name] = ep.load()
            except Exception:  # pragma: no cover - a broken plugin must not break ORACLE
                pass


def available_selectors() -> Dict[str, Type[ModelSelector]]:
    from . import bandit, static, acrouter, routellm  # noqa: F401  (registers the built-ins)

    _load_entry_points()
    return dict(_REGISTRY)


def make_selector(name: str, models: Sequence[str], **kwargs) -> ModelSelector:
    """Instantiate a selector by name, e.g. ``make_selector("linucb", ["27b", "9b"], alpha=1.0)``."""
    reg = available_selectors()
    if name not in reg:
        raise KeyError(f"unknown selector {name!r}; available: {sorted(reg)}")
    return reg[name](models, **kwargs)
