"""Configuration: a dataclass tree that maps 1:1 onto ``oracle.yaml``.

Everything optional has a working default; the smallest useful file is::

    models:
      - name: strong
        url: http://localhost:8001/v1
      - name: weak
        url: http://localhost:8002/v1

See ``examples/configs/`` for the full set of knobs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    name: str
    """Name used by the router (and by clients in the ``model`` field, optional)."""
    url: str = ""
    """OpenAI-compatible base URL of the backend serving this model (ends with /v1)."""
    served_name: Optional[str] = None
    """The ``model`` string the backend expects; defaults to ``name``."""
    backend: Optional[str] = None
    """Scheduler backend name; defaults to ``name`` (one model per backend)."""
    price_in: float = 0.0
    """Dollars per 1M prompt tokens (for the cost term and the dashboard)."""
    price_out: float = 0.0
    """Dollars per 1M completion tokens."""
    capacity: str = "auto"
    """``auto`` (detect vllm/sglang from the backend), ``vllm``, ``sglang`` or ``static``."""
    capacity_tokens: Optional[int] = None
    """KV-cache capacity in tokens for ``static``."""
    api_key: str = "EMPTY"


@dataclass
class RouterConfig:
    enabled: bool = True
    selector: str = "linucb"
    selector_kwargs: Dict[str, Any] = field(default_factory=dict)
    features: str = "hashing"
    """``hashing`` | ``embedding`` (uses the verification embedder) | ``task_type``."""
    feature_dim: int = 64
    reward_alpha: float = 0.0
    reward_cost_ref: float = 1.0
    max_verify_concurrency: int = 8


@dataclass
class VerifierConfig:
    name: str
    type: str = "reported"
    """``reported`` | ``command`` | ``http`` | ``llm_judge`` | ``python`` (``target``: ``module:function``)."""
    exemplars: List[str] = field(default_factory=list)
    """Example initial requests of the tasks this verifier should score (for adaptive verification)."""
    task_type: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationConfig:
    verifiers: List[VerifierConfig] = field(default_factory=list)
    embedder: str = "hashing"
    """``hashing`` | ``st:<sentence-transformers model>`` | ``openai:<model>`` (+ ``embedder_url``)."""
    embedder_url: Optional[str] = None
    min_sim: float = -1.0
    default: Optional[str] = None
    metadata_key: Optional[str] = None
    """If set, the task type is read from ``metadata[<key>]`` instead of embeddings."""


@dataclass
class SchedulerConfig:
    enabled: bool = True
    beta: float = 0.5
    w0: float = 60.0
    rho: float = 0.85
    peak_prior: int = 20000
    prior_weight: int = 5
    floor: float = 1.0
    per_program_overhead: int = 0
    duration_prior_s: float = 30.0
    max_programs: int = 0
    poll_s: float = 2.0
    max_wait_s: float = 0.0


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8300
    program_id_header: str = "X-Program-ID"
    done_key: str = "program_done"
    """Request field (top level or in ``extra_body``) that marks the program's last request."""
    request_timeout_s: float = 900.0
    ui: bool = True
    log_level: str = "info"


@dataclass
class OracleConfig:
    models: List[ModelConfig] = field(default_factory=list)
    router: RouterConfig = field(default_factory=RouterConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "OracleConfig":
        d = dict(d or {})
        models = [ModelConfig(**m) if isinstance(m, dict) else ModelConfig(name=str(m)) for m in d.get("models", [])]
        ver = dict(d.get("verification", {}))
        ver["verifiers"] = [VerifierConfig(**v) for v in ver.get("verifiers", [])]
        return OracleConfig(
            models=models,
            router=RouterConfig(**d.get("router", {})),
            verification=VerificationConfig(**ver),
            scheduler=SchedulerConfig(**d.get("scheduler", {})),
            server=ServerConfig(**d.get("server", {})),
        )

    @staticmethod
    def load(path: str) -> "OracleConfig":
        import os

        import yaml

        with open(path) as f:
            text = os.path.expandvars(f.read())  # ${API_KEY} etc.
        return OracleConfig.from_dict(yaml.safe_load(text) or {})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Building objects from the config
# ---------------------------------------------------------------------------
def _import_target(target: str):
    mod, _, attr = target.partition(":")
    import importlib

    return getattr(importlib.import_module(mod), attr)


def build_verifiers(cfg: VerificationConfig):
    from .verification import base as vb

    out = {}
    for v in cfg.verifiers:
        o = dict(v.options)
        if v.type == "reported":
            out[v.name] = vb.ReportedVerifier(v.name, **o)
        elif v.type == "command":
            out[v.name] = vb.CommandVerifier(v.name, **o)
        elif v.type == "http":
            out[v.name] = vb.HTTPVerifier(v.name, **o)
        elif v.type == "llm_judge":
            out[v.name] = vb.LLMJudgeVerifier(v.name, **o)
        elif v.type == "python":
            fn = _import_target(o.pop("target"))
            out[v.name] = fn(v.name, **o) if isinstance(fn, type) else vb.CallableVerifier(v.name, fn)
        else:
            raise ValueError(f"unknown verifier type {v.type!r}")
    return out


def build_embedder(cfg: VerificationConfig):
    from .verification.embeddings import make_embedder

    kw = {"url": cfg.embedder_url} if cfg.embedder.startswith("openai:") else {}
    return make_embedder(cfg.embedder, **kw)


def build_verifier_selector(cfg: VerificationConfig, embedder=None):
    from .verification.selector import FixedVerifierSelector, PrototypeVerifierSelector, RuleVerifierSelector

    names = [v.name for v in cfg.verifiers]
    if not names:
        return FixedVerifierSelector("reported")
    if cfg.metadata_key:
        key = cfg.metadata_key
        mapping = {(v.task_type or v.name): v.name for v in cfg.verifiers}
        return RuleVerifierSelector(lambda ctx: str(ctx.metadata.get(key, names[0])), mapping)
    with_ex = [v for v in cfg.verifiers if v.exemplars]
    if len(with_ex) < 2:
        return FixedVerifierSelector(names[0], (cfg.verifiers[0].task_type or names[0]))
    exemplars = {(v.task_type or v.name): v.exemplars for v in with_ex}
    verifier_of = {(v.task_type or v.name): v.name for v in with_ex}
    return PrototypeVerifierSelector(exemplars, verifier_of, embedder=embedder or build_embedder(cfg), min_sim=cfg.min_sim, default=cfg.default)


def build_router(cfg: OracleConfig):
    from .routing.features import ConcatFeatures, EmbeddingFeatures, HashingFeatures, TaskTypeFeatures
    from .routing.reward import Reward
    from .routing.router import Router

    names = [m.name for m in cfg.models]
    embedder = build_embedder(cfg.verification)
    vsel = build_verifier_selector(cfg.verification, embedder)
    task_types = [(v.task_type or v.name) for v in cfg.verification.verifiers]
    if cfg.router.features == "hashing":
        feats = HashingFeatures(cfg.router.feature_dim)
    elif cfg.router.features == "embedding":
        feats = EmbeddingFeatures(embedder, dim=cfg.router.feature_dim)
    elif cfg.router.features == "task_type":
        feats = TaskTypeFeatures(task_types)
    elif cfg.router.features == "hashing+task_type":
        feats = ConcatFeatures(HashingFeatures(cfg.router.feature_dim), TaskTypeFeatures(task_types))
    else:
        raise ValueError(f"unknown features {cfg.router.features!r}")
    skw = dict(cfg.router.selector_kwargs)
    if cfg.router.selector == "acrouter":
        skw.setdefault("embedder", embedder)
    return Router(
        names,
        selector=cfg.router.selector,
        selector_kwargs=skw,
        verifiers=build_verifiers(cfg.verification),
        verifier_selector=vsel,
        features=feats,
        reward=Reward(cfg.router.reward_alpha, cfg.router.reward_cost_ref),
        max_verify_concurrency=cfg.router.max_verify_concurrency,
    )


def build_scheduler(cfg: OracleConfig):
    from .scheduling.capacity import SGLangCapacity, StaticCapacity, VLLMCapacity
    from .scheduling.disc import DISC
    from .scheduling.dispatch import DispatchPolicy

    s = cfg.scheduler
    backends: Dict[str, Any] = {}
    for m in cfg.models:
        b = m.backend or m.name
        if b in backends:
            continue
        if m.capacity == "static" or (m.capacity == "auto" and m.capacity_tokens):
            backends[b] = StaticCapacity(m.capacity_tokens or 0)
        elif m.capacity == "sglang":
            backends[b] = SGLangCapacity(m.url)
        elif m.capacity in ("vllm", "auto"):
            backends[b] = VLLMCapacity(m.url) if m.url else StaticCapacity(0)
        else:
            raise ValueError(f"unknown capacity {m.capacity!r}")
    policy = DispatchPolicy(beta=s.beta, w0=s.w0) if s.beta > 0 else None
    return DISC(
        backends,
        policy=policy,
        ledger_kwargs=dict(rho=s.rho, peak_prior=s.peak_prior, prior_weight=s.prior_weight, floor=s.floor, per_program_overhead=s.per_program_overhead, duration_prior_s=s.duration_prior_s, max_programs=s.max_programs),
        poll_s=s.poll_s,
        max_wait_s=s.max_wait_s,
    )


def build(cfg: OracleConfig):
    from .core import Oracle

    if not cfg.models:
        raise ValueError("config needs at least one model")
    router = build_router(cfg)
    sched = build_scheduler(cfg) if cfg.scheduler.enabled else None
    return Oracle(router, sched, backend_of={m.name: (m.backend or m.name) for m in cfg.models})
