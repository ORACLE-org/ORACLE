# Routing

## `Router`

```python
from oracle import Router
router = Router(models, selector="linucb", selector_kwargs={}, verifiers={}, verifier_selector=None,
                features=None, reward=None, max_verify_concurrency=8)
```

| Method | What it does |
|---|---|
| `bind(program_id, prompt, metadata=None, model=None)` | create the program's row: verifier selection, features, selector choice (or the given `model`) |
| `lookup(program_id)` | the row, or `None` |
| `complete(program_id, outcome)` | remove the row, verify in the background, update the selector; returns the asyncio task |
| `complete_sync(...)` | same, blocking (for scripts without an event loop) |
| `release(program_id)` | drop without feedback |
| `context(...)`, `propose(ctx)`, `estimates(ctx)` | the pieces `bind` is made of (used by DISC) |
| `state()` | JSON for the dashboard |

`models` is an ordered list, strongest (most expensive) first. Several built-ins use that order
for tie-breaks and fallbacks.

## Model selectors (the pluggable router backend)

A selector implements `oracle.routing.ModelSelector`:

```python
class ModelSelector:
    def select(self, ctx: RoutingContext) -> str: ...                   # required
    def update(self, ctx: RoutingContext, model: str, reward: float): ...  # required
    def estimate(self, ctx, model) -> float | None: ...                 # optional, lets DISC price a diversion
    def state(self) -> dict: ...                                        # optional, dashboard
```

`RoutingContext` carries `program_id`, `prompt` (the initial request), `features` (a vector),
`task_type` (from the verifier selector) and `metadata` (whatever the client sent in
`program_metadata`).

Built-ins:

| name | class | learns | context | notes |
|---|---|---|---|---|
| `linucb` | `LinUCB` | yes | feature vector | default; `alpha_ucb` width, `lam` ridge, `dim` |
| `lints` | `LinTS` | yes | feature vector | Thompson sampling, `v` |
| `ucb` | `TypeUCB` | yes | task type | UCB1 per (task type, model); `delta`, `prior` |
| `epsilon_greedy` | `EpsilonGreedy` | yes | task type | `epsilon` |
| `acrouter` | `ACRouterSelector` | yes (memory) | prompt embedding | neighbours `k`, `min_sim`, optional `orchestrator_url` for the LLM step, `explore` |
| `routellm` | `RouteLLMSelector` | no | prompt | `router` (mf, bert, ...), `threshold`; needs `routellm` |
| `fixed` | `Fixed` | no | - | `model` |
| `round_robin`, `random` | | no | - | `random` takes `weights` |
| `callable` | `CallableSelector` | optional | - | wrap `select_fn`, `update_fn`, `estimate_fn` |

```python
oracle selectors            # prints this list with one-line docs
```

### Writing your own

```python
from oracle import ModelSelector, register_selector

@register_selector("my_router")
class MyRouter(ModelSelector):
    def __init__(self, models, threshold=0.5):
        super().__init__(models)
        self.threshold = threshold
    def select(self, ctx):
        return self.models[0] if score(ctx.prompt) > self.threshold else self.models[-1]
    def update(self, ctx, model, reward):
        ...
```

Then `selector: my_router` + `selector_kwargs: {threshold: 0.7}` in the YAML, or
`make_selector("my_router", models, threshold=0.7)`. To ship it in a package, add

```toml
[project.entry-points."oracle.selectors"]
my_router = "my_pkg.routing:MyRouter"
```

`CallableSelector(models, select_fn, update_fn=None)` wraps a function when subclassing is too much.

### Wrapping an external router service

Make `select` call it (HTTP, gRPC, a local model) and map its answer to one of `self.models`.
`ACRouterSelector` is an example: it calls an OpenAI-compatible orchestrator endpoint and falls
back to its memory rule on failure.

## Features

Contextual selectors read `ctx.features`, produced by a `FeatureExtractor` (`__call__(prompt, metadata) -> np.ndarray`, with a `dim`):

* `HashingFeatures(dim=64)`: hashed bag of words + length + bias. Default, no dependencies.
* `EmbeddingFeatures(embedder, dim=None)`: any embedder from `oracle.verification` (sentence-transformers, an `/v1/embeddings` endpoint), optionally random-projected to `dim`.
* `TaskTypeFeatures(task_types)`: one-hot of the verifier selector's task type.
* `ConcatFeatures(a, b, ...)`.

`Router` passes `dim` to `linucb`/`lints` automatically.

## Reward

`Reward(alpha=0.0, cost_ref=1.0)` implements

```
r = (1 - alpha) * score - alpha * min(cost / cost_ref, 1)
```

`score` comes from the verifier, `cost` from the proxy (prices per token in the model config) or
from the outcome you post. `alpha = 0` learns on accuracy alone.
