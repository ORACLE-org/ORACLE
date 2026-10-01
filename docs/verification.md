# Verification

A verifier scores a *finished* program in [0, 1]. ORACLE runs it after the program's slot is
released, so verification never delays the next program.

## Verifiers

```python
from oracle.verification import (ReportedVerifier, CallableVerifier, CommandVerifier,
                                 HTTPVerifier, LLMJudgeVerifier)
```

| class | config `type` | scores by |
|---|---|---|
| `ReportedVerifier(name, default=0.0)` | `reported` | `outcome.success` as posted by the harness |
| `CallableVerifier(name, fn)` | `python` (`target: module:function`) | your sync or async function `fn(outcome) -> float` |
| `CommandVerifier(name, command, timeout_s, parse_score)` | `command` | a shell command; exit 0 = 1.0, or the last stdout line as a float |
| `HTTPVerifier(name, url)` | `http` | `POST url` with the outcome, expects `{"score": x}` |
| `LLMJudgeVerifier(name, url, model)` | `llm_judge` | an OpenAI-compatible judge; `payload.task`, `payload.answer`, optional `payload.reference` |

`ProgramOutcome` carries `program_id`, `cost`, optional `success`, and a free-form `payload` (the
command template and HTTP verifier see it: `"docker exec {container} pytest -q"`). Payload values
usually come from the client that reported the outcome, so `CommandVerifier` shell-quotes each one
before substituting it (`quote=False` turns that off for templates whose values you control).
`Router(verifiers=...)` also accepts plain `fn(outcome) -> score` functions and wraps them.

Subclass `Verifier` for anything else: one `async verify(outcome) -> float`.

## Choosing the verifier per program

`VerifierSelector.select(ctx) -> (task_type, verifier_name)` runs at the program's first request.

* `PrototypeVerifierSelector(exemplars, verifier_of=None, embedder=None, min_sim=-1, default=None)`:
  the paper's adaptive verification. `exemplars = {"swe": [...requests...], "tau2": [...]}`; each
  task type's prototype is the mean embedding of its examples; the nearest prototype wins.
  `add_task_type(name, exemplars, verifier)` extends it at runtime. Default embedder is the
  dependency-free `HashingEmbedder`; use `SentenceTransformerEmbedder("BAAI/bge-m3")` or
  `OpenAIEmbedder(url, model)` for better separation.
* `RuleVerifierSelector(fn, mapping)`: the harness tells you (`program_metadata: {task_type: swe}`).
  In YAML: `verification.metadata_key: task_type`.
* `FixedVerifierSelector(name)`: one verifier for everything.

The chosen task type is also available to selectors as `ctx.task_type` (`ucb` uses it as its
context) and to features via `TaskTypeFeatures`.

## Delayed feedback

`FeedbackLoop` (owned by `Router.feedback`):

1. `submit(binding, outcome)` returns immediately with an asyncio task;
2. the bound verifier runs under a concurrency limit (`max_verify_concurrency`);
3. `reward(score, cost)` is computed and `selector.update` is applied under a lock, in arrival
   order, so programs may finish and be verified in a different order from dispatch;
4. the result is appended to `feedback.history` (dashboard: "Recent feedback") and the optional
   `on_reward(binding, score, reward)` callback fires.

A missing verifier name falls back to `outcome.success`; a verifier that raises or returns a
NaN/inf score is logged, counted in `feedback.failed`, and does not update the selector (a NaN
reward would corrupt a learner for good). `await router.feedback.drain()` waits for everything in
flight (tests, shutdown).
