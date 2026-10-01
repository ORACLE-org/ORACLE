# Architecture

## The program is the unit

An agentic task (fix an issue, book a trip, finish a terminal job) is one *program*: a sequence of
LLM requests that share a growing prefix, interleaved with tool calls. ORACLE routes, schedules and
learns at that level, not per request. Requests of one program always go to the same backend, so
its prefix cache stays on one GPU.

Every program gets one row in a lookup table when its first request arrives:

```
R = <ID, P, M, V>        program ID  ->  (model m, verifier v, backend)
```

`oracle.types.Binding` is that row. It is created once, reused by every later request of the
program, and removed after the program's verifier has run.

## Request flow (`oracle serve`, router + DISC)

```
client ── request(program_id) ──► proxy
                                    │ first request of the ID?
                                    ├─ no  ─► lookup table ─► forward to the bound backend
                                    └─ yes ─► Router.context()      embed the request, pick the verifier (task type)
                                              Router.propose()      model selector picks a model
                                              DISC.dispatch()       read every ledger's wait forecast; maybe divert
                                              DISC.admit()          wait until the backend has room; reserve the peak
                                              Binding stored        forward
                                  ◄── response + usage ─────────  observe_request(): cost, context length -> ledger
...
last request (program_done) or POST /programs/{id}/complete
                                    DISC.release()                  free the reservation now
                                    FeedbackLoop.submit()           verifier runs in the background
                                    reward = (1-α)·score - α·cost   selector.update()   (arrival order)
```

Routing only: skip the two DISC steps. Scheduling only: the "selector" is `fixed` (or your own
proposal) and nothing learns.

## Module map

```
oracle/
├── types.py             RoutingContext, Binding, ProgramOutcome
├── core.py              Oracle (router + DISC)
├── config.py            OracleConfig dataclasses, YAML loading, build()
├── routing/
│   ├── base.py          ModelSelector (select / update / estimate)
│   ├── registry.py      register_selector, make_selector, entry points
│   ├── bandit.py        linucb (default), lints, ucb, epsilon_greedy
│   ├── static.py        fixed, round_robin, random, callable
│   ├── acrouter.py      ACRouter adapter (memory + orchestrator LLM)
│   ├── routellm.py      RouteLLM adapter
│   ├── features.py      HashingFeatures, EmbeddingFeatures, TaskTypeFeatures
│   ├── reward.py        Reward (Eq. 3)
│   └── router.py        Router: lookup table + selectors + feedback loop
├── verification/
│   ├── base.py          Verifier; Reported / Callable / Command / HTTP / LLMJudge
│   ├── embeddings.py    HashingEmbedder, SentenceTransformerEmbedder, OpenAIEmbedder
│   ├── selector.py      PrototypeVerifierSelector (Eq. 2), RuleVerifierSelector, FixedVerifierSelector
│   └── feedback.py      FeedbackLoop (delayed feedback)
├── scheduling/
│   ├── ledger.py        ReservationLedger (per backend)
│   ├── admission.py     AdmissionGate (async FIFO)
│   ├── dispatch.py      DispatchPolicy (Eq. 4)
│   ├── capacity.py      StaticCapacity, VLLMCapacity, SGLangCapacity
│   └── disc.py          DISC facade
├── server/
│   ├── app.py           FastAPI routes
│   ├── proxy.py         forwarding, SSE usage extraction
│   └── ui/index.html    dashboard
├── integrations/        thunderagent.py, fastapi.py (DISCMiddleware), litellm.py
└── sim.py               synthetic workload
```

## Design rules

* **Two pieces, no cross-imports.** `scheduling` imports nothing from `routing` or
  `verification`; the router (`routing` + `verification`) imports nothing from `scheduling`.
  Each can be used on its own.
* **Policies are plugins.** Model selectors and verifiers are small base classes with a registry;
  nothing else in the code knows which one is in use.
* **Pure logic, thin I/O.** Ledger, gate and dispatch rule are plain Python objects with injectable
  clocks; HTTP and metrics parsing live in `server/` and `capacity.py`.
* **Clients change one field.** `program_id`, plus an optional `program_done`.
