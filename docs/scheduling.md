# Scheduling: DISC

DISC decides *when* a program may start on a backend and, optionally, *which* backend, once per
program. It is independent of the router and of the serving engine: a ledger per backend, an
async gate per ledger, and a dispatch rule. Drop it into an existing scheduler (see
[integrations](integrations.md)) or your own loop.

## Reservation ledger

`ReservationLedger(capacity_tokens, rho=0.85, peak_prior=20000, prior_weight=5, floor=1.0,
per_program_overhead=0, duration_prior_s=30, max_programs=0)`

* `reserve(pid, tokens)`: the program is admitted; it is charged `max(current context, floor·ĉ) + overhead`.
* `update(pid, tokens)`: current context length after each request.
* `release(pid, peak_tokens)`: frees the reservation and updates two running estimates:
  `ĉ` (expected peak context, from finished programs' peaks) and `d̂` (expected run time).
* `fits()`: one more program fits iff `reserved + ĉ + overhead <= rho · capacity` (an empty backend
  always admits one).
* `wait_forecast(queued)`: 0 if the program would start now, otherwise `(queued - free_slots + 1) · d̂ / admitted`.
* `observe_usage(used_fraction, capacity_tokens)`: feed a measurement from a capacity source.

Both estimates start from the prior (`peak_prior`, `duration_prior_s`) weighted by `prior_weight`
pseudo-counts, so a cold ledger is conservative and converges after a handful of programs.
`max_programs` turns the ledger into a plain semaphore (useful as a baseline).

## Admission gate

`AdmissionGate(ledger)` is a strict FIFO queue: `await admit(pid, tokens=0, disconnected=None)`
returns the seconds waited. `release(pid, peak)` frees and pumps the queue. A cancelled or
disconnected waiter leaves the queue (and gives back a reservation granted in the same instant).
`max_wait_s` bounds the hold as a liveness valve (0 = wait forever).

## Dispatch rule

`DispatchPolicy(beta=0.5, w0=60)` (`DISC(policy=None)` disables diversion). For the proposed model `p` and each alternative `m` with a
shorter forecast wait, divert to the `m` with the largest positive margin

```
beta · (w_p - w_m) / w0  -  (1 - beta) · (â_p - â_m)
```

where `â` is the model selector's `estimate()`. No estimate for a pair means no diversion for
that pair; `beta = 0` disables diversion (admission only); `beta = 1` is least forecast wait.
`w0` is the wait at which the wait term equals `beta`: with `w0 = 60` and `beta = 0.5`, a one-minute
longer wait is worth a 50-point accuracy gap... so pick `w0` from the waits you actually see.

## `DISC`

```python
disc = DISC({"gpu-a": 120_000, "gpu-b": VLLMCapacity("http://localhost:8002/v1")},
            policy=DispatchPolicy(beta=0.5, w0=60), ledger_kwargs={"rho": 0.85, "peak_prior": 30_000})
await disc.start()                                   # polls capacity sources (if any)
backend, detail = disc.dispatch(proposed, estimates)  # estimates: {backend: â} or None
wait = await disc.admit(pid, backend, tokens=0)       # blocks
disc.update(pid, context_tokens)                      # per request
disc.release(pid, peak_tokens)                        # at the end
disc.snapshot()                                       # dashboard JSON
```

`dispatch` counts the program as *pending* on its backend until `admit` is called, so programs
bound but not yet at the gate are included in the next forecast. When several models share a
backend (`backend:` in the model config), the dispatch rule sees the best estimate among them and a
diverted program runs the best-estimated model of its new backend.

## Capacity sources

| class | reads |
|---|---|
| `StaticCapacity(tokens)` | a number you give |
| `AutoCapacity(url)` | tries vLLM's `/metrics` first, then SGLang's `/get_server_info`, and remembers which one answered (`capacity: auto`) |
| `VLLMCapacity(url)` | `/metrics`: `vllm:cache_config_info` (block_size × num_gpu_blocks) and `vllm:gpu_cache_usage_perc` |
| `SGLangCapacity(url)` | `/get_server_info` (`max_total_num_tokens`) and `/metrics` (`sglang:token_usage`) |

Any object with `async fetch() -> (capacity_tokens, used_fraction)` works.

## Tuning notes

* `rho`: headroom for decode growth and for programs the ledger cannot see (other clients).
  0.85 is a safe start; raise it if the measured usage on the dashboard stays low.
* `peak_prior`: set it to a typical peak context of your agent (for SWE-style agents 30-60k, for
  short tool-use tasks 10-20k). It only matters until a few programs have finished.
* `per_program_overhead`: engines with hybrid attention keep a fixed state per running request;
  if the measured KV usage exceeds the ledger's reservation at steady state, put the difference
  per program here.
* `beta`/`w0`: start with `beta = 0` (admission only) and look at the forecast waits on the
  dashboard; then set `w0` to the wait you are willing to trade for a modest accuracy drop.
