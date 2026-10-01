# Server and HTTP API

```bash
oracle serve --config oracle.yaml
oracle serve --models strong=http://host:8001/v1 weak=http://host:8002/v1 [--selector linucb] [--no-scheduler] [--router none] [--beta 0.5] [--capacity-tokens N] [--port 8300]
```

## Request fields (OpenAI-compatible)

Send to `POST /v1/chat/completions` or `POST /v1/completions`. Everything below may be at the
top level of the body, inside `extra_body` (what the OpenAI SDK's `extra_body=` produces), or,
for the id, in a header.

| field | meaning |
|---|---|
| `program_id` (or header `X-Program-ID` / `X-Session-ID`) | the program this request belongs to. Missing: `"default"`. |
| `program_done: true` | this is the program's last request: release the slot after the response, run the verifier |
| `program_success: 0..1` | the harness's own score (used by `reported` verifiers) |
| `program_payload: {...}` | data for the verifier (container id, repo path, final state, ...) |
| `program_metadata: {...}` | anything for the selector / verifier selector (`task_type`, user, budget) |
| `force_model: name` | bypass the selector for this program (still scheduled and verified) |
| `model` | ignored for routing (use any string); ORACLE replaces it with the backend's served name |

These fields are stripped before forwarding. Streaming is passed through; ORACLE turns on
`stream_options.include_usage` to read token usage from the last chunk.

## Endpoints

| method, path | purpose |
|---|---|
| `POST /v1/chat/completions`, `POST /v1/completions` | proxy |
| `GET /v1/models` | configured models |
| `GET /programs` | active bindings |
| `GET /programs/{id}` | one binding (active or recently finished) |
| `POST /programs/{id}/complete` | body `{"success": 0..1, "cost": $, "payload": {...}}`; releases the slot and schedules verification |
| `POST /programs/complete` | same with `program_id` in the body |
| `POST /programs/{id}/release` | drop without feedback |
| `GET /state` | full JSON snapshot (router, selector, feedback, ledgers) |
| `GET /health` | liveness + counts |
| `GET /ui` | dashboard |

A client that disconnects while held at the admission gate gets HTTP 499 and loses its place in
the queue.

## Dashboard

`/ui` polls `/state` every two seconds: active programs and verifications, per-model selection
counts and realised accuracy, per-backend ledger bars (reserved vs budget, queue, learned peak
and run time, forecast wait, measured KV usage), the active program table (with diversions
marked), recent feedback and the raw selector state.

## Embedding the app

```python
from oracle.config import OracleConfig, build
from oracle.server import create_app

cfg = OracleConfig.load("oracle.yaml")
app = create_app(build(cfg), cfg.models, cfg.server)   # a FastAPI app; add your own routes
```
