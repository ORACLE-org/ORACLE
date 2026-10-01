# Configuration (`oracle.yaml`)

`${ENV_VAR}` is expanded anywhere in the file (an unset variable is left as written). Every key is
optional except `models`; an unknown key, a wrong type or a duplicate model name is a `ValueError` at load time.

```yaml
models:                      # ordered strongest -> weakest
  - name: strong             # router name; clients may put anything in "model"
    url: http://host:8001/v1 # OpenAI-compatible base URL (ends with /v1)
    served_name: Qwen/...    # what the backend expects in "model" (default: name)
    backend: gpu-a           # scheduler backend name (default: name; share one for co-located models)
    price_in: 0.3            # $ per 1M prompt tokens
    price_out: 0.9           # $ per 1M completion tokens
    capacity: auto           # auto | vllm | sglang | static
    capacity_tokens: null    # tokens: required for static; auto uses it when set, else detects vLLM/SGLang at url
    api_key: EMPTY

router:
  enabled: true              # false = scheduling only: the fixed selector (first model or force_model), nothing learns
  selector: linucb           # see docs/routing.md
  selector_kwargs: {}        # passed to the selector's __init__
  features: hashing          # hashing | embedding | task_type | hashing+task_type
  feature_dim: 64
  reward_alpha: 0.0          # Eq. 3 cost weight
  reward_cost_ref: 1.0       # cost that counts as 1
  max_verify_concurrency: 8

verification:
  embedder: hashing          # hashing | st:<model> | openai:<model> (+ embedder_url)
  embedder_url: null
  min_sim: -1.0              # below this similarity use `default`
  default: null
  metadata_key: null         # read the task type from program_metadata[<key>] instead of embeddings
  verifiers:
    - name: patch_tests
      type: command          # reported | command | http | llm_judge | python
      task_type: swe         # default: name
      exemplars: [...]       # example initial requests (adaptive verification needs >= 2 verifiers with exemplars)
      options: {...}         # constructor arguments of the verifier class

scheduler:
  enabled: true
  beta: 0.5                  # 0 = admission only
  w0: 60
  rho: 0.85
  peak_prior: 20000
  prior_weight: 5
  floor: 1.0
  per_program_overhead: 0
  duration_prior_s: 30
  max_programs: 0            # >0 = plain semaphore baseline
  poll_s: 2.0                # capacity polling
  max_wait_s: 0              # liveness valve; 0 = wait forever

server:
  host: 0.0.0.0
  port: 8300
  program_id_header: X-Program-ID
  done_key: program_done
  request_timeout_s: 900
  ui: true
  log_level: info
```

Command-line flags override the file: `--models`, `--selector`, `--no-scheduler`, `--router none`,
`--beta`, `--capacity-tokens`, `--host`, `--port`.
