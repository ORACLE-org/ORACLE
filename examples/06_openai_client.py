"""Talk to a running `oracle serve` with the OpenAI SDK.  The only change from a
plain client is the program_id (and program_done on the last request).

    oracle serve --models strong=http://localhost:8001/v1 weak=http://localhost:8002/v1 --port 8300
    python examples/06_openai_client.py
"""
import uuid

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8300/v1", api_key="EMPTY")
program_id = f"task-{uuid.uuid4().hex[:8]}"
messages = [{"role": "user", "content": "Fix the failing test in src/parser.py; it crashes on empty input."}]

# first request binds the program to a model and a verifier
r = client.chat.completions.create(model="auto", messages=messages, extra_body={"program_id": program_id})
messages += [{"role": "assistant", "content": r.choices[0].message.content}, {"role": "user", "content": "Now run the tests and report."}]

# ... more requests reuse the same model (prefix cache stays on one backend) ...

# last request: tell ORACLE the program is done so the slot is released and the verifier runs.
# If your harness grades the task itself, send program_success; otherwise the bound verifier scores it.
r = client.chat.completions.create(model="auto", messages=messages, extra_body={"program_id": program_id, "program_done": True, "program_success": 1.0})
print(r.choices[0].message.content)

# Alternative to program_done: POST /programs/{program_id}/complete {"success": 1.0, "cost": 0.12, "payload": {...}}
