"""Adapters that drop ORACLE's pieces into other systems.

* :mod:`.thunderagent` - run DISC inside ThunderAgent's scheduler;
* :mod:`.fastapi` - a middleware that puts DISC in front of any OpenAI-compatible ASGI server (SGLang, vLLM, your proxy);
* :mod:`.litellm` - use the router as a LiteLLM custom routing strategy.

Each module imports its host lazily, so importing this package never requires them.
"""
