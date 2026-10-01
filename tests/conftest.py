import asyncio

import pytest


@pytest.fixture
def run():
    def _run(coro):
        return asyncio.run(coro)

    return _run
