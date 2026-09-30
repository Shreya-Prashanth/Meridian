import asyncio
import pytest

@pytest.fixture
def run():
    def runner(coro):
        return asyncio.run(coro)
    return runner
