"""
Provenance smoke test — two Academy agents, FlowCept plugin, perf timers.

Agents
------
CounterAgent   : holds an integer counter.  Actions: increment(n), get_value().
SummaryAgent   : calls CounterAgent N times and returns the final sum.
               Loop: runs one summary cycle then exits.

What this checks
----------------
* FlowceptAcademyPlugin starts / stops cleanly
* Academy Runtime patches fire for @action and @loop calls
* _ProvenanceStats accumulates action_emit / loop_emit / lifecycle_emit /
  intercept_task categories
* Summary table is printed to stdout
* Per-run CSV is written to examples/provenance_perf_<wf_id>.csv

Run
---
    cd <repo-root>
    python examples/provenance_smoke_test.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor

# Make sure the repo root is on the path when run directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("FLOWCEPT_USE_DEFAULT", "1")

from academy.agent import Agent, action, loop
from academy.exchange import LocalExchangeFactory
from academy.logging import init_logging
from academy.manager import Manager

from academy_coscientist.plugins.flowcept_plugin import FlowceptAcademyPlugin
import academy_coscientist.utils.utils_logging as _log_mod


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------


class CounterAgent(Agent):
    """Simple integer counter — two @action methods."""

    def __init__(self) -> None:
        super().__init__()
        self._value: int = 0

    @action
    async def increment(self, n: int = 1) -> int:
        """Add *n* to the counter and return the new value."""
        self._value += n
        return self._value

    @action
    async def get_value(self) -> int:
        """Return the current counter value."""
        return self._value


class SummaryAgent(Agent):
    """
    Calls CounterAgent.increment() several times then reads the total.

    The @loop runs exactly one cycle (increments 5 times by 1, 2, 3, 4, 5)
    and sets self._done = True so the driver can detect completion.
    """

    def __init__(self) -> None:
        super().__init__()
        self._counter = None
        self._result: int = 0
        self._done: bool = False

    @action
    async def set_counter(self, counter) -> None:
        """Wire up the CounterAgent handle."""
        self._counter = counter

    @action
    async def get_result(self) -> int:
        return self._result

    @action
    async def is_done(self) -> bool:
        return self._done

    @loop
    async def summary_loop(self, shutdown: asyncio.Event) -> None:
        """Wait for the counter handle, then increment 5 times and stop."""
        # The loop starts immediately after launch; wait until set_counter() fires.
        while self._counter is None and not shutdown.is_set():
            await asyncio.sleep(0.05)
        if shutdown.is_set():
            return

        print("[SummaryAgent] Starting summary cycle …", flush=True)
        for step in range(1, 6):
            value = await self._counter.increment(step)
            print(f"[SummaryAgent]   increment({step}) → counter={value}", flush=True)

        self._result = await self._counter.get_value()
        print(f"[SummaryAgent] Final counter value: {self._result}", flush=True)
        self._done = True


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def _run() -> None:
    exchange = LocalExchangeFactory()
    executor = ThreadPoolExecutor(max_workers=4)

    async with await Manager.from_exchange_factory(
        factory=exchange,
        executors=executor,
    ) as manager:
        # Launch agents
        counter = await manager.launch(CounterAgent)
        await counter.ping()

        summary = await manager.launch(SummaryAgent)
        await summary.ping()

        # Wire up
        await summary.set_counter(counter)

        # Wait for the summary loop to finish (polls every 0.5 s, up to 30 s)
        for _ in range(60):
            await asyncio.sleep(0.5)
            if await summary.is_done():
                break

        result = await summary.get_result()
        print(f"\n[driver] Counter reached {result}  (expected 15 = 1+2+3+4+5)\n",
              flush=True)
        assert result == 15, f"Expected 15 but got {result}"
        print("[driver] Assertion passed.", flush=True)


def main() -> None:
    init_logging("INFO")

    # Wire LLM hooks so the plugin can capture them (no actual LLM calls here,
    # but the hook plumbing is exercised).
    plugin = FlowceptAcademyPlugin(
        config={
            "enabled":       True,
            "workflow_name": "provenance-smoke-test",
            "perf_csv":      "examples/provenance_perf_smoke.csv",
        },
        llm_hook_register=_log_mod.register_llm_hook,
        llm_hook_unregister=_log_mod.unregister_llm_hook,
    )
    plugin.start()
    try:
        asyncio.run(_run())
    finally:
        plugin.stop()


if __name__ == "__main__":
    main()
