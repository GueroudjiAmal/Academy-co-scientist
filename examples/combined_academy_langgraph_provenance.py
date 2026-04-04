"""
examples/combined_academy_langgraph_provenance.py
==================================================

One Academy agent + one LangGraph graph, ALL provenance in a single
shared FlowCept buffer → one JSONL file.  No AI / LLM calls.

Mirrors the structure of provenance_smoke_test.py.

Agents
------
ProducerAgent  (Academy)   — stores a list of ints; action: produce(n)
transform_graph (LangGraph) — two nodes: double_node, sum_node

Pipeline
--------
  ProducerAgent.produce(5)  →  [1, 2, 3, 4, 5]
        │  + agent_id for provenance linkage
        ▼
  LangGraph graph
    double_node  →  [2, 4, 6, 8, 10]
    sum_node     →  total = 30

Provenance (single JSONL, single campaign_id)
---------------------------------------------
  WorkflowObject  master
    └─ WorkflowObject  ProducerAgent sub-wf
         └─ TaskObject  subtype=academy_action    activity_id=produce
         └─ TaskObject  subtype=academy_lifecycle  activity_id=agent_startup/shutdown
    └─ WorkflowObject  LangGraph graph sub-wf
         └─ TaskObject  subtype=langgraph_graph    source_agent_id=<ProducerAgent id>
         └─ TaskObject  subtype=langgraph_node     activity_id=double_node
         └─ TaskObject  subtype=langgraph_node     activity_id=sum_node

Run
---
    python examples/combined_academy_langgraph_provenance.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("FLOWCEPT_USE_DEFAULT", "1")

# ---------------------------------------------------------------------------
# Academy
# ---------------------------------------------------------------------------
from academy.agent import Agent, action
from academy.exchange import LocalExchangeFactory
from academy.logging import init_logging
from academy.manager import Manager

# ---------------------------------------------------------------------------
# FlowCept plugins
# ---------------------------------------------------------------------------
from academy_coscientist.plugins.flowcept_plugin import FlowceptAcademyPlugin
from academy_coscientist.plugins.flowcept_langgraph_plugin import FlowceptLangGraphPlugin
import academy_coscientist.utils.utils_logging as _log_mod

# ---------------------------------------------------------------------------
# LangGraph
# ---------------------------------------------------------------------------
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    print("ERROR: pip install langgraph langchain-core", file=sys.stderr)
    sys.exit(1)


# ===========================================================================
# Academy agent
# ===========================================================================

class ProducerAgent(Agent):
    """Produces a list of consecutive integers starting from 1."""

    @action
    async def produce(self, n: int) -> list[int]:
        return list(range(1, n + 1))


# ===========================================================================
# LangGraph graph  (no LLM)
# ===========================================================================

class TransformState(TypedDict):
    numbers:          list[int]
    doubled:          list[int]
    total:            int
    _source_agent_id: str   # Academy agent that produced the input data


def double_node(state: TransformState) -> TransformState:
    return {"doubled": [x * 2 for x in state["numbers"]]}


def sum_node(state: TransformState) -> TransformState:
    return {"total": sum(state["doubled"])}


def build_graph():
    b = StateGraph(TransformState)
    b.add_node("double_node", double_node)
    b.add_node("sum_node",    sum_node)
    b.set_entry_point("double_node")
    b.add_edge("double_node", "sum_node")
    b.add_edge("sum_node", END)
    return b.compile()


# ===========================================================================
# Driver
# ===========================================================================

async def _run(lg_plugin: FlowceptLangGraphPlugin) -> dict:
    graph = build_graph()
    exchange = LocalExchangeFactory()
    executor = ThreadPoolExecutor(max_workers=2)

    async with await Manager.from_exchange_factory(
        factory=exchange,
        executors=executor,
    ) as manager:
        # ── Academy agent ────────────────────────────────────────────────────
        producer = await manager.launch(ProducerAgent)
        await producer.ping()

        numbers = await producer.produce(5)
        print(f"[driver] ProducerAgent.produce(5) → {numbers}", flush=True)

        # ── LangGraph graph ──────────────────────────────────────────────────
        # Pass the Academy agent's canonical agent_id so the LangGraph plugin
        # records it in every node's custom_metadata, linking the two agents.
        # producer.agent_id is the AgentId object whose str() matches the
        # agent_id field in Academy provenance records.
        initial: TransformState = {
            "numbers":          numbers,
            "doubled":          [],
            "total":            0,
            "_source_agent_id": str(producer.agent_id),
        }
        final = graph.invoke(
            initial,
            config={"callbacks": [lg_plugin.callback_handler]},
        )
        print(
            f"[driver] LangGraph: doubled={final['doubled']}  total={final['total']}",
            flush=True,
        )
        return final


def main() -> None:
    init_logging("WARNING")
    dump_path = "combined_provenance.jsonl"
    if os.path.exists(dump_path):
        os.remove(dump_path)

    # ── Start Academy plugin (owns the single FlowCept buffer) ───────────────
    academy_plugin = FlowceptAcademyPlugin(
        config={
            "enabled":              True,
            "workflow_name":        "academy-langgraph-combined",
            "dump_path":            dump_path,
            "performance_tracking": True,
        },
        llm_hook_register=_log_mod.register_llm_hook,
        llm_hook_unregister=_log_mod.unregister_llm_hook,
    ).start()

    # ── LangGraph plugin shares the SAME buffer (no second Flowcept instance) ─
    lg_plugin = FlowceptLangGraphPlugin.from_academy_plugin(academy_plugin)

    try:
        final = asyncio.run(_run(lg_plugin))
    finally:
        lg_plugin.stop()       # no-op on buffer; prints LangGraph perf stats
        academy_plugin.stop()  # flushes the single shared buffer → dump_path

    # ── Provenance summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"PROVENANCE  →  {dump_path}")
    print("=" * 60)

    if not os.path.exists(dump_path):
        print("  (file not written)")
        return

    records: list[dict] = []
    with open(dump_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    by_subtype: dict[str, list] = {}
    for r in records:
        st = r.get("subtype") or r.get("type") or "other"
        by_subtype.setdefault(st, []).append(r)

    print(f"\nTotal records : {len(records)}")
    print(f"\n{'Subtype':<30} {'Count':>5}")
    print("-" * 38)
    for st, recs in sorted(by_subtype.items()):
        print(f"  {st:<28} {len(recs):>5}")

    # Campaign ID must be identical across all records (shared buffer proof)
    campaign_ids = {r.get("campaign_id") for r in records if r.get("campaign_id")}
    print(f"\nDistinct campaign_id : {len(campaign_ids)}  (must be 1 — shared buffer)")
    for cid in campaign_ids:
        print(f"  {cid}")

    # Show source_agent_id linkage in LangGraph records
    linked = [
        r for r in records
        if r.get("custom_metadata", {}).get("source_agent_id")
    ]
    print(f"\nLangGraph records with source_agent_id : {len(linked)}")
    if linked:
        aid = linked[0]["custom_metadata"]["source_agent_id"]
        print(f"  source_agent_id = {aid}")
        # Confirm that Academy produced a record for this agent
        academy_records = [
            r for r in records
            if r.get("agent_id") == aid
        ]
        print(f"  Academy records for this agent_id    : {len(academy_records)}")


if __name__ == "__main__":
    main()
