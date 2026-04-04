"""
examples/combined_four_framework_provenance.py
===============================================

One pipeline — four frameworks — all provenance in a single shared
FlowCept buffer → one JSONL file.  No real LLM / API key required.

Frameworks
----------
  1. Academy        ProducerAgent   produces a list of research hypotheses
  2. LangGraph      filter_graph    scores and selects the top hypothesis
  3. CrewAI         eval_crew       evaluates the selected hypothesis
  4. AutoGen        refine_team     refines the hypothesis in a 2-agent dialogue

Pipeline
--------
  ProducerAgent.produce(n=3)
       │  hypotheses=[h1, h2, h3]
       ▼
  LangGraph  filter_graph
    score_node   → scores each hypothesis
    select_node  → picks the highest-scored one
       │  top_hypothesis=h2
       ▼
  CrewAI  eval_crew
    Evaluator agent  → strengths, weaknesses, recommendation
       │  evaluation=<text>
       ▼
  AutoGen  refine_team
    scientist → critic → scientist  (3 messages)
       │  final_hypothesis=<revised text>
       ▼
  DONE

Provenance (single JSONL, single campaign_id)
---------------------------------------------
  WorkflowObject  master
    └─ WorkflowObject  ProducerAgent sub-wf
         └─ TaskObject  subtype=academy_lifecycle  agent_startup/shutdown
         └─ TaskObject  subtype=academy_action     produce(3)
    └─ WorkflowObject  LangGraph filter sub-wf
         └─ TaskObject  subtype=langgraph_graph    filter_graph
         └─ TaskObject  subtype=langgraph_node     score_node
         └─ TaskObject  subtype=langgraph_node     select_node
    └─ WorkflowObject  CrewAI eval sub-wf
         └─ TaskObject  subtype=crewai_crew        eval_crew
         └─ TaskObject  subtype=crewai_task        evaluate_hypothesis
         └─ TaskObject  subtype=crewai_agent       Evaluator
         └─ TaskObject  subtype=llm_call           gpt-4o-mini
    └─ WorkflowObject  AutoGen refine sub-wf
         └─ TaskObject  subtype=autogen_run        refine-team
         └─ TaskObject  subtype=autogen_message    scientist (×2)
         └─ TaskObject  subtype=autogen_message    critic    (×1)

Run
---
    python examples/combined_four_framework_provenance.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Sequence, TypedDict
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("FLOWCEPT_USE_DEFAULT", "1")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

# ---------------------------------------------------------------------------
# FlowCept plugins
# ---------------------------------------------------------------------------
from academy_coscientist.plugins.flowcept_plugin import FlowceptAcademyPlugin
from academy_coscientist.plugins.flowcept_langgraph_plugin import FlowceptLangGraphPlugin
from academy_coscientist.plugins.flowcept_crewai_plugin import FlowceptCrewAIPlugin
from academy_coscientist.plugins.flowcept_autogen_plugin import FlowceptAutoGenPlugin
import academy_coscientist.utils.utils_logging as _log_mod

# ---------------------------------------------------------------------------
# Academy
# ---------------------------------------------------------------------------
from academy.agent import Agent, action
from academy.exchange import LocalExchangeFactory
from academy.logging import init_logging
from academy.manager import Manager

# ---------------------------------------------------------------------------
# LangGraph
# ---------------------------------------------------------------------------
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    print("ERROR: pip install langgraph langchain-core", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# CrewAI
# ---------------------------------------------------------------------------
try:
    from crewai import Agent as CrewAgent, Task as CrewTask, Crew, LLM
    from crewai.tools import tool as crewai_tool
except ImportError:
    print("ERROR: pip install crewai", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# AutoGen
# ---------------------------------------------------------------------------
try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_core.models import (
        ChatCompletionClient, CreateResult, RequestUsage, LLMMessage,
    )
    from autogen_core import CancellationToken
    from pydantic import BaseModel
except ImportError:
    print("ERROR: pip install autogen-agentchat", file=sys.stderr)
    sys.exit(1)


# ===========================================================================
# 1. Academy agent
# ===========================================================================

_HYPOTHESES = [
    "H1: Graph-partitioned HNSW indices reduce HPC search latency by ≥20%.",
    "H2: Ontology-guided vector clustering achieves ≥95% recall at 64 nodes.",
    "H3: Hybrid BM25+dense retrieval outperforms pure dense on scientific corpora.",
]


class ProducerAgent(Agent):
    """Produces a list of research hypotheses."""

    @action
    async def produce(self, n: int) -> list[str]:
        return _HYPOTHESES[:n]


# ===========================================================================
# 2. LangGraph graph — score and select top hypothesis
# ===========================================================================

class FilterState(TypedDict):
    hypotheses:       list[str]
    scores:           list[float]
    top_hypothesis:   str
    _source_agent_id: str


def score_node(state: FilterState) -> FilterState:
    """Score each hypothesis (stub: length-based heuristic)."""
    scores = [len(h) / 80.0 for h in state["hypotheses"]]
    return {"scores": scores}


def select_node(state: FilterState) -> FilterState:
    """Select the highest-scored hypothesis."""
    best_idx  = state["scores"].index(max(state["scores"]))
    return {"top_hypothesis": state["hypotheses"][best_idx]}


def build_filter_graph():
    b = StateGraph(FilterState)
    b.add_node("score_node",  score_node)
    b.add_node("select_node", select_node)
    b.set_entry_point("score_node")
    b.add_edge("score_node",  "select_node")
    b.add_edge("select_node", END)
    return b.compile()


# ===========================================================================
# 3. CrewAI crew — evaluate the selected hypothesis
# ===========================================================================

_EVAL_RESPONSES = [
    (
        "Final Answer: The hypothesis is clear and falsifiable. "
        "Strengths: concrete performance targets, leverages existing Faiss. "
        "Weaknesses: cluster stability at scale needs validation. "
        "Recommendation: REVISE — add sensitivity analysis for θ parameter."
    ),
]
_eval_idx = 0

def _make_fake_completion(content: str):
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types import CompletionUsage
    return ChatCompletion(
        id="chatcmpl-stub",
        choices=[Choice(
            finish_reason="stop", index=0,
            message=ChatCompletionMessage(role="assistant", content=content),
            logprobs=None,
        )],
        created=1_700_000_000,
        model="gpt-4o-mini",
        object="chat.completion",
        usage=CompletionUsage(completion_tokens=50, prompt_tokens=100, total_tokens=150),
    )


def _stub_eval_create(*args, **kwargs):
    global _eval_idx
    content = _EVAL_RESPONSES[_eval_idx % len(_EVAL_RESPONSES)]
    _eval_idx += 1
    return _make_fake_completion(content)


def build_eval_crew(hypothesis: str) -> Crew:
    llm = LLM(model="openai/gpt-4o-mini", api_key="stub-key")
    evaluator = CrewAgent(
        role="Evaluator",
        goal="Evaluate the scientific hypothesis for novelty and feasibility.",
        backstory="You are an expert scientific reviewer.",
        llm=llm,
        verbose=False,
        max_iter=2,
    )
    eval_task = CrewTask(
        description=(
            f"Evaluate this hypothesis:\n\n{hypothesis}\n\n"
            "Identify strengths, weaknesses, and give a recommendation."
        ),
        expected_output="Strengths, weaknesses, recommendation.",
        agent=evaluator,
    )
    return Crew(agents=[evaluator], tasks=[eval_task], verbose=False)


# ===========================================================================
# 4. AutoGen team — refine the hypothesis
# ===========================================================================

_REFINE_RESPONSES = [
    (
        "scientist",
        "Revised hypothesis: Ontology-guided HNSW indexing with θ=0.6 and top-3 "
        "community clusters reduces search latency by ≥20% vs. vanilla Faiss HNSW "
        "on a 64-node cluster, maintaining ≥95% recall. Sensitivity analysis for "
        "cluster count (k=2..5) and θ (0.4..0.8) is required."
    ),
    (
        "critic",
        "The revision is solid. The sensitivity analysis requirement is appropriate. "
        "The baseline is clearly specified. APPROVE. TERMINATE"
    ),
]
_refine_idx = 0


class _StubRefineClient(ChatCompletionClient):
    @property
    def model_info(self):
        from autogen_core.models import ModelInfo
        return ModelInfo(
            vision=False, function_calling=False, json_output=False,
            family="stub", structured_output=False,
        )

    @property
    def capabilities(self):
        from autogen_core.models import ModelCapabilities
        return ModelCapabilities(vision=False, function_calling=False, json_output=False)

    async def create(
        self, messages: Sequence[LLMMessage], *, tools=(), tool_choice="auto",
        json_output=None, extra_create_args=None, cancellation_token=None,
    ) -> CreateResult:
        global _refine_idx
        _, content = _REFINE_RESPONSES[_refine_idx % len(_REFINE_RESPONSES)]
        _refine_idx += 1
        return CreateResult(
            content=content,
            usage=RequestUsage(prompt_tokens=40, completion_tokens=80),
            finish_reason="stop",
            cached=False,
        )

    async def create_stream(self, messages, **kwargs) -> AsyncGenerator:
        result = await self.create(messages)
        yield result

    def actual_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    def total_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    def count_tokens(self, messages, **kwargs) -> int:
        return 0

    def remaining_tokens(self, messages, **kwargs) -> int:
        return 4096

    async def close(self) -> None:
        pass


def build_refine_team() -> RoundRobinGroupChat:
    client = _StubRefineClient()
    scientist = AssistantAgent(
        name="scientist",
        model_client=client,
        system_message="Refine the hypothesis based on the evaluator's feedback. End with TERMINATE when done.",
    )
    critic = AssistantAgent(
        name="critic",
        model_client=client,
        system_message="Critically assess the revision. If satisfactory write APPROVE. TERMINATE",
    )
    return RoundRobinGroupChat(
        participants=[scientist, critic],
        termination_condition=MaxMessageTermination(max_messages=2),
    )


# ===========================================================================
# Driver
# ===========================================================================

async def _run(
    lg_plugin:     FlowceptLangGraphPlugin,
    crewai_plugin: FlowceptCrewAIPlugin,
    autogen_plugin: FlowceptAutoGenPlugin,
) -> dict:
    graph  = build_filter_graph()
    refine_team = build_refine_team()

    exchange = LocalExchangeFactory()
    executor = ThreadPoolExecutor(max_workers=2)

    async with await Manager.from_exchange_factory(
        factory=exchange,
        executors=executor,
    ) as manager:

        # ── Step 1: Academy ProducerAgent ────────────────────────────────────
        producer = await manager.launch(ProducerAgent)
        await producer.ping()

        hypotheses = await producer.produce(3)
        print(f"[1] ProducerAgent → {len(hypotheses)} hypotheses", flush=True)
        for h in hypotheses:
            print(f"     {h}", flush=True)

        # ── Step 2: LangGraph filter graph ───────────────────────────────────
        initial: FilterState = {
            "hypotheses":       hypotheses,
            "scores":           [],
            "top_hypothesis":   "",
            "_source_agent_id": str(producer.agent_id),
        }
        final_lg = graph.invoke(
            initial,
            config={"callbacks": [lg_plugin.callback_handler]},
        )
        top_hyp = final_lg["top_hypothesis"]
        print(f"\n[2] LangGraph → top hypothesis: {top_hyp}", flush=True)

        # ── Step 3: CrewAI evaluation crew ───────────────────────────────────
        eval_crew = build_eval_crew(top_hyp)
        with patch(
            "openai.resources.chat.completions.Completions.create",
            side_effect=_stub_eval_create,
        ):
            crew_result = eval_crew.kickoff()
        print(f"\n[3] CrewAI evaluation:\n     {str(crew_result)[:120]}", flush=True)

        # ── Step 4: AutoGen refinement team ──────────────────────────────────
        refine_task = (
            f"Hypothesis to refine:\n{top_hyp}\n\n"
            f"Evaluator feedback:\n{str(crew_result)[:300]}\n\n"
            "Please revise the hypothesis to address the feedback."
        )
        autogen_result = await autogen_plugin.run_team(
            refine_team,
            refine_task,
            team_name="refine-team",
            source_agent_id=str(producer.agent_id),
        )
        final_msg = autogen_result.messages[-1].content if autogen_result.messages else ""
        print(f"\n[4] AutoGen refinement:\n     {final_msg[:120]}", flush=True)

    return {
        "hypotheses":     hypotheses,
        "top_hypothesis": top_hyp,
        "evaluation":     str(crew_result),
        "final":          final_msg,
    }


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    init_logging("WARNING")
    dump_path = "combined_four_framework.jsonl"
    if os.path.exists(dump_path):
        os.remove(dump_path)

    # ── Start Academy plugin (owns the single FlowCept buffer) ───────────────
    academy_plugin = FlowceptAcademyPlugin(
        config={
            "enabled":              True,
            "workflow_name":        "four-framework-pipeline",
            "dump_path":            dump_path,
            "performance_tracking": True,
        },
        llm_hook_register=_log_mod.register_llm_hook,
        llm_hook_unregister=_log_mod.unregister_llm_hook,
    ).start()

    # ── All other plugins share the same buffer ───────────────────────────────
    lg_plugin      = FlowceptLangGraphPlugin.from_academy_plugin(academy_plugin)
    crewai_plugin  = FlowceptCrewAIPlugin.from_academy_plugin(academy_plugin)
    autogen_plugin = FlowceptAutoGenPlugin.from_academy_plugin(academy_plugin)

    try:
        summary = asyncio.run(_run(lg_plugin, crewai_plugin, autogen_plugin))
    finally:
        lg_plugin.stop()
        crewai_plugin.stop()
        autogen_plugin.stop()
        academy_plugin.stop()   # flushes the single shared buffer → dump_path

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
    print(f"\n{'Subtype':<30} {'Framework':<12} {'Count':>5}")
    print("-" * 50)
    framework_map = {
        "academy_lifecycle":  "Academy",
        "academy_action":     "Academy",
        "langgraph_graph":    "LangGraph",
        "langgraph_node":     "LangGraph",
        "crewai_crew":        "CrewAI",
        "crewai_task":        "CrewAI",
        "crewai_agent":       "CrewAI",
        "autogen_run":        "AutoGen",
        "autogen_message":    "AutoGen",
        "llm_call":           "multi",
        "tool_call":          "multi",
        "workflow":           "—",
    }
    for st, recs in sorted(by_subtype.items()):
        fw = framework_map.get(st, "—")
        print(f"  {st:<28} {fw:<12} {len(recs):>5}")

    # Global campaign_id and workflow_id checks
    campaign_ids = {r.get("campaign_id") for r in records if r.get("campaign_id")}
    task_wf_ids  = {
        r.get("workflow_id") for r in records
        if r.get("type") != "workflow" and r.get("workflow_id")
    }
    print(f"\nDistinct campaign_id          : {len(campaign_ids)}  (must be 1 — single run)")
    print(f"Distinct workflow_id (tasks)  : {len(task_wf_ids)}  (must be 1 — global id)")
    for cid in campaign_ids:
        print(f"  campaign_id = {cid}")

    # Cross-framework linkage: source_agent_id in LangGraph and AutoGen records
    print("\nCross-framework linkage (source_agent_id):")
    for r in records:
        sid = (r.get("custom_metadata") or {}).get("source_agent_id")
        if sid:
            print(f"  [{r.get('subtype','?')}] source_agent_id = {sid}")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
