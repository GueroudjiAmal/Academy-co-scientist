"""
examples/crewai_provenance_example.py
======================================

Demonstrates FlowceptCrewAIPlugin with a simple two-agent CrewAI research workflow:

    [researcher] → [writer]

The researcher gathers paper snippets (via a tool), the writer summarises
the findings.  A stub LLM is used so the example runs without an API key.

FlowceptCrewAIPlugin captures the full provenance hierarchy via CrewAI's
native event bus — no code changes to agents or tasks required:

  WorkflowObject  (one per crew kickoff)
    └─ TaskObject  subtype=crewai_crew    activity_id="research-crew"
    └─ TaskObject  subtype=crewai_task    activity_id="research_papers"
         └─ TaskObject  subtype=crewai_agent  activity_id="Researcher"
              └─ TaskObject  subtype=tool_call   activity_id="search_papers"
              └─ TaskObject  subtype=llm_call    activity_id="gpt-4o-mini"
    └─ TaskObject  subtype=crewai_task    activity_id="write_summary"
         └─ TaskObject  subtype=crewai_agent  activity_id="Writer"
              └─ TaskObject  subtype=llm_call    activity_id="gpt-4o-mini"

Requirements
------------
    pip install crewai

Run
---
    python examples/crewai_provenance_example.py
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("FLOWCEPT_USE_DEFAULT", "1")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

# ---------------------------------------------------------------------------
# FlowCept plugin
# ---------------------------------------------------------------------------
from academy_coscientist.plugins.flowcept_crewai_plugin import FlowceptCrewAIPlugin

# ---------------------------------------------------------------------------
# CrewAI imports
# ---------------------------------------------------------------------------
try:
    from crewai import Agent, Task, Crew, LLM
    from crewai.tools import tool as crewai_tool
except ImportError:
    print("ERROR: pip install crewai", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Stub LLM — patches openai client so no API key is required
# ---------------------------------------------------------------------------

def _make_fake_openai_completion(content: str):
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types import CompletionUsage
    return ChatCompletion(
        id="chatcmpl-stub",
        choices=[Choice(
            finish_reason="stop",
            index=0,
            message=ChatCompletionMessage(role="assistant", content=content),
            logprobs=None,
        )],
        created=1_700_000_000,
        model="gpt-4o-mini",
        object="chat.completion",
        usage=CompletionUsage(completion_tokens=40, prompt_tokens=80, total_tokens=120),
    )


# Rotate through canned responses so each LLM call returns something sensible.
_STUB_RESPONSES = [
    (
        "Final Answer: Graph neural networks (GNNs) have emerged as a powerful "
        "paradigm for learning on structured data, enabling applications from "
        "drug discovery to social-network analysis. Key papers: GNN Survey "
        "(Wu 2020), GraphSAGE (Hamilton 2017), GAT (Velickovic 2018)."
    ),
    (
        "Final Answer: GNNs process graph-structured inputs by propagating "
        "messages between neighbouring nodes and aggregating features. They "
        "excel at tasks where relational structure matters, such as molecular "
        "property prediction, citation-network classification, and knowledge-"
        "graph completion. Open challenges include scalability and explainability."
    ),
]
_response_idx = 0

def _stub_create(*args, **kwargs):
    global _response_idx
    content = _STUB_RESPONSES[_response_idx % len(_STUB_RESPONSES)]
    _response_idx += 1
    return _make_fake_openai_completion(content)


# ---------------------------------------------------------------------------
# Mock search tool
# ---------------------------------------------------------------------------

@crewai_tool("search_papers")
def search_papers(query: str) -> str:
    """Search for research papers on a given topic."""
    corpus = {
        "graph neural": (
            "• GNN Survey (Wu et al., 2020): overview of graph neural networks.\n"
            "• GraphSAGE (Hamilton et al., 2017): inductive representation learning.\n"
            "• GAT (Velickovic et al., 2018): graph attention networks."
        ),
        "transformer": (
            "• Attention Is All You Need (Vaswani et al., 2017).\n"
            "• BERT (Devlin et al., 2019): bidirectional transformers."
        ),
    }
    q = query.lower()
    for key, snippets in corpus.items():
        if key in q:
            return snippets
    return "No papers found for this query."


# ---------------------------------------------------------------------------
# Build crew
# ---------------------------------------------------------------------------

def build_crew() -> Crew:
    llm = LLM(model="openai/gpt-4o-mini", api_key="stub-key")

    researcher = Agent(
        role="Researcher",
        goal="Find and summarise relevant research papers.",
        backstory=(
            "You are an expert researcher who uses the search_papers tool to "
            "find relevant literature and synthesise key findings."
        ),
        tools=[search_papers],
        llm=llm,
        verbose=False,
        max_iter=2,
    )

    writer = Agent(
        role="Writer",
        goal="Write a clear executive summary from the researcher's findings.",
        backstory=(
            "You are a scientific writer who turns technical research findings "
            "into concise, grant-proposal-ready summaries."
        ),
        llm=llm,
        verbose=False,
        max_iter=2,
    )

    research_task = Task(
        description=(
            "Search for papers on 'graph neural networks for scientific discovery' "
            "using the search_papers tool and return the key findings."
        ),
        expected_output="A bullet-point list of key papers and their contributions.",
        agent=researcher,
        tools=[search_papers],
    )

    write_task = Task(
        description=(
            "Using the researcher's findings, write a 2-paragraph executive summary "
            "suitable for a grant proposal. Be concise and precise."
        ),
        expected_output="A 2-paragraph executive summary.",
        agent=writer,
        context=[research_task],
    )

    return Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    dump_path = "crewai_provenance.jsonl"
    if os.path.exists(dump_path):
        os.remove(dump_path)

    plugin = FlowceptCrewAIPlugin(
        config={
            "enabled":              True,
            "workflow_name":        "research-crew",
            "dump_path":            dump_path,
            "performance_tracking": True,
        }
    )
    plugin.start()

    crew = build_crew()

    print(f"\n[example] Running CrewAI crew …\n", flush=True)
    try:
        with patch(
            "openai.resources.chat.completions.Completions.create",
            side_effect=_stub_create,
        ):
            result = crew.kickoff()
    finally:
        plugin.stop()

    # ── Results ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CREW OUTPUT")
    print("=" * 60)
    print(f"\n{result}")

    # ── Provenance summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"PROVENANCE  ({dump_path})")
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

    campaign_ids = {r.get("campaign_id") for r in records if r.get("campaign_id")}
    # Only task records (not WorkflowObjects) should share the global workflow_id
    task_wf_ids = {
        r.get("workflow_id") for r in records
        if r.get("type") != "workflow" and r.get("workflow_id")
    }
    print(f"\nDistinct campaign_id          : {len(campaign_ids)}  (must be 1)")
    print(f"Distinct workflow_id (tasks)  : {len(task_wf_ids)}  (must be 1 — global id)")

    # Show sample task records with used+generated
    print("\nSample task records (one per subtype):")
    for st, recs in sorted(by_subtype.items()):
        r = recs[0]
        if r.get("type") != "workflow":
            print(
                f"\n  [{st}]  activity_id={r.get('activity_id','?')}  "
                f"status={r.get('status','?')}  "
                f"has_used={'used' in r}  has_generated={'generated' in r}"
            )


if __name__ == "__main__":
    main()
