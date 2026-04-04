"""
examples/langgraph_provenance_example.py
========================================

Demonstrates FlowceptLangGraphPlugin with a simple three-node LangGraph
research-assistant workflow:

    [retrieve] → [analyze] → [summarize]

Each node is a regular Python function decorated with @node.  An LLM is
called in both 'analyze' and 'summarize' nodes.  A mock search tool is used
in 'retrieve'.

FlowceptLangGraphPlugin captures the full provenance hierarchy:

  WorkflowObject  (one per graph.invoke call)
    └─ TaskObject  subtype=langgraph_node  activity_id="retrieve"
         └─ TaskObject  subtype=tool_call  activity_id="search_papers"
    └─ TaskObject  subtype=langgraph_node  activity_id="analyze"
         └─ TaskObject  subtype=llm_call   activity_id=<model>
    └─ TaskObject  subtype=langgraph_node  activity_id="summarize"
         └─ TaskObject  subtype=llm_call   activity_id=<model>

After the run the provenance buffer is written to a JSONL file and the
overhead timing table is printed.

Requirements
------------
    pip install langgraph langchain-core langchain-openai

Set your OpenAI API key:
    export OPENAI_API_KEY=sk-...

Run:
    python examples/langgraph_provenance_example.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Annotated, TypedDict

# ---------------------------------------------------------------------------
# FlowCept plugin
# ---------------------------------------------------------------------------
from academy_coscientist.plugins.flowcept_langgraph_plugin import FlowceptLangGraphPlugin

# ---------------------------------------------------------------------------
# LangGraph / LangChain imports
# ---------------------------------------------------------------------------
try:
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
except ImportError:
    print(
        "ERROR: LangGraph/LangChain not installed.\n"
        "  pip install langgraph langchain-core langchain-openai",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class ResearchState(TypedDict):
    topic: str
    papers: list[str]           # retrieved paper snippets
    analysis: str               # LLM analysis of papers
    summary: str                # final executive summary


# ---------------------------------------------------------------------------
# Mock search tool (no network required to run the example)
# ---------------------------------------------------------------------------

@tool
def search_papers(query: str) -> str:
    """Search for research papers matching the query. Returns a list of snippets."""
    # Simulated corpus — replace with a real retriever for production use.
    corpus = {
        "graph neural": [
            "GNN survey (Wu et al., 2020): comprehensive overview of graph neural networks.",
            "GraphSAGE (Hamilton et al., 2017): inductive representation learning on graphs.",
            "GAT (Velickovic et al., 2018): graph attention networks.",
        ],
        "transformer": [
            "Attention Is All You Need (Vaswani et al., 2017): introduced the transformer.",
            "BERT (Devlin et al., 2019): pre-training deep bidirectional transformers.",
        ],
        "provenance": [
            "W3C PROV-DM: data model for provenance information.",
            "FlowCept (Skluzacek et al., 2023): runtime provenance for scientific workflows.",
        ],
    }
    q = query.lower()
    for key, snippets in corpus.items():
        if key in q:
            return "\n".join(snippets)
    return "No papers found for this query."


# ---------------------------------------------------------------------------
# LLM (falls back to a stub when OPENAI_API_KEY is not set)
# ---------------------------------------------------------------------------

def _make_llm():
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    # Stub that returns a fixed response — useful for testing without a key.
    print(
        "[example] OPENAI_API_KEY not set — using stub LLM (no real inference).",
        flush=True,
    )
    return _StubLLM()


class _StubLLM:
    """Minimal stub that mimics ChatOpenAI.invoke() and fires the callback hooks."""

    def invoke(self, messages, config=None):
        from langchain_core.messages import AIMessage
        text = (
            "[STUB] This is a simulated LLM response. "
            "Set OPENAI_API_KEY to use a real model."
        )
        return AIMessage(content=text)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def retrieve_node(state: ResearchState) -> ResearchState:
    """Search for papers relevant to the topic using the search_papers tool."""
    topic = state["topic"]
    # Invoke the tool directly (with provenance captured via on_tool_start/end)
    result = search_papers.invoke({"query": topic})
    snippets = [line.strip() for line in result.split("\n") if line.strip()]
    return {"papers": snippets}


def analyze_node(state: ResearchState, llm) -> ResearchState:
    """Ask the LLM to analyze the retrieved papers and identify key themes."""
    papers_text = "\n".join(f"  - {p}" for p in state["papers"]) or "  (none)"
    messages = [
        SystemMessage(content="You are a research analyst. Be concise."),
        HumanMessage(
            content=(
                f"Topic: {state['topic']}\n\n"
                f"Retrieved papers:\n{papers_text}\n\n"
                "Identify the 2-3 most important themes and open questions in 3-4 sentences."
            )
        ),
    ]
    response = llm.invoke(messages)
    return {"analysis": response.content}


def summarize_node(state: ResearchState, llm) -> ResearchState:
    """Produce a final executive summary integrating papers and analysis."""
    messages = [
        SystemMessage(content="You are a scientific writer. Be concise and precise."),
        HumanMessage(
            content=(
                f"Topic: {state['topic']}\n\n"
                f"Analysis:\n{state['analysis']}\n\n"
                "Write a 2-paragraph executive summary suitable for a grant proposal."
            )
        ),
    ]
    response = llm.invoke(messages)
    return {"summary": response.content}


# ---------------------------------------------------------------------------
# Build the LangGraph graph
# ---------------------------------------------------------------------------

def build_graph(llm):
    builder = StateGraph(ResearchState)

    builder.add_node("retrieve",  retrieve_node)
    builder.add_node("analyze",   lambda s: analyze_node(s, llm))
    builder.add_node("summarize", lambda s: summarize_node(s, llm))

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve",  "analyze")
    builder.add_edge("analyze",   "summarize")
    builder.add_edge("summarize", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    topic = "graph neural networks for scientific discovery"
    dump_path = "langgraph_provenance.jsonl"
    # Start fresh each run so the file never accumulates records from old code versions
    if os.path.exists(dump_path):
        os.remove(dump_path)

    llm = _make_llm()
    graph = build_graph(llm)

    plugin = FlowceptLangGraphPlugin(
        config={
            "enabled":              True,
            "workflow_name":        "research-assistant-langgraph",
            "dump_path":            dump_path,
            "performance_tracking": True,
        }
    )
    plugin.start()

    print(f"\n[example] Running graph for topic: {topic!r}\n", flush=True)

    initial_state: ResearchState = {
        "topic":    topic,
        "papers":   [],
        "analysis": "",
        "summary":  "",
    }

    final_state = graph.invoke(
        initial_state,
        config={"callbacks": [plugin.callback_handler]},
    )

    plugin.stop()

    # ── Print results ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GRAPH OUTPUT")
    print("=" * 60)
    print(f"\nTopic   : {final_state['topic']}")
    print(f"\nPapers retrieved ({len(final_state['papers'])}):")
    for p in final_state["papers"]:
        print(f"  • {p}")
    print(f"\nAnalysis:\n{final_state['analysis']}")
    print(f"\nSummary:\n{final_state['summary']}")

    # ── Print provenance sample ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"PROVENANCE  ({dump_path})")
    print("=" * 60)
    if os.path.exists(dump_path):
        records = []
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
            st = r.get("subtype", r.get("type", "other"))
            by_subtype.setdefault(st, []).append(r)

        print(f"\nTotal provenance records: {len(records)}")
        for st, recs in sorted(by_subtype.items()):
            print(f"  {st:<25} {len(recs)} record(s)")

        print("\nSample records (one per subtype):")
        for st, recs in sorted(by_subtype.items()):
            r = recs[0]
            print(
                f"\n  [{st}] activity_id={r.get('activity_id','?')}  "
                f"status={r.get('status','?')}  "
                f"workflow_id={str(r.get('workflow_id','?'))[:12]}…"
            )
    else:
        print(f"  (no file written — FlowCept may not have been enabled)")


if __name__ == "__main__":
    main()
