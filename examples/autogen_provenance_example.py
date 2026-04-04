"""
examples/autogen_provenance_example.py
========================================

Demonstrates FlowceptAutoGenPlugin with a simple two-agent AutoGen team:

    [assistant] → [critic]  (RoundRobinGroupChat, 2 rounds)

The assistant proposes a hypothesis, the critic evaluates it, and the assistant
refines it.  A stub model client is used so the example runs without an API key.

FlowceptAutoGenPlugin captures provenance by consuming the team's run_stream():

  WorkflowObject  (one per team.run() call)
    └─ TaskObject  subtype=autogen_run      activity_id="research-team"
         └─ TaskObject  subtype=autogen_message  activity_id="assistant"
         └─ TaskObject  subtype=autogen_message  activity_id="critic"
         └─ TaskObject  subtype=autogen_message  activity_id="assistant"
         ...

Requirements
------------
    pip install autogen-agentchat

Run
---
    python examples/autogen_provenance_example.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import AsyncGenerator, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("FLOWCEPT_USE_DEFAULT", "1")

# ---------------------------------------------------------------------------
# FlowCept plugin
# ---------------------------------------------------------------------------
from academy_coscientist.plugins.flowcept_autogen_plugin import FlowceptAutoGenPlugin

# ---------------------------------------------------------------------------
# AutoGen imports
# ---------------------------------------------------------------------------
try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_core.models import (
        ChatCompletionClient, CreateResult, RequestUsage,
        LLMMessage, AssistantMessage,
    )
    from autogen_core import CancellationToken
    from pydantic import BaseModel
except ImportError:
    print("ERROR: pip install autogen-agentchat", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Stub model client — returns canned responses without any API call
# ---------------------------------------------------------------------------

_STUB_RESPONSES = [
    (
        "assistant",
        "Hypothesis: Integrating knowledge-graph semantic clusters into "
        "HNSW indices can reduce approximate-nearest-neighbour search latency "
        "by ≥20% while maintaining ≥95% recall on 64-node HPC clusters."
    ),
    (
        "critic",
        "The hypothesis is promising but needs tighter justification. "
        "Specifically, the 20% latency target should reference a baseline "
        "benchmark (e.g., vanilla Faiss HNSW) and the cluster-count "
        "hyperparameter θ needs a sensitivity analysis. APPROVE REVISION."
    ),
    (
        "assistant",
        "Revised hypothesis: Using top-3 knowledge-graph community clusters "
        "per embedding (θ=0.6 relation-strength threshold) with a vanilla "
        "Faiss HNSW baseline on a 64-node cluster, we target ≥20% latency "
        "reduction and ≥95% recall. TERMINATE"
    ),
]

_stub_idx = 0


class _StubModelClient(ChatCompletionClient):
    """Minimal stub that returns canned responses without a real API call."""

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
        self,
        messages: Sequence[LLMMessage],
        *,
        tools=(),
        tool_choice="auto",
        json_output=None,
        extra_create_args=None,
        cancellation_token=None,
    ) -> CreateResult:
        global _stub_idx
        _, content = _STUB_RESPONSES[_stub_idx % len(_STUB_RESPONSES)]
        _stub_idx += 1
        return CreateResult(
            content=content,
            usage=RequestUsage(prompt_tokens=30, completion_tokens=60),
            finish_reason="stop",
            cached=False,
        )

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        **kwargs,
    ) -> AsyncGenerator:
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


# ---------------------------------------------------------------------------
# Build the AutoGen team
# ---------------------------------------------------------------------------

def build_team(model_client) -> RoundRobinGroupChat:
    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message=(
            "You are a research scientist. Propose a specific, falsifiable "
            "hypothesis about HPC-scale vector search optimisation. "
            "When refined enough, end your message with TERMINATE."
        ),
    )
    critic = AssistantAgent(
        name="critic",
        model_client=model_client,
        system_message=(
            "You are a critical reviewer. Evaluate the hypothesis for "
            "novelty, rigor, and feasibility. Request specific improvements "
            "or write APPROVE REVISION if it looks solid."
        ),
    )
    # Stop after 3 messages total (assistant → critic → assistant)
    termination = MaxMessageTermination(max_messages=3)
    return RoundRobinGroupChat(
        participants=[assistant, critic],
        termination_condition=termination,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _run(plugin: FlowceptAutoGenPlugin, team) -> None:
    task = (
        "Propose and iteratively refine a hypothesis about using "
        "knowledge-graph community detection to accelerate HPC-scale "
        "vector similarity search."
    )
    print(f"\n[example] Task: {task!r}\n", flush=True)

    result = await plugin.run_team(team, task, team_name="research-team")

    print("\n" + "=" * 60)
    print("TEAM CONVERSATION")
    print("=" * 60)
    for msg in result.messages:
        source = getattr(msg, "source", "?")
        content = getattr(msg, "content", "")
        print(f"\n[{source}]\n{content}")
    print(f"\nStop reason: {result.stop_reason}")


def main() -> None:
    dump_path = "autogen_provenance.jsonl"
    if os.path.exists(dump_path):
        os.remove(dump_path)

    model_client = _StubModelClient()
    team = build_team(model_client)

    plugin = FlowceptAutoGenPlugin(
        config={
            "enabled":              True,
            "workflow_name":        "autogen-research",
            "dump_path":            dump_path,
            "performance_tracking": True,
        }
    )
    plugin.start()

    try:
        asyncio.run(_run(plugin, team))
    finally:
        plugin.stop()

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
    task_wf_ids  = {
        r.get("workflow_id") for r in records
        if r.get("type") != "workflow" and r.get("workflow_id")
    }
    print(f"\nDistinct campaign_id          : {len(campaign_ids)}  (must be 1)")
    print(f"Distinct workflow_id (tasks)  : {len(task_wf_ids)}  (must be 1 — global id)")

    # Show message records (parent_task_id links them to the run)
    msg_records = by_subtype.get("autogen_message", [])
    print(f"\nautogen_message records  : {len(msg_records)}")
    if msg_records:
        linked = [r for r in msg_records if r.get("parent_task_id")]
        print(f"  with parent_task_id    : {len(linked)}  (linked to enclosing run)")

    print("\nSample records (one per subtype):")
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
