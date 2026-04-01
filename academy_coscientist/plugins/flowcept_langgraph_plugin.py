# academy_coscientist/plugins/flowcept_langgraph_plugin.py
"""
FlowCept provenance plugin for LangGraph workflows.

Mirrors the design of FlowceptAcademyPlugin but targets LangGraph/LangChain
instead of Academy.  Provenance is captured through LangChain's standard
callback interface, which LangGraph respects for all graph, node, LLM, and
tool executions.

Provenance hierarchy produced:
  WorkflowObject  (one per graph.invoke / graph.ainvoke call)
    └─ TaskObject  subtype=langgraph_node      activity_id=<node_name>
         └─ TaskObject  subtype=llm_call       activity_id=<model_name>   (parent_task_id)
         └─ TaskObject  subtype=tool_call      activity_id=<tool_name>    (parent_task_id)

Key FlowCept fields:
  task_id          — uuid per event (generated here, not LangChain's run_id)
  workflow_id      — per graph-run sub-workflow id
  campaign_id      — from Flowcept.campaign_id
  parent_task_id   — links LLM/tool tasks to their enclosing node task
  group_id         — all tasks within one graph run share a group_id
  activity_id      — node name | model name | tool name
  subtype          — langgraph_node | llm_call | tool_call | langgraph_graph
  used / generated — inputs and outputs
  status           — FINISHED | ERROR (proper Status enum values)
  telemetry_at_start/end — CPU/memory snapshots via TelemetryCapture

Usage (three lines to wire up):

    from academy_coscientist.plugins.flowcept_langgraph_plugin import FlowceptLangGraphPlugin

    plugin = FlowceptLangGraphPlugin(config={"workflow_name": "my-graph"})
    plugin.start()
    result = graph.invoke(state, config={"callbacks": [plugin.callback_handler]})
    plugin.stop()

Or as a context manager:

    with FlowceptLangGraphPlugin(config={"workflow_name": "my-graph"}) as plugin:
        result = graph.invoke(state, config={"callbacks": [plugin.callback_handler]})
"""
from __future__ import annotations

import os
os.environ.setdefault("FLOWCEPT_USE_DEFAULT", "1")

import time
import uuid
import logging
from typing import Any, Union
from uuid import UUID

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provenance overhead timer (identical to flowcept_plugin.py)
# ---------------------------------------------------------------------------

import threading

class _ProvenanceStats:
    """Lightweight thread-safe accumulator for provenance capture timings."""

    __slots__ = ("_lock", "_counts", "_totals", "_mins", "_maxs", "_raw")

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._counts: dict[str, int] = {}
        self._totals: dict[str, float] = {}
        self._mins:   dict[str, float] = {}
        self._maxs:   dict[str, float] = {}
        self._raw: list[tuple[str, str, float]] = []

    def record(self, category: str, elapsed: float) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._lock:
            if category not in self._counts:
                self._counts[category] = 0
                self._totals[category] = 0.0
                self._mins[category]   = float("inf")
                self._maxs[category]   = 0.0
            self._counts[category] += 1
            self._totals[category] += elapsed
            if elapsed < self._mins[category]:
                self._mins[category] = elapsed
            if elapsed > self._maxs[category]:
                self._maxs[category] = elapsed
            self._raw.append((ts, category, elapsed))

    def summary(self) -> str:
        col = 22
        header = (
            f"{'Category':<{col}} {'N':>7} {'Total(ms)':>11} "
            f"{'Mean(µs)':>9} {'Min(µs)':>8} {'Max(µs)':>8}"
        )
        sep = "-" * len(header)
        rows = [header, sep]
        with self._lock:
            for cat in sorted(self._counts):
                n     = self._counts[cat]
                total = self._totals[cat]
                mean  = (total / n) if n else 0.0
                mn    = self._mins.get(cat, 0.0)
                mx    = self._maxs.get(cat, 0.0)
                rows.append(
                    f"{cat:<{col}} {n:>7} {total*1e3:>11.3f} "
                    f"{mean*1e6:>9.1f} {mn*1e6:>8.1f} {mx*1e6:>8.1f}"
                )
        return "\n".join(rows)

    def to_csv(self, path: str, workflow_id: str | None = None) -> None:
        import csv
        write_header = not os.path.exists(path)
        with self._lock:
            raw_snapshot = list(self._raw)
        wf = workflow_id or ""
        rows = [
            {
                "timestamp_utc": ts,
                "workflow_id":   wf,
                "category":      cat,
                "elapsed_us":    round(elapsed * 1e6, 3),
            }
            for ts, cat, elapsed in raw_snapshot
        ]
        fieldnames = ["timestamp_utc", "workflow_id", "category", "elapsed_us"]
        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Interceptor wrapper (same pattern as AcademyInterceptor)
# ---------------------------------------------------------------------------

class LangGraphInterceptor:
    """
    Wraps FlowCept's InstrumentationInterceptor for LangGraph provenance.
    """

    def __init__(self) -> None:
        self._interceptor = None
        self._flowcept = None
        self._workflow_id: str | None = None
        self._campaign_id: str | None = None

    def start(self, workflow_name: str) -> None:
        from flowcept import Flowcept
        from flowcept.flowceptor.adapters.instrumentation_interceptor import (
            InstrumentationInterceptor,
        )
        self._flowcept = Flowcept(
            workflow_name=workflow_name,
            start_persistence=False,
            check_safe_stops=False,
            save_workflow=True,
        )
        self._flowcept.start()
        self._interceptor = InstrumentationInterceptor.get_instance()
        self._workflow_id = self._flowcept.current_workflow_id
        self._campaign_id = self._flowcept.campaign_id

    def stop(self, dump_path: str | None = None) -> None:
        if self._flowcept is None:
            return
        try:
            if dump_path:
                self._flowcept.dump_buffer(dump_path)
            self._flowcept.stop()
        except Exception as e:
            _log.warning("FlowCept stop error: %r", e)

    def flush(self, dump_path: str) -> None:
        if self._flowcept is None:
            return
        try:
            self._flowcept.dump_buffer(dump_path)
        except Exception as e:
            _log.warning("FlowCept flush error: %r", e)

    @property
    def telemetry_capture(self):
        return self._interceptor.telemetry_capture if self._interceptor else None

    def send_graph_workflow(self, graph_name: str, group_id: str) -> str:
        """Emit a WorkflowObject for one graph invocation; return its workflow_id."""
        if self._interceptor is None:
            return str(uuid.uuid4())
        from flowcept.commons.flowcept_dataclasses.workflow_object import WorkflowObject
        wf = WorkflowObject()
        wf.workflow_id = str(uuid.uuid4())
        wf.name = graph_name
        wf.campaign_id = self._campaign_id
        wf.parent_workflow_id = self._workflow_id
        wf.custom_metadata = {"group_id": group_id, "graph_name": graph_name}
        self._interceptor.send_workflow_message(wf)
        return wf.workflow_id

    def intercept_task(self, task_dict: dict) -> None:
        if self._interceptor is None:
            return
        from flowcept.commons.flowcept_dataclasses.task_object import TaskObject
        from flowcept.commons.vocabulary import Status

        task_dict.setdefault("task_id", str(uuid.uuid4()))
        task_dict.setdefault("workflow_id", self._workflow_id)
        task_dict.setdefault("campaign_id", self._campaign_id)

        raw = task_dict.get("status", "FINISHED")
        if isinstance(raw, str):
            try:
                task_dict["status"] = Status[raw].value
            except KeyError:
                task_dict["status"] = Status.FINISHED.value

        TaskObject.enrich_task_dict(task_dict)
        self._interceptor.intercept(task_dict)


# ---------------------------------------------------------------------------
# LangGraph callback handler
# ---------------------------------------------------------------------------

class FlowceptLangGraphCallback:
    """
    LangChain BaseCallbackHandler that records provenance to FlowCept.

    Pass an instance to ``graph.invoke`` / ``graph.ainvoke`` via the
    ``config={"callbacks": [...]}`` argument.

    This class deliberately avoids subclassing ``BaseCallbackHandler`` at
    module import time so LangGraph/LangChain are optional dependencies.
    The actual class is built lazily in ``_build_handler_class()`` and
    cached in ``_HANDLER_CLASS``.
    """

    def __init__(self, interceptor: LangGraphInterceptor, stats: _ProvenanceStats | None) -> None:
        self._interceptor = interceptor
        self._stats = stats
        # Maps LangChain run_id (UUID) → FlowCept task_id (str)
        self._run_to_task: dict[UUID, str] = {}
        # Maps LangChain run_id → start timestamp
        self._run_start: dict[UUID, float] = {}
        # Tracks telemetry snapshots at node start
        self._run_tel_start: dict[UUID, Any] = {}
        # Maps graph-level run_id → group_id (shared by all tasks in one graph invocation)
        self._graph_runs: dict[UUID, str] = {}
        # Maps any child run_id → enclosing graph-level run_id (for group_id lookup)
        self._run_to_graph_run: dict[UUID, UUID] = {}
        # Maps graph-level run_id → Academy agent ID that produced the input data.
        # Set when the LangGraph state contains "_source_agent_id".
        self._graph_source_agent: dict[UUID, str] = {}
        # Buffers the start-time task skeleton (used, custom_metadata, parent_task_id …)
        # so on_chain_end / on_llm_end can emit ONE complete record with both
        # used (inputs) and generated (outputs) — same pattern as the Academy plugin.
        self._run_start_task: dict[UUID, dict] = {}

    # --- helpers ---

    def _is_graph_run(
        self,
        serialized: dict | None,
        tags: list[str] | None,
        parent_run_id=None,
    ) -> bool:
        """True when the chain event represents a top-level LangGraph graph invocation.

        Primary signal: parent_run_id is None (no enclosing run → this IS the graph).
        Secondary signal: serialized id / tags for belt-and-suspenders detection.
        """
        if parent_run_id is None:
            return True
        id_parts = (serialized or {}).get("id", [])
        graph_classes = {
            "CompiledStateGraph", "CompiledGraph", "Pregel",
            "StateGraph", "MessageGraph",
        }
        return bool(
            graph_classes.intersection(id_parts)
            or any(t.startswith("__pregel_") for t in (tags or []))
        )

    def _node_name(self, serialized: dict | None, name: str | None) -> str:
        """Extract a human-readable node name."""
        if name:
            return name
        id_parts = (serialized or {}).get("id", [])
        return id_parts[-1] if id_parts else "unknown_node"

    def _model_name(self, serialized: dict | None) -> str:
        id_parts = (serialized or {}).get("id", [])
        kwargs = (serialized or {}).get("kwargs", {})
        return (
            kwargs.get("model_name")
            or kwargs.get("model")
            or (id_parts[-1] if id_parts else "unknown_model")
        )

    def _tel(self) -> Any:
        tc = self._interceptor.telemetry_capture
        return tc.capture() if tc else None

    def _record(self, category: str, elapsed: float) -> None:
        if self._stats is not None:
            self._stats.record(category, elapsed)

    def _task_id_for(self, run_id: UUID) -> str:
        if run_id not in self._run_to_task:
            self._run_to_task[run_id] = str(uuid.uuid4())
        return self._run_to_task[run_id]

    def _group_id_for(self, run_id: UUID) -> str | None:
        """Return the group_id for the graph invocation enclosing this run."""
        if run_id in self._graph_runs:
            return self._graph_runs[run_id]
        graph_run_id = self._run_to_graph_run.get(run_id)
        if graph_run_id is not None:
            return self._graph_runs.get(graph_run_id)
        return None

    # --- chain events (graph & node) ---

    def on_chain_start(
        self,
        serialized: dict | None,
        inputs: dict | None,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        t0 = time.perf_counter()
        self._run_start[run_id] = time.time()
        self._run_tel_start[run_id] = self._tel()
        task_id = self._task_id_for(run_id)
        serialized = serialized or {}
        inputs = inputs or {}

        if self._is_graph_run(serialized, tags, parent_run_id):
            # Top-level graph invocation → emit a sub-WorkflowObject for
            # hierarchy tracking, and record a group_id shared by all tasks
            # in this invocation.  Tasks use the global workflow_id (not the
            # sub-workflow id) so that campaign_id / workflow_id are uniform.
            group_id = str(uuid.uuid4())
            graph_name = name or self._node_name(serialized, name)
            self._interceptor.send_graph_workflow(graph_name, group_id)
            self._graph_runs[run_id] = group_id
            source_agent_id = inputs.get("_source_agent_id")
            if source_agent_id:
                self._graph_source_agent[run_id] = str(source_agent_id)

            custom: dict[str, Any] = {"tags": tags or [], "graph_name": graph_name}
            if source_agent_id:
                custom["source_agent_id"] = str(source_agent_id)
            # Buffer the skeleton — emitted as ONE complete record in on_chain_end
            self._run_start_task[run_id] = {
                "task_id":    task_id,
                "subtype":    "langgraph_graph",
                "activity_id": graph_name,
                "group_id":   group_id,
                "started_at": self._run_start[run_id],
                "used":       {"inputs": _safe_clip(inputs)},
                "custom_metadata": custom,
            }
        else:
            # Node or sub-chain execution.
            node = self._node_name(serialized, name)
            # Find the enclosing graph run for group_id and source_agent_id.
            graph_run_id = None
            if parent_run_id and parent_run_id in self._graph_runs:
                graph_run_id = parent_run_id
            elif parent_run_id:
                graph_run_id = self._run_to_graph_run.get(parent_run_id)
            group_id = self._graph_runs.get(graph_run_id) if graph_run_id else None
            source_agent_id = self._graph_source_agent.get(graph_run_id) if graph_run_id else None

            # Record child → graph mapping so on_llm_end / on_tool_end can find group_id
            if graph_run_id is not None:
                self._run_to_graph_run[run_id] = graph_run_id

            parent_task_id = self._run_to_task.get(parent_run_id) if parent_run_id else None
            custom = {"tags": tags or [], "node_name": node}
            if source_agent_id:
                custom["source_agent_id"] = source_agent_id
            # Buffer the skeleton — emitted as ONE complete record in on_chain_end
            skeleton: dict[str, Any] = {
                "task_id":      task_id,
                "subtype":      "langgraph_node",
                "activity_id":  node,
                "group_id":     group_id,
                "started_at":   self._run_start[run_id],
                "used":         {"inputs": _safe_clip(inputs)},
                "custom_metadata": custom,
            }
            if parent_task_id:
                skeleton["parent_task_id"] = parent_task_id
            self._run_start_task[run_id] = skeleton

        self._record("chain_start", time.perf_counter() - t0)

    def on_chain_end(
        self,
        outputs: dict,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        t0 = time.perf_counter()
        started_at = self._run_start.pop(run_id, time.time())
        tel_start  = self._run_tel_start.pop(run_id, None)
        tel_end    = self._tel()
        self._run_to_task.pop(run_id, None)

        is_graph = run_id in self._graph_runs
        if is_graph:
            self._graph_runs.pop(run_id)
            self._graph_source_agent.pop(run_id, None)
        else:
            self._run_to_graph_run.pop(run_id, None)

        # Merge buffered start skeleton with completion data → ONE complete record
        task: dict[str, Any] = self._run_start_task.pop(run_id, {})
        task.update({
            "ended_at":  time.time(),
            "status":    "FINISHED",
            "generated": {"outputs": _safe_clip(outputs)},
        })
        if tel_start is not None:
            task["telemetry_at_start"] = _tel_to_dict(tel_start)
        if tel_end is not None:
            task["telemetry_at_end"] = _tel_to_dict(tel_end)

        self._interceptor.intercept_task(task)
        self._record("chain_end", time.perf_counter() - t0)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        t0 = time.perf_counter()
        started_at = self._run_start.pop(run_id, time.time())
        tel_start  = self._run_tel_start.pop(run_id, None)
        self._run_to_task.pop(run_id, None)
        is_graph = run_id in self._graph_runs
        if is_graph:
            self._graph_runs.pop(run_id, None)
            self._graph_source_agent.pop(run_id, None)
        else:
            self._run_to_graph_run.pop(run_id, None)

        task: dict[str, Any] = self._run_start_task.pop(run_id, {})
        task.update({
            "ended_at": time.time(),
            "status":   "ERROR",
            "stderr":   str(error),
        })
        if tel_start is not None:
            task["telemetry_at_start"] = _tel_to_dict(tel_start)
        self._interceptor.intercept_task(task)
        self._record("chain_error", time.perf_counter() - t0)

    # --- LLM events ---

    def on_llm_start(
        self,
        serialized: dict | None,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        t0 = time.perf_counter()
        self._run_start[run_id] = time.time()
        self._run_tel_start[run_id] = self._tel()
        task_id = self._task_id_for(run_id)
        # Propagate enclosing graph run mapping for group_id lookup in on_llm_end
        if parent_run_id is not None:
            graph_run_id = self._run_to_graph_run.get(parent_run_id) or (
                parent_run_id if parent_run_id in self._graph_runs else None
            )
            if graph_run_id is not None:
                self._run_to_graph_run[run_id] = graph_run_id
        group_id = self._group_id_for(run_id)
        parent_task_id = self._run_to_task.get(parent_run_id) if parent_run_id else None
        model_name = self._model_name(serialized)
        skeleton: dict[str, Any] = {
            "task_id":     task_id,
            "subtype":     "llm_call",
            "activity_id": model_name,
            "group_id":    group_id,
            "started_at":  self._run_start[run_id],
            "used":        {"prompts": _safe_clip(prompts), "model": model_name},
            "custom_metadata": {"model": model_name},
        }
        if parent_task_id:
            skeleton["parent_task_id"] = parent_task_id
        self._run_start_task[run_id] = skeleton
        self._record("llm_start", time.perf_counter() - t0)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        t0 = time.perf_counter()
        started_at = self._run_start.pop(run_id, time.time())
        tel_start  = self._run_tel_start.pop(run_id, None)
        tel_end    = self._tel()
        self._run_to_task.pop(run_id, None)
        self._run_to_graph_run.pop(run_id, None)

        # Extract token usage and text from LangChain LLMResult
        usage: dict = {}
        text: str = ""
        model_name: str = "unknown"
        try:
            gen = response.generations
            if gen and gen[0]:
                text = gen[0][0].text if hasattr(gen[0][0], "text") else str(gen[0][0])
            llm_output = response.llm_output or {}
            usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
            model_name = llm_output.get("model_name") or llm_output.get("model") or "unknown"
        except Exception:
            pass

        # Merge buffered start skeleton (includes prompts in used) with response data
        task: dict[str, Any] = self._run_start_task.pop(run_id, {})
        task.update({
            "ended_at": time.time(),
            "status":   "FINISHED",
            "generated": {
                "text":              text,
                "prompt_tokens":     usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens":      usage.get("total_tokens"),
                "model":             model_name,
            },
        })
        # Update activity_id and model in custom_metadata if model is now known
        if model_name != "unknown":
            task["activity_id"] = model_name
            task.setdefault("custom_metadata", {})["model"] = model_name
        if tel_start is not None:
            task["telemetry_at_start"] = _tel_to_dict(tel_start)
        if tel_end is not None:
            task["telemetry_at_end"] = _tel_to_dict(tel_end)

        self._interceptor.intercept_task(task)
        self._record("llm_end", time.perf_counter() - t0)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        t0 = time.perf_counter()
        self._run_start.pop(run_id, None)
        tel_start = self._run_tel_start.pop(run_id, None)
        self._run_to_task.pop(run_id, None)
        self._run_to_graph_run.pop(run_id, None)

        task: dict[str, Any] = self._run_start_task.pop(run_id, {})
        task.update({
            "ended_at": time.time(),
            "status":   "ERROR",
            "stderr":   str(error),
        })
        if tel_start is not None:
            task["telemetry_at_start"] = _tel_to_dict(tel_start)
        self._interceptor.intercept_task(task)
        self._record("llm_error", time.perf_counter() - t0)

    # --- Chat model events (same as LLM but different entry point) ---

    def on_chat_model_start(
        self,
        serialized: dict | None,
        messages: list,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        t0 = time.perf_counter()
        self._run_start[run_id] = time.time()
        self._run_tel_start[run_id] = self._tel()
        task_id = self._task_id_for(run_id)
        # Propagate enclosing graph run mapping for group_id lookup in on_llm_end
        if parent_run_id is not None:
            graph_run_id = self._run_to_graph_run.get(parent_run_id) or (
                parent_run_id if parent_run_id in self._graph_runs else None
            )
            if graph_run_id is not None:
                self._run_to_graph_run[run_id] = graph_run_id
        group_id = self._group_id_for(run_id)
        parent_task_id = self._run_to_task.get(parent_run_id) if parent_run_id else None
        model_name = self._model_name(serialized)
        # Serialize messages: each message is a list of BaseMessage objects
        serialized_messages = _safe_clip(
            [[m.content if hasattr(m, "content") else str(m) for m in turn] for turn in messages]
        )
        skeleton: dict[str, Any] = {
            "task_id":     task_id,
            "subtype":     "llm_call",
            "activity_id": model_name,
            "group_id":    group_id,
            "started_at":  self._run_start[run_id],
            "used":        {"messages": serialized_messages, "model": model_name},
            "custom_metadata": {"model": model_name},
        }
        if parent_task_id:
            skeleton["parent_task_id"] = parent_task_id
        self._run_start_task[run_id] = skeleton
        self._record("chat_model_start", time.perf_counter() - t0)

    # on_chat_model_end fires on_llm_end in practice for most LangChain models

    # --- Tool events ---

    def on_tool_start(
        self,
        serialized: dict | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        t0 = time.perf_counter()
        self._run_start[run_id] = time.time()
        self._run_tel_start[run_id] = self._tel()
        tool_name = (serialized or {}).get("name") or "unknown_tool"
        self._task_id_for(run_id)
        parent_task_id = self._run_to_task.get(parent_run_id) if parent_run_id else None
        # Propagate enclosing graph run mapping for group_id lookup in on_tool_end
        if parent_run_id is not None:
            graph_run_id = self._run_to_graph_run.get(parent_run_id) or (
                parent_run_id if parent_run_id in self._graph_runs else None
            )
            if graph_run_id is not None:
                self._run_to_graph_run[run_id] = graph_run_id
        group_id = self._group_id_for(run_id)
        task_id = self._run_to_task[run_id]
        # Buffer skeleton — emitted as ONE complete record in on_tool_end
        skeleton: dict[str, Any] = {
            "task_id":     task_id,
            "subtype":     "tool_call",
            "activity_id": tool_name,
            "group_id":    group_id,
            "started_at":  self._run_start[run_id],
            "used":        {"input": _safe_clip(input_str)},
            "custom_metadata": {"tool_name": tool_name},
        }
        if parent_task_id:
            skeleton["parent_task_id"] = parent_task_id
        self._run_start_task[run_id] = skeleton
        self._record("tool_start", time.perf_counter() - t0)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        t0 = time.perf_counter()
        tel_start = self._run_tel_start.pop(run_id, None)
        tel_end   = self._tel()
        self._run_start.pop(run_id, None)
        self._run_to_task.pop(run_id, None)
        self._run_to_graph_run.pop(run_id, None)

        task: dict[str, Any] = self._run_start_task.pop(run_id, {})
        task.update({
            "ended_at":  time.time(),
            "status":    "FINISHED",
            "generated": {"output": _safe_clip(output)},
        })
        if tel_start is not None:
            task["telemetry_at_start"] = _tel_to_dict(tel_start)
        if tel_end is not None:
            task["telemetry_at_end"] = _tel_to_dict(tel_end)

        self._interceptor.intercept_task(task)
        self._record("tool_end", time.perf_counter() - t0)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        t0 = time.perf_counter()
        tel_start = self._run_tel_start.pop(run_id, None)
        self._run_start.pop(run_id, None)
        self._run_to_task.pop(run_id, None)
        self._run_to_graph_run.pop(run_id, None)

        task: dict[str, Any] = self._run_start_task.pop(run_id, {})
        task.update({
            "ended_at": time.time(),
            "status":   "ERROR",
            "stderr":   str(error),
        })
        if tel_start is not None:
            task["telemetry_at_start"] = _tel_to_dict(tel_start)
        self._interceptor.intercept_task(task)
        self._record("tool_error", time.perf_counter() - t0)


def _build_handler_class():
    """
    Lazily build a concrete BaseCallbackHandler subclass so LangChain/LangGraph
    is only imported when the plugin is actually used.
    """
    from langchain_core.callbacks import BaseCallbackHandler

    class _ConcreteHandler(FlowceptLangGraphCallback, BaseCallbackHandler):
        """Concrete LangChain callback handler with FlowCept provenance."""

        def __init__(self, interceptor: LangGraphInterceptor, stats: _ProvenanceStats | None) -> None:
            # BaseCallbackHandler.__init__ must be called before our __init__
            # because it sets ignore_* flags that LangChain checks.
            BaseCallbackHandler.__init__(self)
            FlowceptLangGraphCallback.__init__(self, interceptor, stats)

        # Tell LangChain we handle all event types
        ignore_llm = False
        ignore_chain = False
        ignore_agent = False
        ignore_chat_model = False

    return _ConcreteHandler


_HANDLER_CLASS = None


# ---------------------------------------------------------------------------
# Public plugin class
# ---------------------------------------------------------------------------

class FlowceptLangGraphPlugin:
    """
    FlowCept provenance plugin for LangGraph workflows.

    Captures the full provenance graph for each graph invocation:
    graph run → node tasks → LLM/tool child tasks, with CPU/memory telemetry
    and node enrichment.

    Parameters
    ----------
    config : dict, optional
        Plugin configuration keys:
          enabled       (bool, default True)
          workflow_name (str, default "langgraph-workflow")
          dump_path     (str, optional) — JSONL path written on stop.
          performance_tracking (bool, default True)
          perf_csv      (str, optional) — explicit path for timing CSV.

    Usage
    -----
    Option A — explicit start/stop::

        plugin = FlowceptLangGraphPlugin(config={...})
        plugin.start()
        result = graph.invoke(state, config={"callbacks": [plugin.callback_handler]})
        plugin.stop()

    Option B — context manager::

        with FlowceptLangGraphPlugin(config={...}) as plugin:
            result = graph.invoke(state, config={"callbacks": [plugin.callback_handler]})
    """

    def __init__(self, config: dict | None = None, _shared_interceptor=None) -> None:
        cfg = config or {}
        self._enabled: bool = cfg.get("enabled", True)
        self._workflow_name: str = cfg.get("workflow_name", "langgraph-workflow")
        self._dump_path: str | None = cfg.get("dump_path", None)
        self._perf_tracking: bool = cfg.get("performance_tracking", True)
        self._perf_csv: str | None = cfg.get("perf_csv", None)
        # When _shared_interceptor is supplied the plugin does NOT start/stop
        # its own Flowcept instance — it reuses the caller's buffer entirely.
        self._shared_interceptor = _shared_interceptor
        self._interceptor = _shared_interceptor or LangGraphInterceptor()
        self._owns_interceptor: bool = _shared_interceptor is None
        self._stats: _ProvenanceStats | None = None
        self._handler: FlowceptLangGraphCallback | None = None
        self._started = False

    @classmethod
    def from_academy_plugin(
        cls,
        academy_plugin: Any,
        config: dict | None = None,
    ) -> "FlowceptLangGraphPlugin":
        """
        Create a LangGraph plugin that shares the buffer of an already-started
        FlowceptAcademyPlugin.

        All provenance records — from both Academy agents and LangGraph nodes —
        are written to the exact same FlowCept in-memory buffer and end up in
        the same JSONL file when the Academy plugin stops.

        Parameters
        ----------
        academy_plugin : FlowceptAcademyPlugin
            A plugin that has already been started (``plugin.start()`` called).
        config : dict, optional
            Same keys as FlowceptLangGraphPlugin.__init__ (enabled,
            performance_tracking, perf_csv).  dump_path is ignored because the
            Academy plugin owns the buffer and the dump path.

        Returns
        -------
        FlowceptLangGraphPlugin
            A started plugin whose ``callback_handler`` is ready for use.

        Example
        -------
        ::

            academy_plugin = FlowceptAcademyPlugin(config={...}).start()
            lg_plugin = FlowceptLangGraphPlugin.from_academy_plugin(academy_plugin)
            # lg_plugin is already started — no need to call .start()
            result = graph.invoke(state, config={"callbacks": [lg_plugin.callback_handler]})
            # Stop only the Academy plugin; it flushes the shared buffer.
            academy_plugin.stop()
        """
        interceptor = academy_plugin._interceptor
        inst = cls(config=config, _shared_interceptor=interceptor)
        inst._started = True  # mark as started before building handler

        # Inherit the Academy plugin's dump_path so the perf CSV lands next to
        # the shared JSONL file (e.g. combined_provenance_langgraph_perf.csv).
        if inst._dump_path is None and getattr(academy_plugin, "_dump_path", None):
            base = os.path.splitext(academy_plugin._dump_path)[0]
            inst._perf_csv = inst._perf_csv or f"{base}_langgraph_perf.csv"

        global _HANDLER_CLASS
        if _HANDLER_CLASS is None:
            _HANDLER_CLASS = _build_handler_class()
        inst._stats = _ProvenanceStats() if (config or {}).get("performance_tracking", True) else None
        inst._handler = _HANDLER_CLASS(interceptor, inst._stats)
        return inst

    @property
    def callback_handler(self) -> FlowceptLangGraphCallback:
        """
        The LangChain callback handler to pass as ``config={"callbacks": [...]}``.
        Call ``start()`` before accessing this property.
        """
        if self._handler is None:
            raise RuntimeError("Plugin not started — call plugin.start() first.")
        return self._handler

    def start(self) -> "FlowceptLangGraphPlugin":
        if not self._enabled or self._started:
            return self
        if not self._owns_interceptor:
            # Already configured by from_academy_plugin(); nothing to start.
            return self
        try:
            self._stats = _ProvenanceStats() if self._perf_tracking else None
            self._interceptor.start(self._workflow_name)

            global _HANDLER_CLASS
            if _HANDLER_CLASS is None:
                _HANDLER_CLASS = _build_handler_class()

            self._handler = _HANDLER_CLASS(self._interceptor, self._stats)
            self._started = True

            wf_id = self._interceptor._workflow_id
            campaign_id = self._interceptor._campaign_id
            print(
                f"[FlowceptLangGraphPlugin] Started\n"
                f"  workflow_id : {wf_id}\n"
                f"  campaign_id : {campaign_id}\n"
                f"  Capturing   : graph runs (sub-workflows), node execution "
                f"(parent_task_id), LLM + tool calls (child tasks, telemetry).\n"
                f"  Buffer → flowcept_buffer_{wf_id}.jsonl on stop.",
                flush=True,
            )
        except Exception as e:
            print(
                f"[FlowceptLangGraphPlugin] WARNING: failed to start — {e!r}. "
                "Continuing without provenance capture.",
                flush=True,
            )
            _log.exception("FlowceptLangGraphPlugin start failed")
            self._enabled = False
        return self

    def flush(self, dump_path: str | None = None) -> str | None:
        """Flush the in-memory buffer to disk without stopping the plugin."""
        if not self._started:
            return None
        path = dump_path or self._dump_path
        if path is None:
            wf_id = self._interceptor._workflow_id or "unknown"
            path = f"flowcept_buffer_{wf_id}.jsonl"
        try:
            self._interceptor.flush(path)
            print(f"[FlowceptLangGraphPlugin] Buffer flushed → {path}", flush=True)
            return path
        except Exception as e:
            print(f"[FlowceptLangGraphPlugin] Warning: flush failed — {e!r}", flush=True)
            return None

    def stop(self) -> None:
        if not self._started:
            return
        if not self._owns_interceptor:
            # Buffer is owned by the Academy plugin — do not stop or flush it here.
            # Just print the perf stats if enabled.
            self._started = False
            print(
                "[FlowceptLangGraphPlugin] Detached from shared buffer "
                "(buffer flushed by the owning plugin).",
                flush=True,
            )
            self._maybe_write_perf_csv()
            return
        try:
            self._interceptor.stop(dump_path=self._dump_path)
        except Exception as e:
            print(f"[FlowceptLangGraphPlugin] Warning during stop: {e!r}", flush=True)
        self._started = False
        print("[FlowceptLangGraphPlugin] Stopped. Provenance buffer flushed.", flush=True)
        self._maybe_write_perf_csv()

    def _maybe_write_perf_csv(self) -> None:

        if self._stats is not None:
            print(
                "\n[FlowceptLangGraphPlugin] Provenance overhead report:\n"
                + self._stats.summary()
                + "\n  (N = event count; Total/Mean/Min/Max in ms/µs respectively)\n",
                flush=True,
            )
            wf_id = self._interceptor._workflow_id
            if self._perf_csv:
                csv_path = self._perf_csv
            elif self._dump_path:
                base = os.path.splitext(self._dump_path)[0]
                csv_path = f"{base}_perf.csv"
            else:
                csv_path = f"langgraph_provenance_perf_{wf_id}.csv"
            try:
                self._stats.to_csv(csv_path, workflow_id=wf_id)
                print(
                    f"[FlowceptLangGraphPlugin] Performance stats written to {csv_path}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[FlowceptLangGraphPlugin] Warning: could not write perf CSV — {e!r}",
                    flush=True,
                )


    def __enter__(self) -> "FlowceptLangGraphPlugin":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Serialisation helper (identical to flowcept_plugin.py)
# ---------------------------------------------------------------------------

def _safe_clip(obj: Any, _depth: int = 0) -> Any:
    """Recursively convert objects to JSON-serialisable form without truncation."""
    if _depth > 8:
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_clip(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_clip(v, _depth + 1) for v in obj]
    return repr(obj)


def _tel_to_dict(tel: Any) -> dict | None:
    if tel is None:
        return None
    try:
        return tel.to_dict()
    except Exception:
        return None
