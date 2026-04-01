# academy_coscientist/plugins/flowcept_plugin.py
"""
Generic FlowCept provenance plugin for Academy-based applications.

Designed around FlowCept's provenance model (TaskObject, WorkflowObject,
TelemetryCapture, Status).

Provenance hierarchy produced:
  WorkflowObject  (one per run — top-level workflow)
    └─ WorkflowObject  (one per agent — sub-workflow, parent_workflow_id → top)
         └─ TaskObject  activity_id=action_name      subtype=academy_action
              └─ TaskObject  activity_id=llm_call_type  subtype=llm_call  (parent_task_id)
         └─ TaskObject  activity_id=loop_name         subtype=academy_loop
         └─ TaskObject  activity_id=agent_startup|shutdown  subtype=academy_lifecycle

Key FlowCept fields used correctly:
  task_id                    — uuid per task record
  workflow_id                — per-agent sub-workflow id (linked to top-level via parent)
  campaign_id                — from Flowcept.campaign_id
  parent_task_id             — LLM tasks are children of the action that spawned them
  group_id                   — loop events share a group_id for grouping
  activity_id                — action name, llm call type, loop name, lifecycle event
  subtype                    — academy_action | academy_loop | academy_lifecycle | llm_call
  used / generated           — inputs and outputs
  status                     — proper Status enum values (FINISHED | ERROR)
  telemetry_at_start/end     — CPU/memory snapshots via TelemetryCapture
  node_name, hostname, etc.  — via TaskObject.enrich_task_dict()

LLM calls are linked to their parent action via a contextvars.ContextVar so that
any LLM call happening inside an async action is automatically a child of that action.

Usage (zero agent code changes required):

    import my_llm_logging
    plugin = FlowceptAcademyPlugin(
        config={"workflow_name": "my-app", "dump_path": "provenance.jsonl"},
        llm_hook_register=my_llm_logging.register_llm_hook,
        llm_hook_unregister=my_llm_logging.unregister_llm_hook,
    )
    plugin.start()
    try:
        asyncio.run(my_academy_app())
    finally:
        plugin.stop()

LLM hook contract:
    The hook function receives a dict payload with at minimum:
      "type"     — one of: parsed_json_result | chat_completion |
                           embed_result_local | embed_result_openai
      "context"  — dict of caller-supplied key/value tags (agent, call_type, ...)
      "model"    — model identifier string
    All top-level fields from parsed JSON responses are automatically hoisted
    into the generated record so they are directly queryable.
"""

from __future__ import annotations

# Set FlowCept to offline / no-MQ / no-DB mode BEFORE any flowcept import.
import os
os.environ.setdefault("FLOWCEPT_USE_DEFAULT", "1")

import contextvars
import json as _json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any


# ---------------------------------------------------------------------------
# Provenance overhead timer
# ---------------------------------------------------------------------------

class _ProvenanceStats:
    """
    Lightweight thread-safe accumulator for provenance capture timings.

    Records call counts, total/min/max wall-clock elapsed time (seconds) per
    named *category*.  All arithmetic uses ``time.perf_counter()`` so the
    resolution is sub-microsecond on all supported platforms.

    Categories captured automatically:
      ``action_emit``   — time spent inside ``_emit_action()``
      ``loop_emit``     — time spent inside ``_emit_loop_event()``
      ``lifecycle_emit``— time spent inside ``_emit_lifecycle()``
      ``llm_hook``      — time spent processing each LLM call in ``_on_llm_call()``
      ``intercept_task``— time spent inside ``AcademyInterceptor.intercept_task()``
                          (serialisation + buffer append)
      ``flush``         — wall-clock time for ``dump_buffer()`` + ``stop()``
    """

    __slots__ = ("_lock", "_counts", "_totals", "_mins", "_maxs", "_raw")

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._counts: dict[str, int] = {}
        self._totals: dict[str, float] = {}
        self._mins:   dict[str, float] = {}
        self._maxs:   dict[str, float] = {}
        # Raw per-event buffer: list of (timestamp_utc, category, elapsed_s)
        self._raw: list[tuple[str, str, float]] = []

    def record(self, category: str, elapsed: float) -> None:
        """Add one observation for *category* (elapsed in seconds)."""
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
        """Return a formatted table of all recorded categories."""
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
        """Append one row per captured event to a CSV file (header written once).

        Each row is one individual provenance-capture observation so the file
        can be used for distribution analysis, percentile queries, or time-series
        plots without losing any detail.

        Columns
        -------
        timestamp_utc, workflow_id, category, elapsed_us
        """
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

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()
            self._totals.clear()
            self._mins.clear()
            self._maxs.clear()
            self._raw.clear()


# Module-level singleton; activated when the plugin starts.
_PROV_STATS: _ProvenanceStats | None = None


@contextmanager
def _timed(category: str):
    """Context manager: record wall-clock time for *category* in ``_PROV_STATS``."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if _PROV_STATS is not None:
            _PROV_STATS.record(category, time.perf_counter() - t0)

_log = logging.getLogger(__name__)

# ContextVar: holds the task_id of the currently-executing Academy action (or
# loop) so that LLM calls made inside it can be registered as child tasks.
_current_action_task_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_action_task_id", default=None
)

# ContextVar: holds the real Academy agent ID (e.g. "AgentId<82278eb6>") for
# the agent whose coroutine is currently executing.  Set once per agent at
# startup so every LLM call emitted from within that agent carries the right ID.
_current_academy_agent_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_academy_agent_id", default=None
)

# LLM payload types that carry a complete request+response pair.
_CAPTURE_LLM_TYPES = frozenset({
    "parsed_json_result",
    "chat_completion",
    "embed_result_local",
    "embed_result_openai",
})


# ---------------------------------------------------------------------------
# Academy interceptor — wraps FlowCept's InstrumentationInterceptor
# ---------------------------------------------------------------------------

class AcademyInterceptor:
    """
    Wraps FlowCept's InstrumentationInterceptor and exposes:
      - start(workflow_name)       — initialises Flowcept and the interceptor
      - stop(dump_path)            — flushes the buffer to JSONL
      - intercept_task(task_dict)  — enriches with FlowCept standard fields and sends
      - send_agent_workflow(...)   — emits a WorkflowObject for an agent
      - telemetry_capture          — the underlying TelemetryCapture instance
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
        with _timed("flush"):
            try:
                if dump_path:
                    self._flowcept.dump_buffer(dump_path)
                self._flowcept.stop()
            except Exception as e:
                _log.warning("FlowCept stop error: %r", e)

    @property
    def telemetry_capture(self):
        return self._interceptor.telemetry_capture if self._interceptor else None

    def send_agent_workflow(self, agent_type: str, agent_id: str) -> str:
        """Emit a WorkflowObject for an agent sub-workflow; return its workflow_id."""
        if self._interceptor is None:
            return str(uuid.uuid4())
        from flowcept.commons.flowcept_dataclasses.workflow_object import WorkflowObject
        wf = WorkflowObject()
        wf.workflow_id = str(uuid.uuid4())
        wf.name = f"{agent_type}:{agent_id}"
        wf.campaign_id = self._campaign_id
        wf.parent_workflow_id = self._workflow_id
        wf.custom_metadata = {"agent_type": agent_type, "agent_id": agent_id}
        self._interceptor.send_workflow_message(wf)
        return wf.workflow_id

    def send_graph_workflow(self, graph_name: str, group_id: str) -> str:
        """Emit a WorkflowObject for a LangGraph run sub-workflow; return its workflow_id.

        Same structure as send_agent_workflow but with graph-oriented metadata.
        Allows AcademyInterceptor to serve as the shared interceptor for both
        FlowceptAcademyPlugin and FlowceptLangGraphPlugin so that all provenance
        records go into the same FlowCept buffer.
        """
        if self._interceptor is None:
            return str(uuid.uuid4())
        from flowcept.commons.flowcept_dataclasses.workflow_object import WorkflowObject
        wf = WorkflowObject()
        wf.workflow_id = str(uuid.uuid4())
        wf.name = graph_name
        wf.campaign_id = self._campaign_id
        wf.parent_workflow_id = self._workflow_id
        wf.custom_metadata = {"graph_name": graph_name, "group_id": group_id}
        self._interceptor.send_workflow_message(wf)
        return wf.workflow_id

    def intercept_task(self, task_dict: dict) -> None:
        """Enrich task_dict with FlowCept standard fields and append to buffer."""
        if self._interceptor is None:
            return
        with _timed("intercept_task"):
            from flowcept.commons.flowcept_dataclasses.task_object import TaskObject
            from flowcept.commons.vocabulary import Status

            task_dict.setdefault("type", "task")
            task_dict.setdefault("task_id", str(uuid.uuid4()))
            task_dict.setdefault("workflow_id", self._workflow_id)
            task_dict.setdefault("campaign_id", self._campaign_id)

            # Normalise status to proper enum value
            raw = task_dict.get("status", "FINISHED")
            if isinstance(raw, str):
                try:
                    task_dict["status"] = Status[raw].value
                except KeyError:
                    task_dict["status"] = Status.FINISHED.value

            # Enrich with node_name, login_name, hostname, public_ip, private_ip
            TaskObject.enrich_task_dict(task_dict)

            self._interceptor.intercept(task_dict)


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_PATCHER_INSTALLED: bool = False
_ACTIVE_INTERCEPTOR: AcademyInterceptor | None = None

# Per-agent sub-workflow IDs  {agent_id_str -> sub_workflow_id}
_AGENT_WORKFLOWS: dict[str, str] = {}

# Reverse map: Academy agent ID → agent class name, for enriching records that
# only have the academy ID (e.g. embed calls whose ctx has no "agent" key).
_AGENT_ID_TO_TYPE: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Academy Runtime patcher
# ---------------------------------------------------------------------------

def _install_runtime_patches() -> None:
    """
    Patch academy.runtime.Runtime at the class level so every Runtime instance
    (one per agent) captures provenance without any agent code changes.
    """
    global _PATCHER_INSTALLED
    if _PATCHER_INSTALLED:
        return

    from academy.runtime import Runtime

    # ---- @action dispatch ------------------------------------------------
    _orig_action = Runtime.action

    async def _action_with_prov(
        self, action: str, source_id: Any, *, args: Any, kwargs: Any
    ) -> Any:
        interceptor = _ACTIVE_INTERCEPTOR
        if interceptor is None:
            return await _orig_action(self, action, source_id, args=args, kwargs=kwargs)

        tel_cap = interceptor.telemetry_capture
        task_id = str(uuid.uuid4())
        tel_start = tel_cap.capture() if tel_cap else None
        started_at = time.time()

        # Propagate action's task_id into context so nested LLM calls see it
        token = _current_action_task_id.set(task_id)
        error: BaseException | None = None
        result: Any = None
        try:
            result = await _orig_action(self, action, source_id, args=args, kwargs=kwargs)
            return result
        except Exception as exc:
            error = exc
            raise
        finally:
            _current_action_task_id.reset(token)
            ended_at = time.time()
            tel_end = tel_cap.capture() if tel_cap else None
            try:
                _emit_action(
                    interceptor, self, action, source_id,
                    args, kwargs, result, error,
                    task_id, started_at, ended_at, tel_start, tel_end,
                )
            except Exception:
                pass

    Runtime.action = _action_with_prov  # type: ignore[method-assign]

    # ---- @loop execution -------------------------------------------------
    _orig_execute_loop = Runtime._execute_loop

    async def _loop_with_prov(self, name: str, method: Any) -> None:
        interceptor = _ACTIVE_INTERCEPTOR
        if interceptor is None:
            return await _orig_execute_loop(self, name, method)

        tel_cap = interceptor.telemetry_capture
        task_id = str(uuid.uuid4())
        group_id = str(uuid.uuid4())
        tel_start = tel_cap.capture() if tel_cap else None
        t0 = time.time()
        _emit_loop_event(interceptor, self, name, "start", task_id, group_id, t0, tel_start)

        # Set context so LLM calls made inside the loop body are linked to
        # this loop task.  The action patch will override this for any
        # @action dispatches that happen concurrently.
        token = _current_action_task_id.set(task_id)
        try:
            await _orig_execute_loop(self, name, method)
            tel_end = tel_cap.capture() if tel_cap else None
            _emit_loop_event(interceptor, self, name, "exit", task_id, group_id, t0, tel_end)
        except Exception as exc:
            tel_end = tel_cap.capture() if tel_cap else None
            _emit_loop_event(
                interceptor, self, name, "error", task_id, group_id, t0, tel_end, error=exc
            )
            raise
        finally:
            _current_action_task_id.reset(token)

    Runtime._execute_loop = _loop_with_prov  # type: ignore[method-assign]

    # ---- agent startup ---------------------------------------------------
    _orig_start = Runtime._start

    async def _start_with_prov(self) -> None:
        # CRITICAL: set the ContextVar BEFORE calling _orig_start.
        #
        # Academy's _start() creates all loop tasks and the exchange listener
        # task via asyncio.create_task / spawn_guarded_background_task.  Each
        # new task gets a *copy* of the context at the moment of creation.
        # If we set the ContextVar after _orig_start returns, those tasks
        # already have stale context copies and will never see the agent ID.
        # Setting it first ensures every task spawned during startup inherits
        # the correct Academy agent ID automatically.
        agent_type, agent_id = _agent_info(self)
        _current_academy_agent_id.set(agent_id)

        await _orig_start(self)

        interceptor = _ACTIVE_INTERCEPTOR
        if interceptor is None:
            return
        try:
            sub_wf_id = interceptor.send_agent_workflow(agent_type, agent_id)
            _AGENT_WORKFLOWS[agent_id] = sub_wf_id
            _AGENT_ID_TO_TYPE[agent_id] = agent_type
            _emit_lifecycle(interceptor, self, "agent_startup")
        except Exception:
            pass

    Runtime._start = _start_with_prov  # type: ignore[method-assign]

    # ---- agent shutdown --------------------------------------------------
    _orig_shutdown = Runtime._shutdown

    async def _shutdown_with_prov(self) -> None:
        interceptor = _ACTIVE_INTERCEPTOR
        if interceptor is not None:
            try:
                _emit_lifecycle(interceptor, self, "agent_shutdown")
            except Exception:
                pass
        await _orig_shutdown(self)

    Runtime._shutdown = _shutdown_with_prov  # type: ignore[method-assign]

    _PATCHER_INSTALLED = True


def _uninstall_runtime_patches() -> None:
    """Clear the active interceptor so all patches become no-ops."""
    global _ACTIVE_INTERCEPTOR
    _ACTIVE_INTERCEPTOR = None
    _AGENT_WORKFLOWS.clear()
    _AGENT_ID_TO_TYPE.clear()


# ---------------------------------------------------------------------------
# Emit helpers
# ---------------------------------------------------------------------------

def _agent_info(runtime: Any) -> tuple[str, str]:
    try:
        agent_type = type(runtime.agent).__name__
    except Exception:
        agent_type = "unknown"
    try:
        agent_id = str(runtime.agent_id)
    except Exception:
        agent_id = "unknown"
    return agent_type, agent_id


def _agent_workflow_id(agent_id: str, fallback: str | None) -> str | None:
    return _AGENT_WORKFLOWS.get(agent_id, fallback)


def _parse_source(source_id: Any) -> tuple[str | None, str | None, str | None]:
    """
    Parse Academy source_id into (raw_str, agent_uid, source_workflow_id).

    source_id is either:
      - UserId<hex>  → a user/launcher, not an agent
      - AgentId<hex> → another agent; look up its sub-workflow
    Returns (source_str, source_agent_id, source_workflow_id).
    """
    s = str(source_id)
    if s.startswith("AgentId"):
        wf_id = _AGENT_WORKFLOWS.get(s)
        return s, s, wf_id
    return s, None, None


def _tel_to_dict(tel: Any) -> dict | None:
    if tel is None:
        return None
    try:
        return tel.to_dict()
    except Exception:
        return None


def _emit_action(
    interceptor: AcademyInterceptor,
    runtime: Any,
    action: str,
    source_id: Any,
    args: Any,
    kwargs: Any,
    result: Any,
    error: BaseException | None,
    task_id: str,
    started_at: float,
    ended_at: float,
    tel_start: Any,
    tel_end: Any,
) -> None:
    with _timed("action_emit"):
        agent_type, agent_id = _agent_info(runtime)
        source_str, source_agent_id, source_workflow_id = _parse_source(source_id)

        custom: dict[str, Any] = {
            "agent_type": agent_type,
            "source_id": source_str,
            # Always record whether this was a cross-agent call
            "cross_agent_call": source_agent_id is not None,
        }
        if source_agent_id is not None:
            # Explicit inter-agent provenance: who called this action and from
            # which sub-workflow, so the full agent-to-agent graph is queryable.
            custom["source_agent_id"] = source_agent_id
            if source_workflow_id is not None:
                custom["source_workflow_id"] = source_workflow_id

        task: dict[str, Any] = {
            "task_id": task_id,
            "subtype": "academy_action",
            "activity_id": action,
            "agent_id": agent_id,
            "custom_metadata": custom,
            "started_at": started_at,
            "ended_at": ended_at,
            "status": "ERROR" if error else "FINISHED",
            "used": {"args": _safe_clip(args), "kwargs": _safe_clip(kwargs)},
            "generated": _safe_clip(result) if error is None else None,
            "stderr": str(error) if error else None,
        }
        if tel_start is not None:
            task["telemetry_at_start"] = _tel_to_dict(tel_start)
        if tel_end is not None:
            task["telemetry_at_end"] = _tel_to_dict(tel_end)
        interceptor.intercept_task(task)


def _emit_loop_event(
    interceptor: AcademyInterceptor,
    runtime: Any,
    loop_name: str,
    event: str,
    task_id: str,
    group_id: str,
    started_at: float,
    tel: Any,
    error: BaseException | None = None,
) -> None:
    with _timed("loop_emit"):
        agent_type, agent_id = _agent_info(runtime)
        task: dict[str, Any] = {
            "task_id": task_id,
            "subtype": "academy_loop",
            "activity_id": loop_name,
            "group_id": group_id,
            "agent_id": agent_id,
            "custom_metadata": {
                "agent_type": agent_type,
                "loop_event": event,
            },
            "started_at": started_at,
            "ended_at": time.time(),
            "status": "ERROR" if error else "FINISHED",
            "stderr": str(error) if error else None,
        }
        tel_key = "telemetry_at_start" if event == "start" else "telemetry_at_end"
        if tel is not None:
            task[tel_key] = _tel_to_dict(tel)
        interceptor.intercept_task(task)


def _emit_lifecycle(
    interceptor: AcademyInterceptor, runtime: Any, event: str
) -> None:
    with _timed("lifecycle_emit"):
        agent_type, agent_id = _agent_info(runtime)
        now = time.time()
        task: dict[str, Any] = {
            "subtype": "academy_lifecycle",
            "activity_id": event,
            "agent_id": agent_id,
            "custom_metadata": {"agent_type": agent_type},
            "started_at": now,
            "ended_at": now,
            "status": "FINISHED",
        }
        interceptor.intercept_task(task)


# ---------------------------------------------------------------------------
# LLM hook — each LLM call becomes a child TaskObject of its parent action
# ---------------------------------------------------------------------------

def _on_llm_call(payload: dict) -> None:
    interceptor = _ACTIVE_INTERCEPTOR
    if interceptor is None:
        return
    call_type = payload.get("type", "")
    if call_type not in _CAPTURE_LLM_TYPES:
        return
    with _timed("llm_hook"):
        _process_llm_call(interceptor, payload, call_type)


def _process_llm_call(interceptor: AcademyInterceptor, payload: dict, call_type: str) -> None:
    """Inner body of ``_on_llm_call``, extracted so ``_timed`` wraps everything."""
    now = time.time()
    ctx = payload.get("context") or {}
    model = payload.get("model", "unknown")
    model_used = payload.get("model_used", model)
    has_error = "error" in payload

    # parent_task_id: links this LLM call to the enclosing action or loop task
    parent_task_id = _current_action_task_id.get(None)

    # agent_id: always use the real Academy ID from the ContextVar (set before
    # task creation so all tasks inherit it).  ctx["agent"] is the class name
    # logged by application code — useful but not a reliable unique identifier.
    academy_agent_id = _current_academy_agent_id.get(None)
    ctx_agent_name = ctx.get("agent") or (
        _AGENT_ID_TO_TYPE.get(academy_agent_id, "unknown") if academy_agent_id else "unknown"
    )
    # Use the real Academy ID as the primary agent_id in provenance records.
    # Fall back to the class name only when the ID is genuinely unavailable.
    agent_id = academy_agent_id or ctx_agent_name

    # --- temperature ---
    temp_requested = payload.get("temperature_requested", payload.get("temperature"))
    temp_sent = payload.get("temperature_sent", payload.get("temperature"))
    temp_suppressed = payload.get("temperature_suppressed", False)

    if temp_requested is None:
        if temp_suppressed:
            temp_null_reason = "reasoning_model_does_not_accept_temperature"
        elif call_type in ("embed_result_local", "embed_result_openai"):
            temp_null_reason = "not_applicable_for_embeddings"
        else:
            temp_null_reason = "not_set"
    else:
        temp_null_reason = None

    # --- used (request / inputs) ---
    used: dict[str, Any] = {
        "model": model,
        "model_used": model_used,
        "call_type": ctx.get("call_type") or call_type,
        "temperature_requested": temp_requested,
        "temperature_sent": temp_sent,
        "temperature_suppressed": temp_suppressed,
        # Always store both the class-name agent and the Academy agent ID so
        # the record is self-contained for provenance queries.
        "agent_class": ctx_agent_name,
        "academy_agent_id": academy_agent_id,
    }
    if temp_null_reason is not None:
        used["temperature_null_reason"] = temp_null_reason
    if "messages" in payload:
        used["messages"] = payload["messages"]
    if "system_instructions" in payload:
        used["system_instructions"] = payload["system_instructions"]
    if "user_prompt" in payload:
        used["user_prompt"] = payload["user_prompt"]
    if "schema_hint" in payload:
        used["schema_hint"] = payload["schema_hint"]
    for k, v in ctx.items():
        used.setdefault(f"ctx_{k}", v)

    # --- generated (response / outputs) ---
    usage = payload.get("usage") or {}
    generated: dict[str, Any] = {
        "finish_reason": payload.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }

    if has_error:
        generated["error"] = str(payload["error"])

    elif call_type == "parsed_json_result":
        # _call_llm_json path: parsed_response contains the full structured output.
        parsed_resp = payload.get("parsed_response", {})
        generated["parsed_response"] = parsed_resp
        generated["fallback_used"] = payload.get("fallback_used", False)
        generated["raw_response_text"] = payload.get("raw_response_text", "")
        # Hoist all top-level fields from the parsed response so they are
        # directly queryable without unwrapping parsed_response.
        if isinstance(parsed_resp, dict):
            for k, v in parsed_resp.items():
                generated.setdefault(k, v)

    elif call_type == "chat_completion":
        response_text = payload.get("text", "")
        generated["response_text"] = response_text

        # Try to parse JSON embedded in the response text.
        parsed_inline: dict | None = None
        if response_text:
            s = response_text.strip()
            if s.startswith("```"):
                lines = s.splitlines()
                s = "\n".join(lines[1:-1]).strip()
            try:
                parsed_inline = _json.loads(s)
            except Exception:
                start = min(
                    (s.find("{") if s.find("{") != -1 else len(s)),
                    (s.find("[") if s.find("[") != -1 else len(s)),
                )
                end = max(s.rfind("}"), s.rfind("]"))
                if 0 <= start < end:
                    try:
                        parsed_inline = _json.loads(s[start : end + 1])
                    except Exception:
                        pass

        if isinstance(parsed_inline, dict):
            generated["parsed_response"] = parsed_inline
            # Hoist all top-level fields so they are directly queryable.
            for k, v in parsed_inline.items():
                generated.setdefault(k, v)

    elif call_type in ("embed_result_local", "embed_result_openai"):
        generated["num_vectors"] = payload.get("num_vectors")
        generated["embed_dim"] = payload.get("dim")

    task: dict[str, Any] = {
        "subtype": "llm_call",
        "activity_id": ctx.get("call_type") or call_type,
        # agent_id uses the real Academy ID when available
        "agent_id": agent_id,
        "custom_metadata": {
            "llm_call_type": call_type,
            "hyp_id": ctx.get("hyp_id"),
        },
        "started_at": now,
        "ended_at": now,
        "status": "ERROR" if has_error else "FINISHED",
        "used": used,
        "generated": generated,
    }
    if parent_task_id is not None:
        task["parent_task_id"] = parent_task_id

    interceptor.intercept_task(task)


# ---------------------------------------------------------------------------
# Public plugin class
# ---------------------------------------------------------------------------

class FlowceptAcademyPlugin:
    """
    Generic FlowCept provenance plugin for any Academy-based application.

    Patches Academy's Runtime at the class level (zero agent code changes required).
    Produces a full provenance graph: campaign → workflow → agent sub-workflows →
    action tasks → LLM child tasks, with CPU/memory telemetry and node enrichment.

    Parameters
    ----------
    config : dict, optional
        Plugin configuration keys:
          enabled (bool, default True)
          workflow_name (str, default "academy-workflow")
          dump_path (str, optional) — JSONL path to dump the buffer on stop.
    llm_hook_register : callable, optional
        Called as llm_hook_register(_on_llm_call) on start to wire up LLM telemetry.
        Should match the signature used by your application's logging/hook module.
    llm_hook_unregister : callable, optional
        Called as llm_hook_unregister(_on_llm_call) on stop to remove the hook.
    """

    def __init__(
        self,
        config: dict | None = None,
        llm_hook_register: Any = None,
        llm_hook_unregister: Any = None,
    ) -> None:
        cfg = config or {}
        self._enabled: bool = cfg.get("enabled", False)
        self._workflow_name: str = cfg.get("workflow_name", "academy-workflow")
        self._dump_path: str | None = cfg.get("dump_path", None)
        self._perf_tracking: bool = cfg.get("performance_tracking", True)
        self._perf_csv: str | None = cfg.get("perf_csv", None)  # explicit override
        self._llm_hook_register = llm_hook_register
        self._llm_hook_unregister = llm_hook_unregister
        self._interceptor = AcademyInterceptor()
        self._started = False

    def start(self) -> "FlowceptAcademyPlugin":
        if not self._enabled or self._started:
            return self
        global _ACTIVE_INTERCEPTOR, _PROV_STATS
        try:
            _PROV_STATS = _ProvenanceStats() if self._perf_tracking else None
            self._interceptor.start(self._workflow_name)
            _ACTIVE_INTERCEPTOR = self._interceptor
            _install_runtime_patches()
            if self._llm_hook_register is not None:
                self._llm_hook_register(_on_llm_call)
            self._started = True
            wf_id = self._interceptor._workflow_id
            campaign_id = self._interceptor._campaign_id
            llm_status = "enabled" if self._llm_hook_register else "disabled (no hook provided)"
            print(
                f"[FlowceptAcademyPlugin] Started\n"
                f"  workflow_id : {wf_id}\n"
                f"  campaign_id : {campaign_id}\n"
                f"  Capturing   : agent lifecycle (sub-workflows), action dispatch "
                f"(telemetry + parent_task_id), loop events (group_id).\n"
                f"  LLM capture : {llm_status}\n"
                f"  Buffer → flowcept_buffer_{wf_id}.jsonl on stop.",
                flush=True,
            )
        except Exception as e:
            print(
                f"[FlowceptAcademyPlugin] WARNING: failed to start — {e!r}. "
                "Continuing without provenance capture.",
                flush=True,
            )
            _log.exception("FlowceptAcademyPlugin start failed")
            self._enabled = False
        return self

    def flush(self, dump_path: str | None = None) -> str | None:
        """
        Flush the in-memory FlowCept buffer to disk without stopping the plugin.

        Useful before generating reports mid-run so provenance records written
        so far are visible in the buffer file.

        Parameters
        ----------
        dump_path : str, optional
            Path to write the JSONL buffer.  Defaults to the configured
            dump_path, or an auto-named file based on workflow_id.

        Returns
        -------
        str or None
            The path the buffer was written to, or None if the plugin is not
            started or the flush failed.
        """
        if not self._started or self._interceptor._flowcept is None:
            return None
        path = dump_path or self._dump_path
        if path is None:
            wf_id = self._interceptor._workflow_id or 'unknown'
            path = f'flowcept_buffer_{wf_id}.jsonl'
        try:
            self._interceptor._flowcept.dump_buffer(path)
            print(f'[FlowceptAcademyPlugin] Buffer flushed → {path}', flush=True)
            return path
        except Exception as e:
            print(f'[FlowceptAcademyPlugin] Warning: flush failed — {e!r}', flush=True)
            return None

    def stop(self) -> None:
        if not self._started:
            return
        if self._llm_hook_unregister is not None:
            self._llm_hook_unregister(_on_llm_call)
        _uninstall_runtime_patches()
        try:
            self._interceptor.stop(dump_path=self._dump_path)
        except Exception as e:
            print(f"[FlowceptAcademyPlugin] Warning during stop: {e!r}", flush=True)
        self._started = False
        print("[FlowceptAcademyPlugin] Stopped. Provenance buffer flushed.", flush=True)
        if _PROV_STATS is not None:
            print(
                "\n[FlowceptAcademyPlugin] Provenance overhead report:\n"
                + _PROV_STATS.summary()
                + "\n  (N = event count; Total/Mean/Min/Max in ms/µs respectively)\n",
                flush=True,
            )
            # Derive CSV path: explicit config > same stem as dump_path > default
            wf_id = self._interceptor._workflow_id
            if self._perf_csv:
                csv_path = self._perf_csv
            elif self._dump_path:
                base = os.path.splitext(self._dump_path)[0]
                csv_path = f"{base}_perf.csv"
            else:
                csv_path = f"provenance_perf_{wf_id}.csv"
            try:
                _PROV_STATS.to_csv(csv_path, workflow_id=wf_id)
                print(
                    f"[FlowceptAcademyPlugin] Performance stats written to {csv_path}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[FlowceptAcademyPlugin] Warning: could not write perf CSV — {e!r}",
                    flush=True,
                )


# Keep old name working
FlowceptPlugin = FlowceptAcademyPlugin


# ---------------------------------------------------------------------------
# Serialisation helper
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
