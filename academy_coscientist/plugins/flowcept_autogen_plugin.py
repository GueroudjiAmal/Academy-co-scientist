# academy_coscientist/plugins/flowcept_autogen_plugin.py
"""
FlowCept provenance plugin for AutoGen (autogen_agentchat 0.7+) workflows.

Captures provenance for AutoGen team runs by consuming the ``run_stream()``
generator and recording every message as a FlowCept TaskObject, with the
overall team run as a WorkflowObject.

Provenance hierarchy produced:
  WorkflowObject  (one per team.run() call)
    └─ TaskObject  subtype=autogen_run      activity_id=<team_name>
         └─ TaskObject  subtype=autogen_message  activity_id=<agent_name>
              (one record per message yielded by run_stream)

Key FlowCept fields:
  task_id          — uuid per event
  workflow_id      — global shared workflow id (setdefault pattern)
  campaign_id      — from Flowcept.campaign_id
  parent_task_id   — messages are children of the enclosing run task
  group_id         — all tasks within one team.run() share a group_id
  activity_id      — team name | agent name | model name
  subtype          — autogen_run | autogen_message | autogen_result
  used / generated — message content, source, recipient
  status           — FINISHED | ERROR

Usage
-----
Standalone::

    plugin = FlowceptAutoGenPlugin(config={"workflow_name": "my-team", "dump_path": "out.jsonl"})
    plugin.start()
    result = asyncio.run(plugin.run_team(team, "Do something useful"))
    plugin.stop()

Or as a context manager::

    with FlowceptAutoGenPlugin(config={"workflow_name": "my-team"}) as plugin:
        result = asyncio.run(plugin.run_team(team, "task"))

Shared with Academy plugin::

    academy_plugin = FlowceptAcademyPlugin(config={...}).start()
    autogen_plugin = FlowceptAutoGenPlugin.from_academy_plugin(academy_plugin)
    result = asyncio.run(autogen_plugin.run_team(team, "task"))
    academy_plugin.stop()   # flushes the shared buffer
"""
from __future__ import annotations

import os
os.environ.setdefault("FLOWCEPT_USE_DEFAULT", "1")

import asyncio
import time
import uuid
import threading
import logging
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provenance overhead timer
# ---------------------------------------------------------------------------

class _ProvenanceStats:
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
# Standalone interceptor wrapper
# ---------------------------------------------------------------------------

class _AutoGenInterceptor:
    """Standalone FlowCept interceptor for use without an Academy plugin."""

    def __init__(self) -> None:
        self._interceptor = None
        self._flowcept    = None
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

    def send_team_workflow(self, team_name: str, group_id: str) -> str:
        if self._interceptor is None:
            return str(uuid.uuid4())
        from flowcept.commons.flowcept_dataclasses.workflow_object import WorkflowObject
        wf = WorkflowObject()
        wf.workflow_id = str(uuid.uuid4())
        wf.name = team_name
        wf.campaign_id = self._campaign_id
        wf.parent_workflow_id = self._workflow_id
        wf.custom_metadata = {"group_id": group_id, "framework": "autogen"}
        self._interceptor.send_workflow_message(wf)
        return wf.workflow_id

    # Accept the same name used by AcademyInterceptor so shared interceptors work
    send_graph_workflow = send_team_workflow

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
# Provenance stream runner
# ---------------------------------------------------------------------------

def _safe_clip(obj: Any, depth: int = 0) -> Any:
    if depth > 6:
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_clip(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_clip(v, depth + 1) for v in obj]
    # AutoGen message objects
    try:
        if hasattr(obj, "model_dump"):
            return _safe_clip(obj.model_dump(), depth + 1)
    except Exception:
        pass
    try:
        if hasattr(obj, "__dict__"):
            return _safe_clip(vars(obj), depth + 1)
    except Exception:
        pass
    return repr(obj)


async def _run_with_provenance(
    team: Any,
    task: str,
    interceptor: Any,
    stats: _ProvenanceStats | None,
    team_name: str = "autogen_team",
    source_agent_id: str | None = None,
) -> Any:
    """
    Consume team.run_stream() and emit a FlowCept provenance record for each
    message plus one overall run record.

    Returns the final TaskResult.
    """
    from autogen_agentchat.base import TaskResult
    from autogen_agentchat.messages import BaseChatMessage, BaseAgentEvent
    from autogen_core import CancellationToken

    group_id    = str(uuid.uuid4())
    run_task_id = str(uuid.uuid4())
    run_start   = time.time()

    # Emit a sub-WorkflowObject for this team run
    interceptor.send_graph_workflow(team_name, group_id)

    custom_meta: dict = {"team_name": team_name, "framework": "autogen"}
    if source_agent_id:
        custom_meta["source_agent_id"] = source_agent_id

    messages_captured: list[dict] = []
    final_result: Any = None

    t_stream_start = time.perf_counter()

    try:
        stream = team.run_stream(task=task, cancellation_token=CancellationToken())
        async for item in stream:
            t0 = time.perf_counter()
            if isinstance(item, TaskResult):
                final_result = item
            else:
                # Each item is a BaseChatMessage or BaseAgentEvent
                msg_task_id = str(uuid.uuid4())
                source      = getattr(item, "source", None) or "unknown"
                content     = getattr(item, "content", None)
                msg_type    = type(item).__name__

                # Serialize content
                if isinstance(content, str):
                    content_clip = content
                else:
                    content_clip = _safe_clip(content)

                msg_record: dict = {
                    "task_id":      msg_task_id,
                    "subtype":      "autogen_message",
                    "activity_id":  source,
                    "group_id":     group_id,
                    "parent_task_id": run_task_id,
                    "started_at":   time.time(),
                    "ended_at":     time.time(),
                    "status":       "FINISHED",
                    "used":         {"task": task, "agent": source},
                    "generated":    {"content": content_clip, "message_type": msg_type},
                    "custom_metadata": {
                        "agent_name":   source,
                        "message_type": msg_type,
                        "framework":    "autogen",
                    },
                }
                interceptor.intercept_task(msg_record)
                messages_captured.append({
                    "source":  source,
                    "content": content_clip[:200] if isinstance(content_clip, str) else content_clip,
                })
                if stats is not None:
                    stats.record("message_intercept", time.perf_counter() - t0)

    except Exception as exc:
        # Emit the run record as ERROR then re-raise
        run_task: dict = {
            "task_id":     run_task_id,
            "subtype":     "autogen_run",
            "activity_id": team_name,
            "group_id":    group_id,
            "started_at":  run_start,
            "ended_at":    time.time(),
            "status":      "ERROR",
            "used":        {"task": task},
            "generated":   {"messages": messages_captured},
            "stderr":      str(exc),
            "custom_metadata": custom_meta,
        }
        interceptor.intercept_task(run_task)
        raise

    if stats is not None:
        stats.record("stream_total", time.perf_counter() - t_stream_start)

    # Emit the overall run record (ONE complete record with inputs + outputs)
    summary = _safe_clip(getattr(final_result, "stop_reason", None) or "completed")
    msg_count = len(messages_captured)
    last_msg = messages_captured[-1]["content"] if messages_captured else ""

    run_task = {
        "task_id":     run_task_id,
        "subtype":     "autogen_run",
        "activity_id": team_name,
        "group_id":    group_id,
        "started_at":  run_start,
        "ended_at":    time.time(),
        "status":      "FINISHED",
        "used":        {"task": task},
        "generated":   {
            "stop_reason":   summary,
            "message_count": msg_count,
            "last_message":  last_msg,
            "messages":      messages_captured,
        },
        "custom_metadata": custom_meta,
    }
    interceptor.intercept_task(run_task)

    return final_result


# ---------------------------------------------------------------------------
# Public plugin class
# ---------------------------------------------------------------------------

class FlowceptAutoGenPlugin:
    """
    FlowCept provenance plugin for AutoGen (autogen_agentchat 0.7+) team runs.

    Wraps ``team.run_stream()`` to capture provenance without patching the
    AutoGen library.  Each message yielded by the stream becomes a TaskObject
    child of the overall run TaskObject.

    Parameters
    ----------
    config : dict, optional
        Plugin configuration keys:
          enabled              (bool, default True)
          workflow_name        (str, default "autogen-workflow")
          dump_path            (str, optional) — JSONL path written on stop.
          performance_tracking (bool, default True)
          perf_csv             (str, optional) — explicit path for timing CSV.

    Usage
    -----
    Standalone::

        plugin = FlowceptAutoGenPlugin(config={"workflow_name": "my-team"})
        plugin.start()
        result = asyncio.run(plugin.run_team(team, "Solve the problem"))
        plugin.stop()

    Shared with Academy plugin::

        academy_plugin = FlowceptAcademyPlugin(config={...}).start()
        autogen_plugin = FlowceptAutoGenPlugin.from_academy_plugin(academy_plugin)
        result = asyncio.run(autogen_plugin.run_team(team, "task"))
        academy_plugin.stop()
    """

    def __init__(self, config: dict | None = None, _shared_interceptor=None) -> None:
        cfg = config or {}
        self._enabled:        bool       = cfg.get("enabled", True)
        self._workflow_name:  str        = cfg.get("workflow_name", "autogen-workflow")
        self._dump_path:      str | None  = cfg.get("dump_path", None)
        self._perf_tracking:  bool       = cfg.get("performance_tracking", True)
        self._perf_csv:       str | None  = cfg.get("perf_csv", None)
        self._shared_interceptor = _shared_interceptor
        self._interceptor    = _shared_interceptor or _AutoGenInterceptor()
        self._owns_interceptor: bool = _shared_interceptor is None
        self._stats: _ProvenanceStats | None = None
        self._started = False

    @classmethod
    def from_academy_plugin(
        cls,
        academy_plugin: Any,
        config: dict | None = None,
    ) -> "FlowceptAutoGenPlugin":
        """
        Create an AutoGen plugin that shares the buffer of a running
        FlowceptAcademyPlugin.
        """
        interceptor = academy_plugin._interceptor
        inst = cls(config=config, _shared_interceptor=interceptor)
        inst._started = True

        if inst._dump_path is None and getattr(academy_plugin, "_dump_path", None):
            base = os.path.splitext(academy_plugin._dump_path)[0]
            inst._perf_csv = inst._perf_csv or f"{base}_autogen_perf.csv"

        inst._stats = _ProvenanceStats() if (config or {}).get("performance_tracking", True) else None
        return inst

    def start(self) -> "FlowceptAutoGenPlugin":
        if not self._enabled or self._started:
            return self
        if not self._owns_interceptor:
            return self
        try:
            self._stats = _ProvenanceStats() if self._perf_tracking else None
            self._interceptor.start(self._workflow_name)
            self._started = True
            wf_id = self._interceptor._workflow_id
            print(
                f"[FlowceptAutoGenPlugin] Started\n"
                f"  workflow_id : {wf_id}\n"
                f"  campaign_id : {self._interceptor._campaign_id}\n"
                f"  Capturing   : team run (sub-workflow), messages (child tasks).",
                flush=True,
            )
        except Exception as e:
            print(
                f"[FlowceptAutoGenPlugin] WARNING: failed to start — {e!r}. "
                "Continuing without provenance capture.",
                flush=True,
            )
            _log.exception("FlowceptAutoGenPlugin start failed")
            self._enabled = False
        return self

    async def run_team(
        self,
        team: Any,
        task: str,
        team_name: str | None = None,
        source_agent_id: str | None = None,
    ) -> Any:
        """
        Run a team and capture provenance for the entire conversation.

        Parameters
        ----------
        team : RoundRobinGroupChat | SelectorGroupChat | any team with run_stream
            The AutoGen team to run.
        task : str
            The task / initial message to send to the team.
        team_name : str, optional
            Human-readable name for this run in provenance records.
            Defaults to the team's ``name`` attribute or "autogen_team".
        source_agent_id : str, optional
            ID of an upstream agent (e.g. Academy AgentId) that produced the
            input data.  Stored in custom_metadata for cross-framework linkage.

        Returns
        -------
        TaskResult
            The final result returned by AutoGen.
        """
        name = team_name or getattr(team, "name", None) or "autogen_team"
        if not self._started:
            # plugin not started — run the team without provenance
            from autogen_agentchat.base import TaskResult
            from autogen_core import CancellationToken
            results = []
            async for item in team.run_stream(task=task, cancellation_token=CancellationToken()):
                if isinstance(item, TaskResult):
                    results.append(item)
            return results[-1] if results else None

        return await _run_with_provenance(
            team=team,
            task=task,
            interceptor=self._interceptor,
            stats=self._stats,
            team_name=name,
            source_agent_id=source_agent_id,
        )

    def stop(self) -> None:
        if not self._started:
            return
        if not self._owns_interceptor:
            self._started = False
            print(
                "[FlowceptAutoGenPlugin] Detached from shared buffer "
                "(flushed by the owning plugin).",
                flush=True,
            )
            self._maybe_write_perf_csv()
            return
        try:
            self._interceptor.stop(dump_path=self._dump_path)
        except Exception as e:
            print(f"[FlowceptAutoGenPlugin] Warning during stop: {e!r}", flush=True)
        self._started = False
        print("[FlowceptAutoGenPlugin] Stopped. Provenance buffer flushed.", flush=True)
        self._maybe_write_perf_csv()

    def _maybe_write_perf_csv(self) -> None:
        if self._stats is None:
            return
        print(
            "\n[FlowceptAutoGenPlugin] Provenance overhead report:\n"
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
            csv_path = f"autogen_provenance_perf_{wf_id}.csv"
        try:
            self._stats.to_csv(csv_path, workflow_id=wf_id)
            print(f"[FlowceptAutoGenPlugin] Performance stats → {csv_path}", flush=True)
        except Exception as e:
            print(f"[FlowceptAutoGenPlugin] Warning: could not write perf CSV — {e!r}", flush=True)

    def __enter__(self) -> "FlowceptAutoGenPlugin":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()
