# academy_coscientist/agents/poc_agent.py
"""
ProofOfConceptAgent — orchestrates PoC code generation, execution, and interpretation.

After all team cycles are complete, this agent:
  1. Polls the global tournament until at least one hypothesis is available.
  2. Selects the highest-scoring hypothesis.
  3. Uses the LLM to generate a self-contained Python PoC script.
  4. Delegates execution to DockerExecutorAgent (retry up to max_retries times).
  5. Uses the LLM to interpret the results.
  6. Saves a full report to poc_results/<hyp_id>/.

Wire the executor before starting:
    await poc_agent.set_executor(docker_executor_handle)

The agent runs its @loop once then stops (single-shot after start_delay).
"""
from __future__ import annotations

import asyncio
import os
import time
import traceback
from typing import Any

from academy.agent import Agent, action, loop

from academy_coscientist.utils import utils_llm
from academy_coscientist.utils.utils_logging import log_action, make_struct_logger

_POC_POLL_INTERVAL = 30.0    # seconds between tournament polls while waiting for hypotheses
_POC_MAX_WAIT      = 3600.0  # give up after 1 h if no hypotheses ever appear


class ProofOfConceptAgent(Agent):
    """
    Generates and executes a proof-of-concept for the top-ranked hypothesis.

    Parameters
    ----------
    start_delay : float
        Seconds to wait before the first (and only) PoC cycle.  Set this to
        at least ``max_cycles * interval * n_teams`` so the agent doesn't
        fire before all team cycles have completed.
    output_dir : str
        Directory where PoC reports are written (created if absent).
    """

    def __init__(
        self,
        start_delay: float = 300.0,
        output_dir: str = "poc_results",
        top_n: int = 3,
        max_retries: int = 3,
    ) -> None:
        super().__init__()
        self.logger = make_struct_logger("ProofOfConceptAgent")
        self._start_delay = float(start_delay)
        self._output_dir = str(output_dir)
        self._top_n = max(1, int(top_n))
        self._max_retries = max(1, int(max_retries))
        self._tournament = None
        self._topic: str = ""
        self._done = False
        self._team_coordinators: list = []
        self._executor = None  # DockerExecutorAgent handle; set via set_executor()
        # Docker config — mirrored here so _save_report can write accurate run.sh
        self._docker_image: str = "python:3.11-slim"
        self._docker_network: str = "none"
        self._docker_memory: str = "512m"
        self._docker_timeout: int = 90

    # ------------------------------------------------------------------
    # Configuration actions
    # ------------------------------------------------------------------

    @action
    async def set_tournament(self, tournament) -> None:
        self._tournament = tournament
        log_action(self.logger, "set_tournament", {"tournament": str(tournament)}, {"ok": True})

    @action
    async def set_topic(self, topic: str) -> None:
        self._topic = str(topic)
        log_action(self.logger, "set_topic", {"topic": topic}, {"ok": True})

    @action
    async def is_done(self) -> bool:
        """Returns True once the PoC pipeline has completed."""
        return self._done

    @action
    async def set_executor(self, executor) -> None:
        """Wire a DockerExecutorAgent handle for isolated code execution."""
        self._executor = executor
        log_action(self.logger, "set_executor", {"executor": str(executor)}, {"ok": True})

    @action
    async def set_docker_config(
        self,
        image: str,
        network: str,
        memory: str,
        timeout: int,
    ) -> None:
        """Mirror the DockerExecutorAgent's config so run.sh matches exactly."""
        self._docker_image = str(image)
        self._docker_network = str(network)
        self._docker_memory = str(memory)
        self._docker_timeout = int(timeout)
        log_action(self.logger, "set_docker_config",
                   {"image": image, "network": network, "memory": memory, "timeout": timeout},
                   {"ok": True})

    @action
    async def add_team_coordinator(self, coordinator) -> None:
        """Register a TeamCoordinatorAgent handle so the PoC agent can poll for completion."""
        self._team_coordinators.append(coordinator)
        log_action(self.logger, "add_team_coordinator",
                   {"total": len(self._team_coordinators)}, {"ok": True})

    # ------------------------------------------------------------------
    # Core PoC action
    # ------------------------------------------------------------------

    @action
    async def run_poc(self, top_n: int | None = None) -> list[dict[str, Any]]:
        """
        Full PoC pipeline for the top `top_n` hypotheses.

        For each hypothesis: generate code → execute (retry up to 3×) → interpret → save.
        Returns a list of report dicts.
        """
        if self._tournament is None:
            return [{"error": "no tournament configured"}]

        n = top_n if top_n is not None else self._top_n

        # --- 1. Fetch top hypotheses ---
        try:
            board = await self._tournament.get_leaderboard()
        except Exception as e:
            return [{"error": f"leaderboard fetch failed: {e!r}"}]

        if not board:
            return [{"error": "tournament is empty — no hypotheses to test"}]

        candidates = board[:n]
        print(
            f"\n[PoC] Running PoC for top {len(candidates)} hypothesis/es:\n"
            + "\n".join(
                f"  #{i+1} score={s:.3f}  {pl.get('title', hid)!r}"
                for i, (hid, s, pl) in enumerate(candidates)
            ),
            flush=True,
        )

        reports: list[dict[str, Any]] = []
        for rank, (hid, score, payload) in enumerate(candidates, 1):
            # Flatten payload; also surface nested meta fields for richer context
            meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}
            hypothesis = {
                "id": hid,
                "score": score,
                "title": payload.get("title", hid),
                "description": payload.get("description", ""),
                "rationale": meta.get("rationale", ""),
                "weaknesses": meta.get("weaknesses", []),
                "risks": meta.get("risks", []),
                "reasoning": meta.get("reasoning", ""),
                "validated_by_team": meta.get("validated_by_team", payload.get("validated_by_team", "")),
            }

            print(
                f"\n[PoC #{rank}] {hypothesis['title']!r} (score={score:.3f})",
                flush=True,
            )
            log_action(self.logger, "poc_start",
                       {"rank": rank, "hyp_id": hid, "score": score},
                       {"title": hypothesis["title"]})

            ctx = {"agent": "ProofOfConceptAgent", "hyp_id": hid,
                   "team": hypothesis["validated_by_team"], "rank": rank}

            # --- 2. Generate PoC code ---
            print(f"[PoC #{rank}] Generating proof-of-concept code...", flush=True)
            code = ""
            try:
                code = await utils_llm.generate_poc_code(
                    hypothesis=hypothesis,
                    topic=self._topic,
                    context=ctx,
                )
            except Exception as e:
                self.logger.error("poc_codegen_failed", extra={"error": repr(e), "rank": rank})
                err_report = {
                    "rank": rank, "hypothesis_id": hid,
                    "hypothesis_title": hypothesis["title"],
                    "hypothesis_score": score,
                    "topic": self._topic,
                    "code": "", "stdout": "", "stderr": repr(e), "returncode": -1,
                    "verdict": "CODEGEN_FAILED", "confidence": 0.0,
                    "interpretation": f"Code generation failed: {e!r}", "next_steps": "",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                self._save_report(f"{hid}_rank{rank}", "", err_report)
                reports.append(err_report)
                continue

            # --- 3. Execute, retrying on error up to max_retries times ---
            stdout, stderr, returncode = "", "", 1
            for attempt in range(1, self._max_retries + 1):
                print(
                    f"[PoC #{rank}] Executing script (attempt {attempt}/{self._max_retries})...",
                    flush=True,
                )
                try:
                    stdout, stderr, returncode = await self._run_code(code)
                except Exception as exec_err:
                    stderr = repr(exec_err)
                    returncode = -1
                    print(f"[PoC #{rank}] Executor error: {exec_err}", flush=True)
                print(
                    f"[PoC #{rank}] returncode={returncode}\n"
                    f"--- stdout ---\n{stdout[:2000]}\n"
                    f"--- stderr ---\n{stderr[:800] if stderr else '(none)'}\n",
                    flush=True,
                )
                if returncode == 0:
                    break
                if attempt < self._max_retries:
                    print(f"[PoC #{rank}] Failed — asking LLM to fix (attempt {attempt})...",
                          flush=True)
                    log_action(self.logger, "poc_fix_attempt",
                               {"hyp_id": hid, "attempt": attempt, "returncode": returncode},
                               {"stderr_snippet": stderr[:300]})
                    try:
                        code = await utils_llm.fix_poc_code(
                            code=code, stderr=stderr, stdout=stdout,
                            returncode=returncode,
                            context={**ctx, "fix_attempt": attempt},
                        )
                    except Exception as fix_err:
                        print(f"[PoC #{rank}] fix_poc_code failed: {fix_err}", flush=True)
                        break

            # --- 4. Interpret results ---
            print(f"[PoC #{rank}] Interpreting results...", flush=True)
            try:
                interpretation = await utils_llm.interpret_poc_results(
                    hypothesis=hypothesis, code=code,
                    stdout=stdout, stderr=stderr, returncode=returncode,
                    context=ctx,
                )
            except Exception as e:
                self.logger.error("poc_interpret_failed", extra={"error": repr(e), "rank": rank})
                interpretation = {
                    "verdict": "INCONCLUSIVE", "confidence": 0.0,
                    "interpretation": f"Interpretation failed: {e!r}", "next_steps": "",
                }

            # --- 5. Save report ---
            report = {
                "rank": rank,
                "hypothesis_id": hid,
                "hypothesis_title": hypothesis["title"],
                "hypothesis_score": score,
                "hypothesis_description": hypothesis["description"],
                "hypothesis_rationale": hypothesis["rationale"],
                "hypothesis_weaknesses": hypothesis["weaknesses"],
                "validated_by_team": hypothesis["validated_by_team"],
                "topic": self._topic,
                "code": code,
                "stdout": stdout,
                "stderr": stderr,
                "returncode": returncode,
                "verdict": interpretation.get("verdict"),
                "confidence": interpretation.get("confidence"),
                "interpretation": interpretation.get("interpretation"),
                "next_steps": interpretation.get("next_steps"),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self._save_report(f"{hid}_rank{rank}", code, report)

            verdict = interpretation.get("verdict")
            print(
                f"\n{'='*60}\n"
                f"  PoC #{rank} VERDICT: {verdict!r}  "
                f"(confidence={interpretation.get('confidence', 0):.2f})\n"
                f"  {interpretation.get('interpretation', '')[:200]}\n"
                f"{'='*60}\n",
                flush=True,
            )
            log_action(self.logger, "poc_complete",
                       {"hyp_id": hid, "rank": rank, "returncode": returncode},
                       {"verdict": verdict,
                        "confidence": interpretation.get("confidence")})
            reports.append(report)

            # --- 6. Feed back INCONCLUSIVE results to the last team for re-review ---
            if verdict == "INCONCLUSIVE" and self._team_coordinators:
                poc_feedback = {
                    "verdict": verdict,
                    "confidence": interpretation.get("confidence"),
                    "interpretation": interpretation.get("interpretation", ""),
                    "next_steps": interpretation.get("next_steps", ""),
                    "stdout_snippet": stdout[:500],
                    "stderr_snippet": stderr[:300],
                    "returncode": returncode,
                }
                coordinator = self._team_coordinators[-1]
                try:
                    result = await coordinator.resubmit_with_poc_feedback(hypothesis, poc_feedback)
                    log_action(
                        self.logger, "poc_feedback_sent",
                        {"hyp_id": hid, "rank": rank},
                        result,
                    )
                    print(
                        f"[PoC #{rank}] INCONCLUSIVE — resubmitted hyp={hid} "
                        f"to team coordinator for re-review.",
                        flush=True,
                    )
                except Exception as fb_err:
                    self.logger.warning(
                        "poc_feedback_failed",
                        extra={"hyp_id": hid, "error": repr(fb_err)},
                    )

        self._done = True
        return reports

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _run_code(self, code: str) -> tuple[str, str, int]:
        """Delegate code execution to DockerExecutorAgent."""
        if self._executor is None:
            return "", "No executor configured. Call set_executor() before running PoC.", 1
        result = await self._executor.run_code(code)
        return result["stdout"], result["stderr"], result["returncode"]

    def _save_report(self, hyp_id: str, code: str, report: dict[str, Any]) -> None:
        import json
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(hyp_id))
        out_dir = os.path.join(self._output_dir, safe_id)
        os.makedirs(out_dir, exist_ok=True)

        # --- core files ---
        with open(os.path.join(out_dir, "poc_code.py"), "w", encoding="utf-8") as f:
            f.write(code)
        with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # --- Dockerfile (standalone, copies poc_code.py into the image) ---
        title = report.get("hypothesis_title", hyp_id)
        dockerfile = (
            f"# Proof-of-concept experiment: {title}\n"
            f"# Auto-generated by ProofOfConceptAgent\n"
            f"FROM {self._docker_image}\n"
            f"WORKDIR /experiment\n"
            f"COPY poc_code.py .\n"
            f"CMD [\"python\", \"poc_code.py\"]\n"
        )
        with open(os.path.join(out_dir, "Dockerfile"), "w", encoding="utf-8") as f:
            f.write(dockerfile)

        # --- run.sh (reproduces the exact DockerExecutorAgent invocation via stdin) ---
        run_sh = (
            "#!/usr/bin/env bash\n"
            "# Reproduce the PoC execution exactly as run by DockerExecutorAgent\n"
            "# Usage:\n"
            "#   ./run.sh                # run via stdin (matches simulator behaviour)\n"
            "#   ./run.sh --build        # build a standalone image and run it\n"
            "set -euo pipefail\n"
            "\n"
            f'IMAGE="{self._docker_image}"\n'
            f'MEMORY="{self._docker_memory}"\n'
            f'NETWORK="{self._docker_network}"\n'
            f'TIMEOUT="{self._docker_timeout}"\n'
            "\n"
            'if [[ "${1:-}" == "--build" ]]; then\n'
            '    docker build -t poc-experiment .\n'
            '    docker run --rm \\\n'
            f'        --network "$NETWORK" \\\n'
            f'        --memory  "$MEMORY" \\\n'
            '        --cpus 1 \\\n'
            '        poc-experiment\n'
            "else\n"
            '    # stdin mode — no filesystem mounts, matches production behaviour\n'
            '    docker run --rm --interactive \\\n'
            f'        --network "$NETWORK" \\\n'
            f'        --memory  "$MEMORY" \\\n'
            '        --cpus 1 \\\n'
            f'        "$IMAGE" python - < poc_code.py\n'
            "fi\n"
        )
        run_sh_path = os.path.join(out_dir, "run.sh")
        with open(run_sh_path, "w", encoding="utf-8") as f:
            f.write(run_sh)
        os.chmod(run_sh_path, 0o755)

        print(f"[PoC] Report saved to {out_dir}/", flush=True)

    # ------------------------------------------------------------------
    # Autonomous loop — runs once after start_delay, then exits
    # ------------------------------------------------------------------

    @loop
    async def poc_loop(self, shutdown: asyncio.Event) -> None:
        """
        Wait for all team coordinators to finish their cycles, then run PoC on top 3.

        If no coordinator handles were registered (e.g. flat mode), falls back to
        the time-based `start_delay` wait, then polls the tournament for hypotheses.
        """
        # Brief startup wait so that add_team_coordinator wiring actions (which are
        # delivered asynchronously through the Academy message bus) have time to
        # arrive before we decide whether to use coordinator-polling or time-based wait.
        _WIRING_WAIT = 15.0
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=_WIRING_WAIT)
            return  # shutdown signalled during wiring wait
        except asyncio.TimeoutError:
            pass

        # --- Phase 1: wait for all team coordinators to finish ---
        _teams_known_done = False  # set to True when coordinators explicitly finish
        if self._team_coordinators:
            print(
                f"[PoC] Polling {len(self._team_coordinators)} team coordinator(s) "
                "for completion...",
                flush=True,
            )
            waited = 0.0
            while not shutdown.is_set():
                done_flags = []
                for c in self._team_coordinators:
                    try:
                        flag = await asyncio.wait_for(c.is_done(), timeout=10.0)
                        done_flags.append(bool(flag))
                    except asyncio.TimeoutError:
                        # Agent not responding — it has likely finished and been cleaned up
                        done_flags.append(True)
                    except Exception as e:
                        self.logger.warning("coordinator_poll_failed", extra={"error": repr(e)})
                        done_flags.append(True)  # treat unreachable agent as done
                if done_flags and all(done_flags):
                    print("[PoC] All teams finished — proceeding to PoC.", flush=True)
                    _teams_known_done = True
                    break
                if waited >= _POC_MAX_WAIT:
                    print("[PoC] Timed out waiting for teams. Proceeding anyway.", flush=True)
                    break
                n_done = sum(1 for f in done_flags if f)
                print(
                    f"[PoC] Teams done: {n_done}/{len(self._team_coordinators)} "
                    f"— waiting {_POC_POLL_INTERVAL:.0f}s (total: {waited:.0f}s)...",
                    flush=True,
                )
                try:
                    await asyncio.wait_for(shutdown.wait(), timeout=_POC_POLL_INTERVAL)
                    return
                except asyncio.TimeoutError:
                    waited += _POC_POLL_INTERVAL
        elif self._start_delay > 0:
            # Fallback: fixed time delay
            print(
                f"[PoC] Waiting {self._start_delay:.0f}s for teams to complete cycles...",
                flush=True,
            )
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=self._start_delay)
                return
            except asyncio.TimeoutError:
                pass

        if shutdown.is_set():
            return

        # --- Phase 2: poll until at least one hypothesis is in the tournament ---
        # When teams are known to be done, no new hypotheses will arrive — cap the wait
        # at 2 poll intervals (60 s) rather than the full 1-hour _POC_MAX_WAIT.
        phase2_max = _POC_POLL_INTERVAL * 2 if _teams_known_done else _POC_MAX_WAIT
        waited = 0.0
        while not shutdown.is_set():
            try:
                board = await self._tournament.get_leaderboard() if self._tournament else []
            except Exception:
                board = []
            if board:
                break
            if waited >= phase2_max:
                print(
                    "[PoC] Tournament still empty after waiting"
                    f" {waited:.0f}s — no hypotheses to test. Exiting.",
                    flush=True,
                )
                self._done = True
                return
            print(
                f"[PoC] Tournament empty — waiting {_POC_POLL_INTERVAL:.0f}s "
                f"(total waited: {waited:.0f}s / max {phase2_max:.0f}s)...",
                flush=True,
            )
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=_POC_POLL_INTERVAL)
                return
            except asyncio.TimeoutError:
                waited += _POC_POLL_INTERVAL

        if shutdown.is_set():
            return

        try:
            await self.run_poc()
        except Exception as e:
            print(f"[PoC] ERROR: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
            self.logger.error("poc_loop_error", extra={"error": repr(e)})
        finally:
            # Always mark done so _poc_watcher / simulator can detect completion
            # even when run_poc() raises or returns early (empty tournament).
            self._done = True
