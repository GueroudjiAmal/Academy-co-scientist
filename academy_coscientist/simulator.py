# academy_coscientist/simulator.py
"""
Autonomous multi-agent simulator built on Academy.

Each agent category runs in its own @loop, interacting continuously
through shared TournamentAgent state.  Everything is driven by a YAML
config so you can tune counts, intervals, topics, and backend without
touching Python.

Usage
-----
    python -m academy_coscientist.simulator --config configs/simulator_config.yaml

    # Override topic or agent counts at the CLI
    python -m academy_coscientist.simulator \\
        --config configs/simulator_config.yaml \\
        --topic "quantum error correction" \\
        --generation-agents 3 \\
        --review-agents 4
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import multiprocessing
import os
import signal
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

import yaml

from academy.manager import Manager
from academy.logging import init_logging

from academy_coscientist.agents.generation_agent import HypothesisGenerationAgent
from academy_coscientist.agents.review_agent import ReviewAgent
from academy_coscientist.agents.tournament_agent import TournamentAgent
from academy_coscientist.agents.research_vector_agent import ResearchVectorDBAgent
from academy_coscientist.agents.paper_harvester_agent import PaperHarvesterAgent
from academy_coscientist.agents.refiner_agent import HypothesisRefinerAgent
from academy_coscientist.agents.team_coordinator_agent import TeamCoordinatorAgent
from academy_coscientist.agents.poc_agent import ProofOfConceptAgent
from academy_coscientist.agents.docker_executor_agent import DockerExecutorAgent
from academy_coscientist.agents.provenance_report_agent import ProvenancePDFReportAgent
from academy_coscientist.utils.config import load_config
from academy_coscientist.utils.utils_logging import init_run_context, make_struct_logger
import academy_coscientist.utils.utils_logging as _llm_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_simulator_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return raw


def _get(cfg: dict, *keys, default=None):
    """Nested dict getter with a default."""
    node = cfg
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, default)
        if node is None:
            return default
    return node


# ---------------------------------------------------------------------------
# Exchange / executor factory
# ---------------------------------------------------------------------------

def _build_exchange(sim_cfg: dict):
    """Return an Academy ExchangeFactory based on config."""
    exchange_type = _get(sim_cfg, "exchange", "type", default="local")
    if exchange_type == "redis":
        from academy.exchange import RedisExchangeFactory
        host = _get(sim_cfg, "exchange", "redis_host", default="localhost")
        port = _get(sim_cfg, "exchange", "redis_port", default=6379)
        return RedisExchangeFactory(host, port=port)
    elif exchange_type == "http":
        from academy.exchange.cloud.client import HttpExchangeFactory
        address = _get(sim_cfg, "exchange", "address", default="https://exchange.academy-agents.org")
        auth = _get(sim_cfg, "exchange", "auth_method", default="globus")
        return HttpExchangeFactory(address, auth_method=auth)
    else:
        from academy.exchange import LocalExchangeFactory
        return LocalExchangeFactory()


# def _worker_init() -> None:
#     """ProcessPoolExecutor subprocess initializer.
#
#     Re-applies Flowcept Runtime patches inside each worker process using the
#     workflow/campaign IDs that the main process published via env vars.
#     Each worker gets its own BaseInterceptor that writes task records to the
#     shared buffer under the same workflow_id, keeping all provenance linked.
#     """
#     init_logging()
#     workflow_id = os.environ.get("FLOWCEPT_WORKFLOW_ID")
#     campaign_id = os.environ.get("FLOWCEPT_CAMPAIGN_ID") or None
#     if not workflow_id:
#         return
#     try:
#         import flowcept.agents.academy.academy_plugin as _ap_mod
#         from flowcept.agents.academy.academy_plugin import (
#             AcademyInterceptor,
#             _install_runtime_patches,
#         )
#         from flowcept.flowceptor.adapters.base_interceptor import BaseInterceptor
#
#         interceptor = AcademyInterceptor()
#         interceptor._workflow_id = workflow_id
#         interceptor._campaign_id = campaign_id
#         base = BaseInterceptor(kind="academy")
#         base.start(bundle_exec_id=workflow_id, check_safe_stops=False)
#         interceptor._interceptor = base
#         _ap_mod._ACTIVE_INTERCEPTOR = interceptor
#         _install_runtime_patches()
#     except Exception as _e:
#         print(f"[worker] Flowcept subprocess init failed: {_e!r}", flush=True)


def _build_single_executor(cfg: dict):
    """Build one executor from a flat executor-config dict."""
    exec_type = str(cfg.get("type", "thread")).lower()
    max_workers = int(cfg.get("max_workers", 16))
    if exec_type == "process":
        mp_context = multiprocessing.get_context("spawn")
        return ProcessPoolExecutor(
            max_workers=max_workers,
            # initializer=_worker_init,
            mp_context=mp_context,
        )
    elif exec_type == "globus":
        from globus_compute_sdk import Executor as GlobusComputeExecutor
        endpoint_id = cfg.get("endpoint_id")
        if not endpoint_id:
            raise RuntimeError(
                "endpoint_id is required for a globus executor. "
                "Set it under the executor entry in your config."
            )
        user_endpoint_config = cfg.get("user_endpoint_config") or {}
        return GlobusComputeExecutor(
            endpoint_id=endpoint_id,
            user_endpoint_config=user_endpoint_config or None,
        )
    elif exec_type in ("local", "event_loop"):
        # None tells Academy Manager to run the agent in the event loop
        return None
    else:
        return ThreadPoolExecutor(max_workers=max_workers)


def _build_executor(sim_cfg: dict):
    """
    Return ``(executors, default_executor_name)`` for the Academy Manager.

    Three modes driven by ``executor.type`` in config:

    * ``"thread"`` / ``"process"`` / ``"globus"`` — single executor returned as
      a plain object; Academy wraps it as ``{"default": executor}`` internally.
      Returns ``(executor, None)`` so the caller passes it directly.

    * ``"federated"`` — returns a named dict and the default executor name so
      the caller can pass ``executor="aurora"`` on individual ``manager.launch()``
      calls to pin specific agents to specific endpoints.
      Config shape::

          executor:
            type: federated
            default: "local"
            executors:
              local:
                type: thread
                max_workers: 16
              aurora:
                type: globus
                endpoint_id: "xxxx-..."
    """
    exec_type = _get(sim_cfg, "executor", "type", default="thread")

    if exec_type == "federated":
        named_cfgs = _get(sim_cfg, "executor", "executors", default={}) or {}
        default_name = str(_get(sim_cfg, "executor", "default", default="local"))
        executors: dict = {}
        for name, cfg in named_cfgs.items():
            executors[name] = _build_single_executor(cfg or {})
        if not executors:
            raise RuntimeError("executor.type=federated requires at least one entry under executor.executors")
        if default_name not in executors:
            raise RuntimeError(
                f"executor.default={default_name!r} is not among the defined executors: "
                + ", ".join(executors)
            )
        return executors, default_name

    # Single executor — return as-is; Manager wraps it under "default" key.
    single = _build_single_executor(_get(sim_cfg, "executor", default={}) or {})
    return single, None


# ---------------------------------------------------------------------------
# Agent launchers
# ---------------------------------------------------------------------------

async def _launch_generation_agents(
    manager,
    sim_cfg: dict,
    topics: list[str],
    tournament,
    vectordb,
    stagger: float,
) -> list:
    count = _get(sim_cfg, "agents", "generation", default=1)
    interval = _get(sim_cfg, "intervals", "generation", default=60.0)
    n_hyps = _get(sim_cfg, "hypotheses_per_cycle", default=3)
    rag_top_k = int(_get(sim_cfg, "rag_top_k", default=5))

    agents = []
    for i in range(count):
        topic = topics[i % len(topics)]
        agent = await manager.launch(
            HypothesisGenerationAgent,
            args=(True, float(interval), float(i) * stagger, int(n_hyps), rag_top_k),
        )
        await agent.ping(timeout=30.0)
        await agent.set_topic(topic)
        await agent.set_tournament(tournament)
        if vectordb:
            await agent.set_vectordb(vectordb)
        agents.append(agent)
        logger.info("Launched HypothesisGenerationAgent #%d (topic=%r)", i, topic)
    return agents


async def _launch_review_agents(
    manager,
    sim_cfg: dict,
    tournament,
    stagger: float,
) -> list:
    count = _get(sim_cfg, "agents", "review", default=2)
    interval = _get(sim_cfg, "intervals", "review", default=45.0)
    rag_top_k = int(_get(sim_cfg, "rag_top_k", default=5))

    agents = []
    for i in range(count):
        agent = await manager.launch(
            ReviewAgent,
            args=(f"reviewer_{i}", True, float(interval), float(i) * stagger, rag_top_k),
        )
        await agent.ping(timeout=30.0)
        await agent.set_tournament(tournament)
        agents.append(agent)
        logger.info("Launched ReviewAgent #%d", i)
    return agents


async def _launch_tournament_agents(
    manager,
    sim_cfg: dict,
    stagger: float,
) -> list:
    count = _get(sim_cfg, "agents", "tournament", default=1)
    interval = _get(sim_cfg, "intervals", "tournament", default=30.0)
    max_cycles_cfg = _get(sim_cfg, "teams", "max_cycles", default=None)
    max_cycles = int(max_cycles_cfg) if max_cycles_cfg is not None else None

    agents = []
    for i in range(count):
        agent = await manager.launch(
            TournamentAgent,
            args=(True, float(interval), float(i) * stagger, max_cycles),
        )
        await agent.ping(timeout=30.0)
        agents.append(agent)
        logger.info("Launched TournamentAgent #%d (max_cycles=%s)", i, max_cycles)
    return agents


async def _launch_harvester_agents(
    manager,
    sim_cfg: dict,
    topics: list[str],
    vectordb,
    stagger: float,
    remote_exec: str | None = None,
) -> list:
    """
    Launch PaperHarvesterAgent instances.

    Each harvester:
    - downloads the most-cited open-access papers for its topic slice
    - saves PDFs (and .txt abstracts) to research_papers/
    - rebuilds the vector index after each harvest run
    - repeats on the configured weekly schedule
    """
    count = _get(sim_cfg, "agents", "harvester", default=1)
    if count <= 0:
        return []

    h_cfg = _get(sim_cfg, "harvester", default={}) or {}
    papers_per_topic   = int(_get(h_cfg, "papers_per_topic",   default=20))
    loop_interval      = float(_get(h_cfg, "loop_interval",    default=7 * 24 * 3600.0))
    start_delay        = float(_get(h_cfg, "loop_start_delay", default=0.0))
    download_pdfs      = bool(_get(h_cfg, "download_pdfs",     default=True))
    min_citations      = int(_get(h_cfg, "min_citation_count", default=0))

    agents = []
    for i in range(count):
        # Distribute topics evenly across harvesters
        chunk = max(1, len(topics) // count)
        topic_slice = topics[i * chunk: (i + 1) * chunk] or topics
        delay = start_delay + float(i) * stagger

        launch_kw = {"executor": remote_exec} if remote_exec else {}
        agent = await manager.launch(
            PaperHarvesterAgent,
            args=(
                topic_slice,
                papers_per_topic,
                None,           # research_papers_dir → resolved from config paths
                download_pdfs,
                min_citations,
                True,           # loop_enabled
                loop_interval,
                delay,
            ),
            **launch_kw,
        )
        await agent.ping(timeout=60.0)
        if vectordb:
            await agent.set_vectordb(vectordb)
        agents.append(agent)
        logger.info(
            "Launched PaperHarvesterAgent #%d (topics=%r, interval_h=%.1f)",
            i,
            topic_slice,
            loop_interval / 3600,
        )
    return agents


# ---------------------------------------------------------------------------
# Team-based launch
# ---------------------------------------------------------------------------

async def _launch_teams(
    manager,
    sim_cfg: dict,
    poc_cfg: dict,
    topics: list[str],
    global_tournament,
    vectordb,
    stagger: float,
) -> list:
    """
    Launch co-scientist teams.

    Each team = 1 generator + 1 refiner + N reviewers + 1 TeamCoordinator.

    Teams are chained: Team 1 generates from scratch and forwards validated
    hypotheses to Team 2's staging tournament, which Team 2 then further
    refines to a higher standard before pushing to the global tournament.

    If only one team is configured, it pushes directly to the global tournament.

    Config keys (under simulator.teams):
      count             : number of teams (default 2)
      reviewers_per_team: number of ReviewAgents per team (default 2)
      hypotheses_per_cycle: hypotheses generated/pulled per team loop cycle
      score_threshold   : score needed for a hypothesis to pass team review
      min_pass_score    : score floor — below this, hypothesis is discarded
      max_refinement_rounds: max refine-then-re-review iterations per hypothesis
      interval          : seconds between team loop cycles
    """
    teams_cfg = _get(sim_cfg, "teams", default={}) or {}
    n_teams = int(_get(teams_cfg, "count", default=2))
    n_reviewers = int(_get(teams_cfg, "reviewers_per_team", default=2))
    n_hyps = int(_get(teams_cfg, "hypotheses_per_cycle", default=3))
    score_threshold = float(_get(teams_cfg, "score_threshold", default=0.65))
    min_pass_score = float(_get(teams_cfg, "min_pass_score", default=0.40))
    max_rounds = int(_get(teams_cfg, "max_refinement_rounds", default=3))
    interval = float(_get(teams_cfg, "interval", default=120.0))
    max_cycles_cfg = _get(teams_cfg, "max_cycles", default=None)
    max_cycles = int(max_cycles_cfg) if max_cycles_cfg is not None else None
    rag_top_k = int(_get(sim_cfg, "rag_top_k", default=0))

    all_agents: list = []
    coordinators: list = []  # collected separately for PoC agent wiring

    # Each team's outgoing: team[i] → staging[i+1] → ... → global_tournament
    # Create staging tournaments for intermediate teams (last team → global).
    staging_tournaments: list = [None] * n_teams
    for i in range(n_teams - 1):
        staging = await manager.launch(TournamentAgent, args=(False,))
        await staging.ping(timeout=30.0)
        staging_tournaments[i] = staging  # team[i+1] reads from here
        all_agents.append(staging)
        logger.info("Launched staging TournamentAgent for team %d → team %d", i, i + 1)

    for team_idx in range(n_teams):
        topic = topics[team_idx % len(topics)]
        team_name = f"team_{team_idx}"

        # Raise score threshold for each successive team (last team is most strict)
        team_threshold = min(0.95, score_threshold + team_idx * 0.10)
        team_stagger = float(team_idx) * stagger * 2

        # Generator (first team only; downstream teams pull from staging)
        generator = None
        if team_idx == 0:
            generator = await manager.launch(
                HypothesisGenerationAgent,
                args=(False, interval, team_stagger, n_hyps, rag_top_k),
            )
            await generator.ping(timeout=30.0)
            await generator.set_topic(topic)
            await generator.set_team_name(team_name)
            if vectordb:
                await generator.set_vectordb(vectordb)
            all_agents.append(generator)
            logger.info("Launched generator for %s (topic=%r)", team_name, topic)

        # Refiner
        refiner = await manager.launch(HypothesisRefinerAgent)
        await refiner.ping(timeout=30.0)
        await refiner.set_topic(topic)
        await refiner.set_team_name(team_name)
        all_agents.append(refiner)
        logger.info("Launched refiner for %s", team_name)

        # Reviewers
        reviewers = []
        for r_idx in range(n_reviewers):
            reviewer = await manager.launch(
                ReviewAgent,
                args=(f"{team_name}_reviewer_{r_idx}", False, interval, 0.0, rag_top_k),
            )
            await reviewer.ping(timeout=30.0)
            await reviewer.set_team_name(team_name)
            reviewers.append(reviewer)
            all_agents.append(reviewer)
        logger.info("Launched %d reviewer(s) for %s", n_reviewers, team_name)

        # Determine incoming (None for first team) and outgoing.
        # ALL teams push validated hypotheses to the global tournament so the
        # leaderboard is populated immediately.  Non-final teams additionally
        # push to a staging tournament so the next team can pick them up and
        # refine them further (chain_outgoing).
        incoming = staging_tournaments[team_idx - 1] if team_idx > 0 else None
        is_final = team_idx == n_teams - 1
        chain_to = None if is_final else staging_tournaments[team_idx]

        # TeamCoordinator
        coordinator = await manager.launch(
            TeamCoordinatorAgent,
            args=(
                team_name,
                team_threshold,
                min_pass_score,
                max_rounds,
                n_hyps,
                interval,
                team_stagger,
                max_cycles,
            ),
        )
        await coordinator.ping(timeout=30.0)
        await coordinator.set_topic(topic)
        if generator:
            await coordinator.set_generator(generator)
        await coordinator.set_refiner(refiner)
        for reviewer in reviewers:
            await coordinator.add_reviewer(reviewer)
        if incoming is not None:
            await coordinator.set_incoming(incoming)
        # Every team's primary destination is the global tournament.
        await coordinator.set_outgoing(global_tournament)
        # Non-final teams also chain into their downstream staging tournament.
        if chain_to is not None:
            await coordinator.set_chain_outgoing(chain_to)

        all_agents.append(coordinator)
        coordinators.append(coordinator)
        logger.info(
            "Launched TeamCoordinatorAgent '%s' (threshold=%.2f, max_rounds=%d, "
            "incoming=%s, chain_to=%s)",
            team_name,
            team_threshold,
            max_rounds,
            "staging" if incoming else "None",
            "staging" if chain_to else "None (final)",
        )

    # --- Proof-of-concept agent -------------------------------------------
    # Skip entirely when poc.enabled is explicitly false.
    if not poc_cfg.get("enabled", True):
        logger.info("PoC agent disabled via config (poc.enabled: false)")
        return all_agents, None

    # Fires once after all teams have had time to complete their cycles.
    # Delay = (max_cycles * interval) per team * n_teams, plus generous buffer.
    if max_cycles is not None:
        poc_delay = max_cycles * interval * n_teams + stagger * n_teams * 4 + 60.0
    else:
        poc_delay = interval * n_teams * 4 + 120.0  # rough heuristic when uncapped

    poc_output_dir = str(_get(poc_cfg, "output_dir", default="poc_results"))
    poc_top_n = int(_get(poc_cfg, "top_n", default=3))
    poc_max_retries = int(_get(poc_cfg, "max_retries", default=3))

    # --- Docker executor for PoC code execution ---
    code_timeout   = int(_get(poc_cfg, "code_timeout", default=90))
    docker_image   = str(_get(poc_cfg, "docker_image",   default="python:3.11-slim"))
    docker_memory  = str(_get(poc_cfg, "docker_memory",  default="512m"))
    docker_network = str(_get(poc_cfg, "docker_network", default="none"))

    docker_agent = await manager.launch(
        DockerExecutorAgent,
        args=(docker_image, code_timeout, docker_memory, docker_network),
    )
    _executor_label = f"DockerExecutorAgent (image={docker_image!r})"

    try:
        await docker_agent.ping(timeout=30.0)
    except Exception as ping_err:
        print(f"[simulator] WARNING: {_executor_label} ping failed ({ping_err!r})", flush=True)
        docker_agent = None
    else:
        logger.info("Launched %s (timeout=%ds)", _executor_label, code_timeout)
        if docker_agent:
            all_agents.append(docker_agent)

    poc_agent = await manager.launch(
        ProofOfConceptAgent,
        args=(poc_delay, poc_output_dir, poc_top_n, poc_max_retries),
    )
    try:
        await poc_agent.ping(timeout=30.0)
    except Exception as ping_err:
        print(f"[simulator] WARNING: PoC agent ping failed ({ping_err!r}) — PoC will be skipped.",
              flush=True)
        return all_agents, None
    await poc_agent.set_tournament(global_tournament)
    await poc_agent.set_topic(topics[0])
    if docker_agent is not None:
        await poc_agent.set_executor(docker_agent)
    # Register all team coordinators so the PoC agent polls them for completion
    # instead of relying solely on a fixed time delay.
    for coord in coordinators:
        await poc_agent.add_team_coordinator(coord)
    all_agents.append(poc_agent)
    logger.info(
        "Launched ProofOfConceptAgent (start_delay=%.0fs, output=%r, coordinators=%d, "
        "docker=%s)",
        poc_delay,
        poc_output_dir,
        len(coordinators),
        docker_image if docker_agent else "none",
    )

    return all_agents, poc_agent


# ---------------------------------------------------------------------------
# Report agent launcher
# ---------------------------------------------------------------------------

async def _launch_report_agent(
    manager,
    sim_cfg: dict,
    full_cfg: dict,
    tournament,
) -> Any:
    """
    Launch a ProvenancePDFReportAgent and wire it to the tournament.

    Returns the agent handle, or None when the report section is disabled.
    """
    report_cfg = full_cfg.get('report', {}) or {}
    if not report_cfg.get('enabled', True):
        return None

    agent = await manager.launch(ProvenancePDFReportAgent)
    await agent.ping(timeout=30.0)

    if tournament:
        await agent.set_tournament(tournament)

    poc_cfg  = full_cfg.get('poc') or {}
    poc_dir  = str(_get(poc_cfg, 'output_dir', default='poc_results'))
    await agent.set_poc_dir(poc_dir)

    top_n = int(report_cfg.get('top_n', 5))
    await agent.set_top_n(top_n)

    # Wire provenance paths — the buffer path is derived from the flowcept
    # config (or the default auto-named file); the perf CSV uses the same stem.
    flowcept_cfg = full_cfg.get('flowcept', {}) or {}
    dump_path    = flowcept_cfg.get('dump_path', None)
    perf_csv     = flowcept_cfg.get('perf_csv', None)

    if dump_path:
        await agent.set_provenance_buffer_path(dump_path)
        if not perf_csv:
            import os
            base = os.path.splitext(dump_path)[0]
            perf_csv = f'{base}_perf.csv'
    # The buffer path without a config value is set after flush() is called
    # (see _generate_pdf_report below), where we know the actual filename.

    if perf_csv:
        await agent.set_provenance_perf_csv(perf_csv)

    # Pass the run-log directory so the agent can read actions.jsonl and llm_calls.jsonl
    try:
        from academy_coscientist.utils.utils_logging import get_run_dir
        await agent.set_run_dir(get_run_dir())
    except Exception:
        pass

    logger.info('Launched ProvenancePDFReportAgent (top_n=%d)', top_n)
    return agent


async def _generate_pdf_report(
    report_agent,
    topic: str,
    full_cfg: dict,
) -> None:
    """
    Generate PDF report → log result.

    Called from the simulator shutdown sequence while the Academy manager
    context is still active so the report agent can fetch live data.
    """
    if report_agent is None:
        return

    report_cfg   = full_cfg.get('report', {}) or {}
    flowcept_cfg = full_cfg.get('flowcept', {}) or {}
    output_dir   = report_cfg.get('output_dir', '.')
    filename_tpl = report_cfg.get(
        'output_filename',
        'coscientist_report_{date}.pdf',
    )
    date_str  = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
    filename  = filename_tpl.replace('{date}', date_str)
    import os
    output_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)

    # Buffer path and perf CSV are already wired via _launch_report_agent from
    # the flowcept config dump_path / perf_csv keys.  No flush() needed —
    # the agentic flowcept plugin writes on stop().

    try:
        import asyncio as _asyncio
        pdf_path = await _asyncio.wait_for(
            report_agent.generate_pdf_report(topic=topic, output_path=output_path),
            timeout=600.0,
        )
        if pdf_path:
            print(f'\n[simulator] PDF report → {pdf_path}', flush=True)
        else:
            print('\n[simulator] Report generation returned no path (reportlab missing?)',
                  flush=True)
    except Exception as exc:
        logger.warning('PDF report generation failed: %s', exc)
        print(f'\n[simulator] WARNING: report generation failed — {exc!r}', flush=True)


# ---------------------------------------------------------------------------
# Status reporter
# ---------------------------------------------------------------------------

async def _status_loop(
    tournament,
    interval: float,
    shutdown: asyncio.Event,
) -> None:
    """Periodically print a leaderboard snapshot to stdout."""
    while not shutdown.is_set():
        await asyncio.sleep(interval)
        try:
            board = await tournament.get_leaderboard()
            print(f"\n{'='*60}")
            print(f"  LEADERBOARD  ({len(board)} hypotheses)")
            print(f"{'='*60}")
            for rank, (hid, score, payload) in enumerate(board[:10], 1):
                title = payload.get("title", hid)[:70]
                print(f"  #{rank:>2}  score={score:.3f}  {title}")
            print(f"{'='*60}\n")
        except Exception as e:
            logger.warning("Status loop error: %s", e)


# ---------------------------------------------------------------------------
# Core async runner
# ---------------------------------------------------------------------------

def _print_startup_diagnostics(cfg: dict) -> None:
    """Print a clear pre-flight summary so problems are visible before anything launches."""
    import os
    sim_cfg = cfg.get("simulator", {})
    agents_cfg = sim_cfg.get("agents", {})
    models_cfg = cfg.get("models", {})

    print("\n" + "=" * 64, flush=True)
    print("  SIMULATOR PRE-FLIGHT CHECK", flush=True)
    print("=" * 64, flush=True)

    # API key
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        print(f"  OPENAI_API_KEY : set ({key[:8]}...)", flush=True)
    else:
        print("  OPENAI_API_KEY : NOT SET — LLM calls will fail!", flush=True)

    # Models
    print(f"  reasoning model: {models_cfg.get('reasoning', '(unset)')}", flush=True)
    print(f"  writing   model: {models_cfg.get('writing',   '(unset)')}", flush=True)
    print(f"  embedding model: {models_cfg.get('embedding', '(unset)')}", flush=True)

    # Topics
    topics = sim_cfg.get("topics") or []
    print(f"  topics         : {topics}", flush=True)

    # Agent counts
    print(f"  agents         : "
          f"gen={agents_cfg.get('generation', 1)}  "
          f"review={agents_cfg.get('review', 2)}  "
          f"tournament={agents_cfg.get('tournament', 1)}  "
          f"vectordb={agents_cfg.get('vectordb', 1)}  "
          f"harvester={agents_cfg.get('harvester', 1)}", flush=True)

    intervals = sim_cfg.get("intervals", {})
    print(f"  intervals (s)  : gen={intervals.get('generation', 60)}  "
          f"review={intervals.get('review', 45)}  "
          f"tournament={intervals.get('tournament', 30)}", flush=True)

    print("=" * 64 + "\n", flush=True)


async def _run_simulator(
    cfg: dict,
    exchange,
    executor_obj,
    default_exec: str | None,
    remote_exec: str | None,
) -> None:
    _print_startup_diagnostics(cfg)

    sim_cfg = cfg.get("simulator", {})
    from academy_coscientist.utils.utils_papers import resolve_topic
    raw_topics: list[str] = sim_cfg.get("topics") or ["Artificial intelligence for scientific discovery"]
    topics: list[str] = [resolve_topic(t) for t in raw_topics]
    stagger = float(_get(sim_cfg, "stagger_delay", default=5.0))
    max_duration = _get(sim_cfg, "max_duration_seconds", default=None)
    status_interval = float(_get(sim_cfg, "status_interval_seconds", default=60.0))
    use_vectordb = _get(sim_cfg, "agents", "vectordb", default=1) > 0

    struct_logger = make_struct_logger("simulator")
    struct_logger.info("simulator_start", extra={"topics": topics, "config": sim_cfg})

    manager_kwargs: dict = {"factory": exchange, "executors": executor_obj}
    if default_exec is not None:
        manager_kwargs["default_executor"] = default_exec

    async with await Manager.from_exchange_factory(**manager_kwargs) as manager:
        # --- Vector DB (optional, shared) ---
        vectordb = None
        if use_vectordb:
            launch_kw = {"executor": remote_exec} if remote_exec else {}
            vectordb = await manager.launch(ResearchVectorDBAgent, **launch_kw)
            await vectordb.ping(timeout=60.0)
            logger.info("Launched ResearchVectorDBAgent (executor=%r)", remote_exec or "default")

        # --- Paper harvester (weekly, feeds vector DB) ---
        # Launched before generation agents so the first harvest can populate
        # the vector store before hypothesis generation starts.
        harv_agents = await _launch_harvester_agents(
            manager, sim_cfg, topics, vectordb, stagger, remote_exec=remote_exec
        )

        # --- Global tournament (shared destination for validated hypotheses) ---
        tournament_agents = await _launch_tournament_agents(manager, sim_cfg, stagger)
        tournament = tournament_agents[0]

        # --- PDF report agent (wired once; triggered during shutdown) ---
        report_agent = await _launch_report_agent(
            manager, sim_cfg, cfg, tournament,
        )

        # Decide mode: team-based or flat autonomous agents
        team_mode = bool(_get(sim_cfg, "teams", default=None))

        poc_agent_handle = None
        if team_mode:
            # ---- Team mode -----------------------------------------------
            # Each team has its own generator, refiner, and reviewers managed
            # by a TeamCoordinatorAgent that runs the review-refine loop.
            # Teams are chained: team[i] → staging tournament → team[i+1]
            # → ... → global tournament.
            print("\n[simulator] TEAM MODE enabled", flush=True)
            team_agents, poc_agent_handle = await _launch_teams(
                manager, sim_cfg, cfg.get("poc") or {}, topics, tournament, vectordb, stagger
            )
            total = (
                (1 if use_vectordb else 0)
                + len(harv_agents)
                + len(tournament_agents)
                + len(team_agents)
            )
            n_teams = int(_get(sim_cfg, "teams", "count", default=2))
            max_cyc = _get(sim_cfg, "teams", "max_cycles", default=None)

            print(
                f"\nSimulator running with {total} agents in TEAM MODE "
                f"({n_teams} team(s), max_cycles={max_cyc}, "
                f"global tournament shared, PoC agent included).",
                flush=True,
            )
            struct_logger.info(
                "all_agents_launched",
                extra={
                    "mode": "team",
                    "total": total,
                    "n_teams": n_teams,
                    "team_agents": len(team_agents),
                },
            )

        else:
            # ---- Flat mode (original behaviour) --------------------------
            gen_agents = await _launch_generation_agents(
                manager, sim_cfg, topics, tournament, vectordb, stagger
            )
            rev_agents = await _launch_review_agents(manager, sim_cfg, tournament, stagger)

            total = (
                (1 if use_vectordb else 0)
                + len(harv_agents)
                + len(tournament_agents)
                + len(gen_agents)
                + len(rev_agents)
            )
            print(f"\nSimulator running with {total} autonomous agents (flat mode).", flush=True)
            struct_logger.info(
                "all_agents_launched",
                extra={
                    "mode": "flat",
                    "total": total,
                    "harvester": len(harv_agents),
                    "generation": len(gen_agents),
                    "review": len(rev_agents),
                    "tournament": len(tournament_agents),
                },
            )

        print(f"Topics: {topics}")
        print("Press Ctrl-C to stop.\n")

        # --- Status reporter & shutdown coordination ---
        shutdown_event = asyncio.Event()

        def _handle_sigint(*_):
            print("\nShutting down simulator...")
            shutdown_event.set()

        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, _handle_sigint)
        loop.add_signal_handler(signal.SIGTERM, _handle_sigint)

        status_task = asyncio.create_task(
            _status_loop(tournament, status_interval, shutdown_event)
        )

        # In team mode: auto-stop once the PoC agent (or all team coordinators) finish.
        if poc_agent_handle is not None:
            async def _poc_watcher():
                while not shutdown_event.is_set():
                    await asyncio.sleep(15.0)
                    try:
                        done = await asyncio.wait_for(poc_agent_handle.is_done(), timeout=10.0)
                        if done:
                            print("\n[simulator] PoC agent finished — stopping simulator.", flush=True)
                            shutdown_event.set()
                            return
                    except Exception:
                        pass
            asyncio.create_task(_poc_watcher())
        elif team_mode:
            # PoC agent unavailable — the PoC loop itself will set _done=True quickly
            # (empty-tournament early-exit) which the simulator can't watch directly.
            # Fall back to a max-duration guard so we don't hang forever.
            # If no max_duration is configured, emit a clear warning.
            if not max_duration:
                print(
                    "\n[simulator] WARNING: PoC agent is unavailable and "
                    "max_duration_seconds is not set. The simulator will run "
                    "indefinitely — press Ctrl-C to stop or set max_duration_seconds "
                    "in your config.",
                    flush=True,
                )

        # Wait for either user interrupt or max duration
        if max_duration:
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=float(max_duration))
            except asyncio.TimeoutError:
                print(f"\nMax duration ({max_duration}s) reached. Stopping.")
        else:
            await shutdown_event.wait()

        status_task.cancel()
        struct_logger.info("simulator_stop", extra={})

        # --- Generate PDF report (while manager is still active so the agent
        #     can fetch live data from tournament and meta agents) ---
        if report_agent is not None:
            print("\n[simulator] Generating PDF report…", flush=True)
            await _generate_pdf_report(
                report_agent, topics[0], cfg
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Academy co-scientist autonomous simulator."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the simulator YAML config file.",
    )
    parser.add_argument("--topic", action="append", dest="topics", metavar="TOPIC",
                        help="Research topic (repeatable; overrides config topics).")
    parser.add_argument("--generation-agents", type=int, dest="generation_agents",
                        help="Number of generation agents (overrides config).")
    parser.add_argument("--review-agents", type=int, dest="review_agents",
                        help="Number of review agents (overrides config).")
    parser.add_argument("--max-duration", type=float, dest="max_duration",
                        help="Stop after this many seconds (overrides config).")
    parser.add_argument("--hypotheses-per-cycle", type=int, dest="hypotheses_per_cycle",
                        help="Hypotheses each generation agent proposes per loop cycle.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    init_logging(args.log_level)

    # Load the main pipeline config (models, paths, etc.) AND the simulator config
    cfg = _load_simulator_config(args.config)

    # Apply CLI overrides into the simulator section
    sim = cfg.setdefault("simulator", {})
    agents_sec = sim.setdefault("agents", {})

    if args.topics:
        sim["topics"] = args.topics
    if args.generation_agents is not None:
        agents_sec["generation"] = args.generation_agents
    if args.review_agents is not None:
        agents_sec["review"] = args.review_agents
    if args.max_duration is not None:
        sim["max_duration_seconds"] = args.max_duration
    if args.hypotheses_per_cycle is not None:
        sim["hypotheses_per_cycle"] = args.hypotheses_per_cycle

    # Load models/paths from the same YAML into the pipeline config.
    # Also propagate the path via env var so subprocess workers (process executor)
    # can auto-load the same config without a separate IPC mechanism.
    load_config(args.config)
    import os as _os
    _os.environ.setdefault("ACADEMY_CONFIG_PATH", str(args.config))
    _run_id, run_dir = init_run_context()

    # ── Route outputs into the run directory (unless the user set an explicit path) ──
    # After init_run_context() we know run_dir (e.g. runs/20260310-031053-…/).
    # For each path-bearing config key: if the user left it as null (or omitted it),
    # default to a sub-path inside run_dir.  An explicit non-null value is kept as-is.

    # FlowCept buffer + perf CSV
    flowcept_cfg = cfg.get("flowcept") or {}
    cfg["flowcept"] = flowcept_cfg
    if not flowcept_cfg.get("dump_path"):
        flowcept_cfg["dump_path"] = os.path.join(run_dir, "provenance_buffer.jsonl")
    if not flowcept_cfg.get("perf_csv"):
        flowcept_cfg["perf_csv"] = os.path.join(run_dir, "provenance_perf.csv")

    # PoC results directory (top-level poc: section)
    poc_section = cfg.get("poc") or {}
    cfg["poc"] = poc_section
    if not poc_section.get("output_dir"):
        poc_section["output_dir"] = os.path.join(run_dir, "poc_results")

    # PDF report directory + filename
    report_section = cfg.get("report") or {}
    cfg["report"] = report_section
    if not report_section.get("output_dir"):
        report_section["output_dir"] = run_dir
    report_section.setdefault("output_filename", "report_{date}.pdf")

    print(f"[simulator] Run directory: {run_dir}", flush=True)

    # Build exchange and executors here (before asyncio.run) so that any
    # blocking I/O — especially GlobusComputeExecutor's OAuth + AMQP init —
    # happens in the main thread rather than inside the event loop.
    sim_cfg = cfg.get("simulator", {})
    exchange = _build_exchange(sim_cfg)
    executor_obj, default_exec = _build_executor(sim_cfg)
    remote_exec = _get(sim_cfg, "executor", "remote", default=None)

    def _run():
        asyncio.run(_run_simulator(cfg, exchange, executor_obj, default_exec, remote_exec))

    # FlowCept provenance — driven by ~/.flowcept/settings.yaml (plugins.academy section).
    # When flowcept.enabled is true in the simulator config we wrap with Flowcept()
    # so the academy plugin auto-starts/stops; otherwise we run without provenance.

    flowcept_cfg = cfg.get("flowcept", {})
    if flowcept_cfg.get("enabled", True):
        import flowcept.configs as _fc_configs
        _fc_configs.DUMP_BUFFER_PATH = os.path.join(run_dir, "flowcept_buffer.jsonl")
        # Inject perf_csv into the academy plugin config so FlowCept writes the
        # perf CSV into the run directory instead of the CWD.
        _perf_csv_path = os.path.join(run_dir, "provenance_perf.csv")
        if isinstance(_fc_configs.PLUGINS.get("academy"), dict):
            _fc_configs.PLUGINS["academy"]["perf_csv"] = _perf_csv_path
        from flowcept import Flowcept
        with Flowcept():
            try:
                _run()
            except KeyboardInterrupt:
                pass
    else:
        try:
            _run()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
