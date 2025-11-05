# academy_coscientist/launcher.py

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from academy.exchange import LocalExchangeFactory
from academy.manager import Manager

from academy_coscientist.agents.generation_agent import HypothesisGenerationAgent
from academy_coscientist.agents.review_agent import ReviewAgent
from academy_coscientist.agents.tournament_agent import TournamentAgent
from academy_coscientist.agents.meta_agent import MetaReviewAgent
from academy_coscientist.agents.report_agent import ReportAgent
from academy_coscientist.agents.supervisor_agent import SupervisorAgent
from academy_coscientist.utils.config import (
    load_config,
    get_launch_param,
    maybe_override,
)
from academy_coscientist.utils.utils_logging import (
    init_run_context,
    make_struct_logger,
)


async def _run_with_manager(
    topic: str,
    n_hypotheses: int,
    review_k: Optional[int],
) -> str:
    """
    Orchestrate all agents using the Academy Manager.

    This function only calls actions that are actually defined on the agents:

      - HypothesisGenerationAgent.set_topic
      - HypothesisGenerationAgent.set_tournament

      - ReviewAgent.set_tournament

      - TournamentAgent: add_hypotheses / apply_reviews / get_leaderboard /
                         get_all_hypotheses / get_top_hypotheses / run_tournament

      - MetaReviewAgent.compute

      - ReportAgent.set_handles
      - ReportAgent.generate_final_report

      - SupervisorAgent.set_topic
      - SupervisorAgent.set_counts
      - SupervisorAgent.set_handles
      - SupervisorAgent.run_full_cycle
    """
    logger = make_struct_logger("launcher.manager")

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(),
    ) as manager:
        # Launch agents. Manager.launch takes the Agent *class* as its sole
        # positional argument.
        generation = await manager.launch(HypothesisGenerationAgent)
        reviewer1 = await manager.launch(ReviewAgent)
        reviewer2 = await manager.launch(ReviewAgent)
        tournament = await manager.launch(TournamentAgent)
        meta = await manager.launch(MetaReviewAgent)
        reporter = await manager.launch(ReportAgent)
        supervisor = await manager.launch(SupervisorAgent)

        # Wire handles.

        # Topic + tournament for the generator.
        await generation.set_topic(topic)
        await generation.set_tournament(tournament)

        # Reviewers know which tournament to use.
        await reviewer1.set_tournament(tournament)
        await reviewer2.set_tournament(tournament)

        # Supervisor orchestrates the whole crew.
        await supervisor.set_topic(topic)
        await supervisor.set_counts(
            hypotheses=n_hypotheses,
            review_k=review_k,
        )
        await supervisor.set_handles(
            generation,
            reviewer1,
            reviewer2,
            tournament,
            meta,
            reporter,
        )

        # Reporter needs tournament + meta handle.
        await reporter.set_handles(tournament, meta)

        # Run the full cycle and get the final report as a plain string.
        logger.info(
            "run_full_cycle_start",
            extra={
                "topic": topic,
                "n_hypotheses": n_hypotheses,
                "review_k": review_k,
            },
        )
        report: str = await supervisor.run_full_cycle()
        logger.info(
            "run_full_cycle_done",
            extra={"report_len": len(report)},
        )
        return report


async def run_pipeline(
    topic: str,
    n_hypotheses: int,
    review_k: Optional[int],
) -> str:
    """
    High-level pipeline entry point used by CLI.

    - assumes config has already been loaded
    - initialises per-run logging directory
    - delegates orchestration to `_run_with_manager`
    """
    run_id, run_dir = init_run_context()
    logger = make_struct_logger("launcher")

    logger.info(
        "pipeline_start",
        extra={
            "run_id": run_id,
            "run_dir": run_dir,
            "topic": topic,
            "n_hypotheses": n_hypotheses,
            "review_k": review_k,
        },
    )

    report = await _run_with_manager(
        topic=topic,
        n_hypotheses=n_hypotheses,
        review_k=review_k,
    )

    logger.info(
        "pipeline_done",
        extra={
            "run_id": run_id,
            "run_dir": run_dir,
            "report_len": len(report),
        },
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Academy co-scientist pipeline.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--topic",
        help="Research topic to investigate. Overrides config.launch.topic if provided.",
    )
    parser.add_argument(
        "--hypotheses-count",
        dest="n_hypotheses",
        type=int,
        help="Number of hypotheses to generate. Overrides config.launch.hypotheses_count.",
    )
    parser.add_argument(
        "--review-k",
        dest="review_k",
        type=int,
        help="How many top hypotheses to send for detailed review.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Load config first so we can resolve defaults.
    load_config(args.config)

    # Resolve launch parameters with CLI overrides.
    cfg_topic = get_launch_param("topic", "") or ""
    topic = args.topic if args.topic is not None else cfg_topic

    if not topic.strip():
        raise SystemExit(
            "No topic provided. Use --topic or set launch.topic in the YAML config."
        )

    cfg_n = get_launch_param("hypotheses_count", 4)
    n_hypotheses = maybe_override(cfg_n, args.n_hypotheses)
    try:
        n_hypotheses = int(n_hypotheses)
    except Exception:
        raise SystemExit(f"Invalid hypotheses_count value: {n_hypotheses!r}")

    cfg_k = get_launch_param("reviewer_top_k", None)
    review_k = maybe_override(cfg_k, args.review_k)
    if review_k is not None:
        try:
            review_k = int(review_k)
        except Exception:
            raise SystemExit(f"Invalid review_k value: {review_k!r}")

    # Friendly CLI prints.
    print(f"🧠 Loading config from: {args.config}")
    print(f"📘 Topic: {topic}")

    report = asyncio.run(
        run_pipeline(
            topic=topic,
            n_hypotheses=n_hypotheses,
            review_k=review_k,
        )
    )

    # Dump the report to stdout.
    print(report)


if __name__ == "__main__":
    main()
