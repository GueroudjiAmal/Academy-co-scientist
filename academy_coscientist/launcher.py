# academy_coscientist/launcher.py

from __future__ import annotations
import argparse
import asyncio
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional


from academy.manager import Manager
from academy.logging import init_logging


from academy_coscientist.agents.generation_agent import HypothesisGenerationAgent
from academy_coscientist.agents.review_agent import ReviewAgent
from academy_coscientist.agents.tournament_agent import TournamentAgent
from academy_coscientist.agents.meta_agent import MetaReviewAgent
from academy_coscientist.agents.report_agent import ReportAgent
from academy_coscientist.agents.supervisor_agent import SupervisorAgent
from academy_coscientist.agents.research_vector_agent import ResearchVectorDBAgent
from academy_coscientist.agents.literature_agent import LiteratureAgent

from academy_coscientist.utils.config import load_config, get_launch_param, maybe_override
from academy_coscientist.utils.utils_logging import init_run_context, make_struct_logger

from globus_compute_sdk import Executor as GCExecutor
from academy.exchange.cloud.client import HttpExchangeFactory

EXCHANGE_ADDRESS = 'https://exchange.academy-agents.org'

async def _run_with_manager(
    topic: str,
    n_hypotheses: int,
    review_k: Optional[int],
) -> str:
    logger = make_struct_logger("launcher.manager")
    if 'ACADEMY_TUTORIAL_ENDPOINT' in os.environ:
        executor = GCExecutor(os.environ['ACADEMY_TUTORIAL_ENDPOINT'])
        print("Setting up globus")

    else:
        mp_context = multiprocessing.get_context('spawn')
        executor = ProcessPoolExecutor(
            max_workers=32,
            initializer=init_logging,
            mp_context=mp_context,
        )

    async with await Manager.from_exchange_factory(
    factory=HttpExchangeFactory(
        EXCHANGE_ADDRESS,
        auth_method='globus',
    ),
    executors=executor,
    ) as manager:
    # async with await Manager.from_exchange_factory(
    #     factory=LocalExchangeFactory(),
    #     executors=ThreadPoolExecutor(),
    # ) as manager:


        # --- Launch all agents ---
        vectordb = await manager.launch(ResearchVectorDBAgent)
        await vectordb.ping()
        generation = await manager.launch(HypothesisGenerationAgent)
        await generation.ping()
        reviewer1 = await manager.launch(ReviewAgent)
        await reviewer1.ping()
        reviewer2 = await manager.launch(ReviewAgent)
        await reviewer2.ping()
        tournament = await manager.launch(TournamentAgent)
        meta = await manager.launch(MetaReviewAgent)
        await meta.ping()
        reporter = await manager.launch(ReportAgent)
        await reporter.ping()
        literature = await manager.launch(LiteratureAgent)
        await literature.ping()
        supervisor = await manager.launch(SupervisorAgent)
        await supervisor.ping()

        # --- Wire them together ---

        # Topic & tournament for generator
        await generation.set_topic(topic)
        await generation.set_tournament(tournament)
        await generation.set_vectordb(vectordb)

        # Reviewers know which tournament to use
        await reviewer1.set_tournament(tournament)
        await reviewer2.set_tournament(tournament)

        # Literature agent: topic + vector backend
        await literature.set_topic(topic)
        await literature.set_vector_agent(vectordb)

        # Reporter needs tournament + meta handle
        await reporter.set_handles(tournament, meta)

        # Supervisor orchestrates everything
        await supervisor.set_topic(topic)
        await supervisor.set_counts(hypotheses=n_hypotheses, review_k=review_k)
        await supervisor.set_handles(
            generation,
            reviewer1,
            reviewer2,
            tournament,
            meta,
            reporter,
            vectordb,   # keep RAG wiring
            literature, # new literature context handle
        )


        logger.info(
            "run_full_cycle_start",
            extra={"topic": topic, "n_hypotheses": n_hypotheses, "review_k": review_k},
        )
        report = await supervisor.run_full_cycle()

        logger.info("run_full_cycle_done", extra={"report_len": len(report)})
        return report


async def run_pipeline(topic: str, n_hypotheses: int, review_k: Optional[int]) -> str:
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

    report = await _run_with_manager(topic, n_hypotheses, review_k)

    logger.info(
        "pipeline_done",
        extra={"run_id": run_id, "run_dir": run_dir, "report_len": len(report)},
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Academy co-scientist pipeline.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    parser.add_argument("--topic", help="Research topic to investigate.")
    parser.add_argument(
        "--hypotheses-count", dest="n_hypotheses", type=int, help="Number of hypotheses to generate."
    )
    parser.add_argument("--review-k", dest="review_k", type=int, help="Top hypotheses to review.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    load_config(args.config)

    cfg_topic = get_launch_param("topic", "") or ""
    topic = args.topic if args.topic is not None else cfg_topic
    if not topic.strip():
        raise SystemExit("No topic provided. Use --topic or set launch.topic in the YAML config.")

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

    print(f"🧠 Loading config from: {args.config}")
    print(f"📘 Topic: {topic}")

    report = asyncio.run(run_pipeline(topic, n_hypotheses, review_k))
    print(report)


if __name__ == "__main__":
    main()
