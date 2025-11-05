# academy_coscientist/launcher.py
from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import os
import pkgutil
from typing import Any

# agents
from academy_coscientist.agents.generation_agent import HypothesisGenerationAgent
from academy_coscientist.agents.meta_agent import MetaReviewAgent
from academy_coscientist.agents.report_agent import ReportAgent
from academy_coscientist.agents.review_agent import ReviewAgent
from academy_coscientist.agents.supervisor_agent import SupervisorAgent
from academy_coscientist.agents.tournament_agent import TournamentAgent
from academy_coscientist.agents.vector_store_agent import VectorStoreAgent
from academy_coscientist.utils.config import get_path
from academy_coscientist.utils.config import load_config
from academy_coscientist.utils.utils_logging import init_run_context
from academy_coscientist.utils.utils_logging import make_struct_logger
from academy_coscientist.utils.vector_store import ensure_embeddings_for_docs

# ------------------------------------------------------------------------------
# Try to use academy.Manager (preferred), else fall back to direct mode
# ------------------------------------------------------------------------------


def _discover_local_exchange_class() -> type[Any] | None:
    try:
        from academy.manager import Manager  # noqa: F401
    except Exception:
        return None

    candidates: list[tuple[str, str]] = [
        ('academy.exchange.local', 'LocalExchange'),
        ('academy.exchange.memory', 'InMemoryExchange'),
        ('academy.exchange.inprocess', 'LocalExchange'),
        ('academy.exchange.local_exchange', 'LocalExchange'),
        ('academy.exchange', 'LocalExchange'),
        ('academy.exchange', 'InMemoryExchange'),
    ]
    for mod_path, class_name in candidates:
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, class_name, None)
            if inspect.isclass(cls):
                return cls
        except Exception:
            continue

    try:
        ex_root = importlib.import_module('academy.exchange')
        if hasattr(ex_root, '__path__'):
            for m in pkgutil.walk_packages(ex_root.__path__, ex_root.__name__ + '.'):
                try:
                    sub = importlib.import_module(m.name)
                except Exception:
                    continue
                for attr_name in dir(sub):
                    if not attr_name.endswith('Exchange'):
                        continue
                    cand = getattr(sub, attr_name)
                    if inspect.isclass(cand):
                        return cand
    except Exception:
        pass

    return None


async def _robust_launch_vstore_with_manager(manager, path: str, logger) -> Any:
    try:
        vstore = await manager.launch(VectorStoreAgent, path)
        return vstore
    except TypeError:
        pass

    vstore = await manager.launch(VectorStoreAgent)
    if hasattr(vstore, 'set_embeddings_dir'):
        func = vstore.set_embeddings_dir
        if inspect.iscoroutinefunction(func):
            await func(path)
        else:
            func(path)
    elif hasattr(vstore, 'set_path'):
        func = vstore.set_path
        if inspect.iscoroutinefunction(func):
            await func(path)
        else:
            func(path)
    elif hasattr(vstore, 'embeddings_dir'):
        vstore.embeddings_dir = path
    else:
        logger.warning(
            'VectorStoreAgent has no setter for embeddings dir; proceeded without setting path',
            extra={'path': path},
        )
    return vstore


def _robust_construct_vstore_direct(path: str, logger):
    try:
        return VectorStoreAgent(path)
    except TypeError:
        pass

    vstore = VectorStoreAgent()
    if hasattr(vstore, 'set_embeddings_dir'):
        func = vstore.set_embeddings_dir
        if inspect.iscoroutinefunction(func):
            vstore._needs_async_set_path = ('set_embeddings_dir', path)  # type: ignore[attr-defined]
        else:
            func(path)
    elif hasattr(vstore, 'set_path'):
        func = vstore.set_path
        if inspect.iscoroutinefunction(func):
            vstore._needs_async_set_path = ('set_path', path)  # type: ignore[attr-defined]
        else:
            func(path)
    elif hasattr(vstore, 'embeddings_dir'):
        vstore.embeddings_dir = path
    else:
        logger.warning(
            'VectorStoreAgent has no setter for embeddings dir; proceeded without setting path',
            extra={'path': path},
        )
    return vstore


async def _maybe_finish_async_set_path(vstore):
    tup = getattr(vstore, '_needs_async_set_path', None)
    if tup:
        method_name, path = tup
        func = getattr(vstore, method_name, None)
        if func and inspect.iscoroutinefunction(func):
            await func(path)
        delattr(vstore, '_needs_async_set_path')


async def _run_with_manager(topic: str, n_hypotheses: int, review_k: int | None, logger) -> str:
    from academy.manager import Manager

    ExchangeCls = _discover_local_exchange_class()
    if ExchangeCls is None:
        raise RuntimeError('No local Exchange found')

    logger.debug(
        'using exchange', extra={'class': f'{ExchangeCls.__module__}.{ExchangeCls.__name__}'}
    )

    async with await Manager.from_exchange_factory(ExchangeCls) as manager:
        generation = await manager.launch(HypothesisGenerationAgent, topic)
        reviewer1 = await manager.launch(ReviewAgent, 'critic_a')
        reviewer2 = await manager.launch(ReviewAgent, 'critic_b')

        emb_path = get_path('embeddings_dir', 'embeddings')
        vstore = await _robust_launch_vstore_with_manager(manager, emb_path, logger)

        tournament = await manager.launch(TournamentAgent)
        meta = await manager.launch(MetaReviewAgent)
        reporter = await manager.launch(ReportAgent)
        supervisor = await manager.launch(SupervisorAgent)

        # Wire + pass counts explicitly
        await supervisor.set_topic(topic)
        await supervisor.set_counts(n_hypotheses, review_k)

        await generation.set_tournament(tournament)
        await reviewer1.set_vector_store(vstore)
        await reviewer2.set_vector_store(vstore)
        if hasattr(reviewer1, 'set_tournament'):
            await reviewer1.set_tournament(tournament)
        if hasattr(reviewer2, 'set_tournament'):
            await reviewer2.set_tournament(tournament)

        await supervisor.set_handles(generation, reviewer1, reviewer2, tournament, meta, reporter)
        await reporter.set_handles(tournament, meta)

        logger.debug('agents launched', extra={})

        final_report: str = await supervisor.run_full_cycle()
        return final_report


async def _run_direct_mode(topic: str, n_hypotheses: int, review_k: int | None, logger) -> str:
    logger.info('Falling back to DIRECT MODE (no academy exchange found)')

    try:
        generation = HypothesisGenerationAgent(topic)
    except TypeError:
        generation = HypothesisGenerationAgent()
        if hasattr(generation, 'set_topic') and inspect.iscoroutinefunction(generation.set_topic):
            await generation.set_topic(topic)
        elif hasattr(generation, 'set_topic'):
            generation.set_topic(topic)

    reviewer1 = ReviewAgent(name='critic_a')
    reviewer2 = ReviewAgent(name='critic_b')

    emb_path = get_path('embeddings_dir', 'embeddings')
    vstore = _robust_construct_vstore_direct(emb_path, logger)
    await _maybe_finish_async_set_path(vstore)

    tournament = TournamentAgent()
    meta = MetaReviewAgent()
    reporter = ReportAgent()
    supervisor = SupervisorAgent()
    await supervisor.set_topic(topic)
    await supervisor.set_counts(n_hypotheses, review_k)

    await generation.set_tournament(tournament)
    await reviewer1.set_vector_store(vstore)
    await reviewer2.set_vector_store(vstore)
    if hasattr(reviewer1, 'set_tournament'):
        await reviewer1.set_tournament(tournament)
    if hasattr(reviewer2, 'set_tournament'):
        await reviewer2.set_tournament(tournament)
    await supervisor.set_handles(generation, reviewer1, reviewer2, tournament, meta, reporter)
    await reporter.set_handles(tournament, meta)

    logger.debug('direct-mode agents launched', extra={})

    final_report: str = await supervisor.run_full_cycle()
    return final_report


# ------------------------------------------------------------------------------
# Pipeline wrapper
# ------------------------------------------------------------------------------


async def run_pipeline(topic: str, n_hypotheses: int, review_k: int | None) -> str:
    run_id, run_dir = init_run_context()
    logger = make_struct_logger('launcher')

    logger.info(
        'launch_params',
        extra={
            'topic': topic,
            'hypotheses_count': n_hypotheses,
            'review_k': review_k,
        },
    )

    docs_dir = get_path('docs_dir', default='DOCs')
    embeddings_dir = get_path('embeddings_dir', default='embeddings')
    abstracts_dir = get_path(
        'abstracts_cache_dir', default=os.path.join(embeddings_dir, 'abstracts')
    )

    # Ensure embeddings (idempotent)
    new_entries, all_entries = await ensure_embeddings_for_docs(
        docs_dir=docs_dir,
        embeddings_dir=embeddings_dir,
        abstracts_dir=abstracts_dir,
        audit_context={'agent': 'launcher', 'run_id': run_id},
    )

    logger.debug(
        'agents instantiated with preload in vector store',
        extra={'new_embeddings': len(new_entries), 'total_embeddings': len(all_entries)},
    )

    try:
        return await _run_with_manager(topic, n_hypotheses, review_k, logger)
    except Exception as e:
        logger.warning('Manager path failed; switching to direct mode', extra={'error': repr(e)})
        return await _run_direct_mode(topic, n_hypotheses, review_k, logger)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config.')
    parser.add_argument('--topic', type=str, required=True, help='Research topic.')
    parser.add_argument(
        '--hypotheses-count', type=int, required=True, help='Number of hypotheses to generate.'
    )
    parser.add_argument(
        '--review-k',
        type=int,
        required=False,
        help='Number of ideas for reviewers; if omitted, reviewers infer from tournament.',
    )
    args = parser.parse_args()

    load_config(args.config)

    final_report = asyncio.run(
        run_pipeline(
            topic=args.topic,
            n_hypotheses=int(args.hypotheses_count),
            review_k=int(args.review_k) if args.review_k is not None else None,
        )
    )

    print('\n=== FINAL REPORT ===\n')
    print(final_report.strip() if final_report else 'Final report unavailable.')
    print('\n====================\n')


if __name__ == '__main__':
    main()
