# academy_coscientist/agents/generation_agent.py
from __future__ import annotations

from typing import Any

from academy.agent import action
from academy.agent import Agent

from academy_coscientist.utils import utils_llm
from academy_coscientist.utils.utils_logging import log_action
from academy_coscientist.utils.utils_logging import make_struct_logger


class HypothesisGenerationAgent(Agent):
    """Generates hypotheses for a given topic via LLM.
    IMPORTANT: No implicit default count here — caller must pass n.
    Includes a resilient local fallback so we never produce zero ideas.
    """

    def __init__(self, topic: str | None = None) -> None:
        super().__init__()
        self.logger = make_struct_logger('HypothesisGenerationAgent')
        self.topic: str | None = topic
        self._tournament = None
        self._ideas: list[dict[str, Any]] = []
        self.logger.debug('HypothesisGenerationAgent init', extra={'has_topic': bool(topic)})

    @action
    async def set_topic(self, topic: str) -> None:
        self.topic = topic
        log_action(self.logger, 'set_topic', {'topic': topic}, {'ok': True})

    @action
    async def set_tournament(self, tournament_handle) -> None:
        self._tournament = tournament_handle
        log_action(
            self.logger, 'set_tournament', {'tournament': str(tournament_handle)}, {'ok': True}
        )

    def _fallback_brainstorm(self, topic: str, n: int) -> list[dict[str, Any]]:
        ideas: list[dict[str, Any]] = []
        templates = [
            'Benchmark {aspect} of {topic} using standardized workloads on representative HPC nodes.',
            'Instrument {component} in {topic} pipelines to capture tail latency and scalability limits on HPC.',
            'Prototype {approach} for {topic} and compare throughput vs. cost across HPC interconnects.',
            'Design failure-injection tests for {topic} to measure resilience under HPC job schedulers.',
            'Co-design data layout for {topic} to exploit NUMA & multi-socket patterns on HPC.',
            'Evaluate hybrid CPU/GPU placement for {topic} under real HPC queues and preemption scenarios.',
            'Quantify I/O contention effects on {topic} using synthetic and real traces on HPC FS.',
            'Establish conformance tests for {topic} APIs under MPI/Slurm environments.',
            'Model performance portability of {topic} across generations of HPC hardware.',
            'Automate reproducible runs for {topic} with container and module systems on HPC.',
        ]
        aspects = [
            ('end-to-end performance', 'indexing'),
            ('latency under load', 'query execution'),
            ('memory behavior', 'ingest path'),
            ('fault tolerance', 'replication'),
            ('scalability', 'sharding'),
            ('scheduler integration', 'batching'),
            ('I/O throughput', 'persistence'),
            ('network sensitivity', 'consistency'),
            ('portability', 'accelerators'),
            ('reproducibility', 'observability'),
        ]
        for i in range(n):
            t = templates[i % len(templates)]
            a1, a2 = aspects[i % len(aspects)]
            text = t.format(aspect=a1, component=a2, approach=a1, topic=topic)
            ideas.append(
                {
                    'title': f'#{i + 1} {a1.title()} for {topic}',
                    'description': text.strip(),
                }
            )
        return ideas

    @action
    async def propose_hypotheses(self, n: int | None = None) -> None:
        """Generate hypotheses for self.topic.
        - n must be provided by the caller (no implicit defaults).
        - Guarantees non-empty output by using a local fallback if the LLM returns none.
        """
        if not self.topic:
            raise RuntimeError('Topic not set on HypothesisGenerationAgent')
        if n is None:
            raise RuntimeError("propose_hypotheses requires 'n' (no implicit default)")

        ideas: list[dict[str, Any]] = []
        llm_error = None
        try:
            ideas = await utils_llm.brainstorm_hypotheses(
                topic=self.topic,
                n=int(n),
                context={'agent': 'HypothesisGenerationAgent', 'action': 'brainstorm'},
            )
            if not isinstance(ideas, list):
                ideas = []
        except Exception as e:
            llm_error = repr(e)
            ideas = []

        if len(ideas) < int(n):
            fallback_needed = int(n) - len(ideas)
            fallback = self._fallback_brainstorm(self.topic, fallback_needed)
            ideas = (ideas or []) + fallback

        if len(ideas) > int(n):
            ideas = ideas[: int(n)]

        self._ideas = ideas

        added = 0
        if self._tournament and self._ideas:
            if hasattr(self._tournament, 'add_hypotheses'):
                await self._tournament.add_hypotheses(self._ideas)
                added = len(self._ideas)
            elif hasattr(self._tournament, 'add'):
                for idea in self._ideas:
                    await self._tournament.add(idea)
                    added += 1

        log_action(
            self.logger,
            'propose_hypotheses',
            {'topic': self.topic, 'n_requested': n, 'llm_error': llm_error},
            {'n_generated': len(self._ideas), 'n_added_to_tournament': added},
        )

    @action
    async def get_ideas(self) -> list[dict[str, Any]]:
        log_action(self.logger, 'get_ideas', {'requested': True}, {'count': len(self._ideas)})
        return list(self._ideas)
