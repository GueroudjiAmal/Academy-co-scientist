from __future__ import annotations

from typing import Any

from academy.agent import action
from academy.agent import Agent

from academy_coscientist.utils import utils_llm
from academy_coscientist.utils.utils_logging import log_action
from academy_coscientist.utils.utils_logging import make_struct_logger


class EvolutionAgent(Agent):
    """Uses critiques to refine leading hypotheses with the LLM,
    then updates TournamentAgent with refined structured ideas.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logger = make_struct_logger('EvolutionAgent')
        self._tournament = None
        self.logger.debug('EvolutionAgent init', extra={})

    @action
    async def set_tournament(self, tournament_handle) -> None:
        self._tournament = tournament_handle
        log_action(
            self.logger, 'set_tournament', {'tournament': str(tournament_handle)}, {'ok': True}
        )

    @action
    async def evolve_top(self, k: int = 10) -> None:
        if not self._tournament:
            log_action(self.logger, 'evolve_top', {'k': k}, {'skipped': 'no tournament'})
            return

        top_list = await self._tournament.get_top(k)
        evolved_ids = []

        for hid, elo, idea in top_list:
            all_critiques = await self._tournament.get_reviews(hid)
            combined = {
                'strengths': [c.get('strengths') for c in all_critiques],
                'weaknesses': [c.get('weaknesses') for c in all_critiques],
                'feasibility_risks': sum(
                    [c.get('feasibility_risks', []) for c in all_critiques], []
                ),
                'suggested_improvements': sum(
                    [c.get('suggested_improvements', []) for c in all_critiques], []
                ),
                'scores': [c.get('score') for c in all_critiques],
            }

            refined: dict[str, Any] = await utils_llm.evolve_hypothesis(
                idea=idea,
                critiques=combined,
                context={'agent': 'EvolutionAgent', 'action': 'refine', 'hid': hid},
            )

            await self._tournament.refine_hypothesis(hid, refined)
            evolved_ids.append(hid)

        log_action(
            self.logger,
            'evolve_top',
            {'k': k},
            {'num_evolved': len(evolved_ids), 'evolved_ids': evolved_ids},
        )
