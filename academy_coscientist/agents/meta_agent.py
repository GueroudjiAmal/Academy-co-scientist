from __future__ import annotations

from typing import List, Tuple, Dict, Any

from academy.agent import Agent, action
from academy_coscientist.utils.utils_logging import make_struct_logger, log_action
from academy_coscientist.utils import utils_llm


class MetaReviewAgent(Agent):
    """
    Produces a portfolio-level synthesis using structured ideas and ELO.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logger = make_struct_logger("MetaReviewAgent")
        self._tournament = None
        self._last_meta: str = ""
        self.logger.debug("MetaReviewAgent init", extra={})

    @action
    async def set_tournament(self, tournament_handle) -> None:
        self._tournament = tournament_handle
        log_action(self.logger, "set_tournament", {"tournament": str(tournament_handle)}, {"ok": True})

    @action
    async def synthesize_portfolio(self, k: int = 10) -> str:
        if not self._tournament:
            log_action(self.logger, "synthesize_portfolio", {"k": k}, {"skipped": "no tournament", "meta_len": len(self._last_meta)})
            return self._last_meta

        top_list: List[Tuple[str, float, Dict[str, Any]]] = await self._tournament.get_top(k)
        meta_summary = await utils_llm.synthesize_meta_report(
            leaderboard=top_list,
            context={"agent": "MetaReviewAgent"},
        )
        self._last_meta = meta_summary

        log_action(
            self.logger,
            "synthesize_portfolio",
            {"k": k, "top_ids": [hid for (hid, _elo, _idea) in top_list]},
            {"meta_len": len(meta_summary), "meta_preview": meta_summary[:200]},
        )
        return meta_summary
