# academy_coscientist/agents/refiner_agent.py
from __future__ import annotations

from typing import Any

from academy.agent import Agent, action

from academy_coscientist.utils import utils_llm
from academy_coscientist.utils.utils_logging import log_action, make_struct_logger


class HypothesisRefinerAgent(Agent):
    """
    Refines a hypothesis given peer-reviewer critique.

    Used by TeamCoordinatorAgent in the iterative review-refine loop.
    Each refinement round produces an improved hypothesis that directly
    addresses identified weaknesses and risks.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logger = make_struct_logger("HypothesisRefinerAgent")
        self._topic: str = ""
        self._team_name: str = ""

    @action
    async def set_topic(self, topic: str) -> None:
        self._topic = topic
        log_action(self.logger, "set_topic", {"topic": topic}, {"ok": True})

    @action
    async def set_team_name(self, name: str) -> None:
        self._team_name = str(name)
        log_action(self.logger, "set_team_name", {"team_name": name}, {"ok": True})

    @action
    async def refine(
        self,
        hypothesis: dict[str, Any],
        critique: dict[str, Any],
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Improve a hypothesis given the current merged critique and full review history.

        Parameters
        ----------
        hypothesis : dict
            Current hypothesis with keys: id, title, description, rationale,
            refinement_round (optional).
        critique : dict
            Merged critique from the most recent review round (score, weaknesses,
            risks, reasoning, recommendation, strengths).
        history : list[dict] | None
            All previous review rounds for this hypothesis, each entry containing:
            ``{"round": int, "avg_score": float, "merged": dict, "critiques": list}``.
            When provided, the LLM is shown the full trajectory so it can identify
            recurring issues that were not addressed in prior refinements.

        Returns
        -------
        dict
            Improved hypothesis with the same id and updated fields plus
            'revision_notes' and 'refinement_round'.
        """
        if not self._topic:
            self.logger.warning("refiner_no_topic", extra={})

        try:
            improved = await utils_llm.refine_hypothesis(
                hypothesis=hypothesis,
                critique=critique,
                topic=self._topic,
                history=history,
                context={
                    "agent": "HypothesisRefinerAgent",
                    "team": self._team_name or None,
                    "instance_id": getattr(self, "instance_id", None),
                    "hyp_id": hypothesis.get("id"),
                },
            )
        except Exception as e:
            self.logger.error(
                "refine_failed",
                extra={"hyp_id": hypothesis.get("id"), "error": repr(e)},
            )
            return hypothesis  # return original on failure

        log_action(
            self.logger,
            "refine",
            {
                "hyp_id":      hypothesis.get("id"),
                "score_before": critique.get("score"),
                "round":       improved.get("refinement_round", 1),
                "history_len": len(history) if history else 0,
            },
            {
                "title_changed": improved.get("title") != hypothesis.get("title"),
                "reasoning": improved.get("reasoning") or "",
                "revision_notes": improved.get("revision_notes") or "",
            },
        )
        return improved
