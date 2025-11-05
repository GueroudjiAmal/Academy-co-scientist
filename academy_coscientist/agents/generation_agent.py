# academy_coscientist/agents/hypothesis_agent.py

from __future__ import annotations

from typing import Any, List, Optional

from academy.agent import Agent, action

from academy_coscientist.utils import utils_llm
from academy_coscientist.utils.utils_logging import log_action, make_struct_logger


class HypothesisGenerationAgent(Agent):
    """
    Agent responsible for proposing candidate ideas / hypotheses for a topic.

    - No deterministic "local" brainstorming fallback: all content comes from
      OpenAI LLM calls via utils_llm.brainstorm_hypotheses.
    - Designed to be topic-generic; domain comes entirely from the `topic` string.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logger = make_struct_logger("HypothesisGenerationAgent")
        self._topic: Optional[str] = None
        self._tournament = None
        self._ideas: List[dict[str, Any]] = []

    # ------------------- configuration -------------------

    @action
    async def set_topic(self, topic: str) -> None:
        self._topic = topic
        log_action(
            self.logger,
            "set_topic",
            {"topic": topic},
            {"ok": True},
        )

    @action
    async def set_tournament(self, tournament) -> None:
        """
        Tournament agent handle. We expect it to implement either:

        - add_hypotheses(list[dict]):
            each dict is a hypothesis/idea, or

        - add(payload: dict):
            called one by one.

        This keeps the agent generic and decoupled from specific tournament
        implementations.
        """
        self._tournament = tournament
        log_action(
            self.logger,
            "set_tournament",
            {"tournament": str(tournament)},
            {"ok": True},
        )

    # ------------------- main behavior -------------------

    @action
    async def propose_hypotheses(self, n: int | None = None) -> None:
        """
        Generate `n` candidate hypotheses / ideas using the LLM.

        No deterministic content fallback:
        - If the LLM call fails, we log and raise.
        - We never fabricate ideas locally.
        """
        if not self._topic:
            raise RuntimeError("HypothesisGenerationAgent: topic not set")

        if n is None:
            raise RuntimeError("propose_hypotheses requires 'n'")

        topic = self._topic

        try:
            ideas = await utils_llm.brainstorm_hypotheses(
                topic=topic,
                n=int(n),
                context={
                    "agent": "HypothesisGenerationAgent",
                    "action": "brainstorm",
                    "instance_id": getattr(self, "instance_id", None),
                    "audit_path": getattr(self, "audit_path", None),
                },
            )
        except Exception as e:
            self.logger.exception(
                "brainstorm_failed",
                extra={
                    "error": repr(e),
                    "topic": topic,
                    "n_requested": n,
                },
            )
            raise

        if not isinstance(ideas, list):
            raise RuntimeError(
                f"LLM brainstorm returned non-list: {type(ideas)!r}"
            )

        self._ideas = ideas
        added = 0

        if self._tournament and self._ideas:
            if hasattr(self._tournament, "add_hypotheses"):
                await self._tournament.add_hypotheses(self._ideas)
                added = len(self._ideas)
            elif hasattr(self._tournament, "add"):
                for idea in self._ideas:
                    await self._tournament.add(idea)
                    added += 1

        log_action(
            self.logger,
            "propose_hypotheses",
            {"topic": topic, "n_requested": n},
            {"n_generated": len(self._ideas), "n_added_to_tournament": added},
        )

    # ------------------- accessors -------------------

    @action
    async def get_ideas(self) -> list[dict[str, Any]]:
        return list(self._ideas)
