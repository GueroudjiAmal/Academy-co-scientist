from __future__ import annotations

from academy.agent import Agent, action
from academy_coscientist.utils.utils_logging import make_struct_logger, log_action
from academy_coscientist.utils import utils_llm


class LiteratureReviewAgent(Agent):
    """
    Maintains a background summary for the research topic using LLM.
    """

    def __init__(self, topic: str) -> None:
        super().__init__()
        self.topic = topic
        self.logger = make_struct_logger("LiteratureAgent")
        self._summary_cache: str = ""
        self.logger.debug(
            "LiteratureReviewAgent init",
            extra={"topic": topic},
        )

    @action
    async def refresh_summary(self) -> None:
        summary = await utils_llm.summarize_background(
            topic=self.topic,
            context={"agent": "LiteratureReviewAgent"},
        )
        self._summary_cache = summary
        log_action(
            self.logger,
            "refresh_summary",
            {"topic": self.topic},
            {
                "summary_len": len(summary),
                "summary_preview": summary[:200],
            },
        )

    @action
    async def get_summary(self) -> str:
        log_action(
            self.logger,
            "get_summary",
            {"topic": self.topic},
            {"summary_len": len(self._summary_cache)},
        )
        return self._summary_cache
