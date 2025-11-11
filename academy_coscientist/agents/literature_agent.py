# academy_coscientist/agents/literature_agent.py
from __future__ import annotations
from typing import Any

from academy.agent import action, Agent
from academy_coscientist.utils import utils_llm
from academy_coscientist.utils.utils_logging import make_struct_logger, log_action


class LiteratureAgent(Agent):
    """Summarizes literature enriched by retrieved vector context."""

    def __init__(self, topic: str | None = None, vector_agent=None):
        super().__init__()
        self.topic = topic or ""
        self.vector_agent = vector_agent
        self.logger = make_struct_logger("LiteratureAgent")
        self._summary_cache = ""

    # ------------------------------------------------------------------ #
    # Wiring / configuration
    # ------------------------------------------------------------------ #

    @action
    async def set_topic(self, topic: str) -> None:
        """Set the topic of the literature review."""
        self.topic = topic
        log_action(
            self.logger,
            "set_topic",
            {"topic": topic},
            {"ok": True},
        )

    @action
    async def set_vector_agent(self, vector_agent) -> None:
        """Link to the vector database agent."""
        self.vector_agent = vector_agent
        log_action(
            self.logger,
            "set_vector_agent",
            {"vector_agent": str(vector_agent)},
            {"ok": True},
        )

    @action
    async def get_last_summary(self) -> str:
        """Return the last cached summary (if any)."""
        return self._summary_cache

    # ------------------------------------------------------------------ #
    # Main functionality
    # ------------------------------------------------------------------ #

    @action
    async def review_with_vector_context(self, top_k: int = 3) -> str:
        """Generate a literature summary using retrieved vector context."""
        if not self.topic:
            raise ValueError("Topic not set. Call set_topic() first.")

        context_blocks: list[str] = []
        if self.vector_agent:
            try:
                # your ResearchVectorDBAgent likely exposes `query` or `query_similar`
                result = await self.vector_agent.query(self.topic, k=top_k)
                matches = result.get("matches", result) if isinstance(result, dict) else result
                if matches:
                    context_blocks = [m.get("text", "") for m in matches if isinstance(m, dict)]
                log_action(
                    self.logger,
                    "vector_query_ok",
                    {"topic": self.topic, "k": top_k},
                    {"matches": len(context_blocks)},
                )
            except Exception as e:
                log_action(
                    self.logger,
                    "vector_query_failed",
                    {"topic": self.topic, "error": str(e)},
                    {"ok": False},
                )
        else:
            log_action(
                self.logger,
                "vector_agent_missing",
                {"topic": self.topic},
                {"ok": False},
            )

        joined_context = "\n\n".join(context_blocks)
        prompt = (
            f"You are summarizing scientific literature for the topic: {self.topic}.\n"
            f"Use the following retrieved context if useful:\n{joined_context}\n\n"
            "Write a clear, concise background summary."
        )

        summary = await utils_llm.call_writing_llm(
            system="You are a helpful research assistant, specialized in litterature reviews.",
            user=prompt,
            temperature=0.4,
            max_completion_tokens=800,
        )
        print("Summary from the litterature:","\n", summary)
        self._summary_cache = summary
        log_action(
            self.logger,
            "literature_review_done",
            {"topic": self.topic},
            {"chars": len(summary)},
        )
        return summary
