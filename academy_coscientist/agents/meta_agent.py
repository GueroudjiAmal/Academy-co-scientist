# academy_coscientist/agents/meta_agent.py
from __future__ import annotations

from typing import Any, Sequence

from academy.agent import Agent, action
from academy_coscientist.utils.utils_logging import make_struct_logger, log_action
from academy_coscientist.utils.utils_llm import (
    rewrite_meta_summary_with_llm,
    summarize_portfolio_with_llm,
)


class MetaReviewAgent(Agent):
    """
    Meta-review agent for the co-scientist pipeline.

    Responsibilities:
    - Compute a meta-review summary after the tournament stage.
    - Generate a final portfolio synthesis for inclusion in the report.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logger = make_struct_logger("MetaReviewAgent")
        self._last_summary: str | None = None
        self._last_raw_payload: Any | None = None

    # -----------------------------------------------------------------------
    # Accessors
    # -----------------------------------------------------------------------

    @action
    async def get_last_summary(self) -> str | None:
        """
        Optional helper: return the last meta-review summary, if any.
        """
        return self._last_summary

    # -----------------------------------------------------------------------
    # Meta-review computation (called by SupervisorAgent)
    # -----------------------------------------------------------------------

    @action
    async def compute(self, tournament_payload: Any | None = None, context: Any | None = None) -> str:
        """
        Meta-review entry point.

        We build a lightweight, schema-agnostic description of what happened,
        then let the LLM rewrite it into a more polished paragraph.
        """

        lines: list[str] = ["# Meta-review summary"]
        if tournament_payload is None:
            lines.append(
                "No structured tournament payload was provided; "
                "meta-review falls back to a generic reflection over the "
                "current set of hypotheses and reviews."
            )
        elif context is None:
            lines.append(
                "No context from litterature was provided; "
            )
        else:
            self._last_raw_payload = tournament_payload
            self._context = context

            lines.append(
                "Received tournament results payload and context from the supervisor. "
                "Providing high-level commentary."
            )


            lines.append(f"- Number of entries in payload: {len(tournament_payload)}")
            lines.append(str(tournament_payload))
            lines.append("And here is the context from the litterature:")
            lines.append(context)

        lines.append(
            "Downstream agents (e.g., the final report generator) may include "
            "this meta-review as contextual commentary on the tournament's outcome."
        )

        #raw_summary = "\n".join(lines)

        # Let the LLM polish this into a nicer paragraph.
        rewritten = await rewrite_meta_summary_with_llm(lines, context=None)
        self._last_summary = rewritten
        log_action(
            self.logger,
            "compute",
            {"payload_type": str(type(tournament_payload)), "context": str(type(context))},
            {"ok": True},
        )
        return rewritten

    # -----------------------------------------------------------------------
    # Portfolio synthesis (called by ReportAgent)
    # -----------------------------------------------------------------------

    @action
    async def synthesize_portfolio(self, *args, **kwargs) -> str:
        """
        Summarize the overall tournament results into a clear, human-readable
        synthesis for the final report.

        We ACCEPT extra kwargs (e.g., 'k') for compatibility with ReportAgent,
        but we don't depend on them.
        """
        # Build a generic but accurate description of the pipeline, without
        # relying on internals of TournamentAgent that we don't see here.
        base_lines: list[str] = []

        base_lines.append(
            "The co-scientist pipeline generated multiple candidate hypotheses "
            "for the given research topic using a dedicated hypothesis "
            "generation agent."
        )
        base_lines.append(
            "Two independent review agents evaluated these hypotheses along "
            "dimensions such as clarity, novelty, feasibility, and potential impact."
        )
        base_lines.append(
            "Their scores and critiques were aggregated and passed to a tournament "
            "agent, which ranked the hypotheses and selected a top subset."
        )
        base_lines.append(
            "The meta-review stage then examined this ranked set to provide "
            "high-level commentary and ensure that the final selection is "
            "coherent with the overall research goals."
        )

        if self._last_summary:
            base_lines.append(
                "\nPrevious meta-review summary (for context):\n" + self._last_summary
            )

        portfolio_text = "\n".join(base_lines)

        summary = await summarize_portfolio_with_llm(
            portfolio_text,
            context={"called_from": "MetaReviewAgent"},
        )

        log_action(
            self.logger,
            "synthesize_portfolio",
            {"args": args, "kwargs": kwargs},
            {"ok": True},
        )
        return summary
