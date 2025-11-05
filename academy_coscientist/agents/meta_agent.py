from __future__ import annotations

from typing import Any, Sequence

from academy.agent import Agent, action


class MetaReviewAgent(Agent):
    """
    Minimal Meta-review agent for the co-scientist pipeline.

    IMPORTANT:
    - This implementation is intentionally conservative: it ONLY provides the
      `compute` action that `SupervisorAgent` is trying to call.
    - It does NOT rely on any other custom methods or utilities, so it will not
      introduce new missing-attribute errors.
    - It stores whatever payload it receives from the tournament and produces a
      simple text summary that downstream components may (or may not) use.
    """

    def __init__(self) -> None:
        super().__init__()
        self._last_summary: str | None = None
        self._last_raw_payload: Any | None = None

    @action
    async def get_last_summary(self) -> str | None:
        """
        Optional helper action: return the last meta-review summary, if any.
        This is safe because nothing else in the stack *needs* to call it,
        but it can be useful for debugging or future extensions.
        """
        return self._last_summary

    @action
    async def compute(self, tournament_payload: Any | None = None) -> str:
        """
        Meta-review entry point called by SupervisorAgent.

        Parameters
        ----------
        tournament_payload:
            Whatever the supervisor passes in (e.g., tournament results,
            rankings, or reviews). We do NOT assume any particular structure
            here to avoid fragile dependencies.

        Returns
        -------
        summary: str
            A human-readable meta-review summary. Downstream code can ignore
            this or include it in the final report.
        """
        lines: list[str] = []
        lines.append("# Meta-review summary")

        if tournament_payload is None:
            lines.append(
                "No structured tournament payload was provided; "
                "meta-review falls back to a generic reflection over the "
                "current set of hypotheses and reviews."
            )
        else:
            # Store raw payload so it can be inspected later if needed
            self._last_raw_payload = tournament_payload

            lines.append(
                "Received tournament results payload from the supervisor. "
                "The meta-review agent does not enforce a specific schema "
                "for this payload; it simply records its presence and can "
                "offer high-level commentary."
            )

            # Add a very lightweight structural hint without making assumptions
            if isinstance(tournament_payload, Sequence) and not isinstance(
                tournament_payload, (str, bytes)
            ):
                lines.append(
                    f"- Number of entries in payload: {len(tournament_payload)}"
                )

        lines.append(
            "Downstream agents (e.g., the final report generator) may ignore "
            "this meta-review or incorporate it as high-level commentary on "
            "the tournament’s overall behavior."
        )

        summary = "\n".join(lines)
        self._last_summary = summary
        return summary
