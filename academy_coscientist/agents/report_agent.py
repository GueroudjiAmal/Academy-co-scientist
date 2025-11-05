# academy_coscientist/agents/report_agent.py

from __future__ import annotations

from typing import Any, List, Tuple

from academy.agent import action, Agent

from academy_coscientist.utils import utils_llm
from academy_coscientist.utils.utils_logging import log_action, make_struct_logger


def _text(s: Any) -> str:
    return "" if s is None else str(s)


LeaderboardRow = Tuple[str, float, dict[str, Any]]  # (id, score, payload)


class ReportAgent(Agent):
    """
    Builds the final technical report.

    Design goals:
    - Always include the original ideas (title + description + scores).
    - Explicitly describe the process used to select the best ideas.
    - Use OpenAI models only; no deterministic content fallback.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logger = make_struct_logger("ReportAgent")
        self._tournament = None
        self._meta = None
        self.logger.debug("ReportAgent init", extra={})

    @action
    async def set_handles(self, tournament, meta) -> None:
        self._tournament = tournament
        self._meta = meta
        log_action(
            self.logger,
            "set_handles",
            {"tournament": str(tournament), "meta": str(meta)},
            {"ok": True},
        )

    # ----------------- internal: fetch rows -----------------

    async def _get_leaderboard_rows(self) -> List[LeaderboardRow]:
        rows: List[LeaderboardRow] = []
        if self._tournament and hasattr(self._tournament, "get_leaderboard"):
            leaderboard = await self._tournament.get_leaderboard()
            for row in leaderboard or []:
                if (
                    isinstance(row, (list, tuple))
                    and len(row) == 3
                    and isinstance(row[0], str)
                    and isinstance(row[2], dict)
                ):
                    try:
                        score = float(row[1])
                    except Exception:
                        score = 0.0
                    rows.append((row[0], score, row[2]))
        return rows

    async def _get_all_hypotheses_rows(self) -> List[LeaderboardRow]:
        rows: List[LeaderboardRow] = []
        t = self._tournament
        if not t:
            return rows

        if hasattr(t, "get_all_hypotheses"):
            hyps = await t.get_all_hypotheses()
        elif hasattr(t, "get_top_hypotheses"):
            hyps = await t.get_top_hypotheses(999999)
        else:
            hyps = []

        for it in hyps or []:
            if (
                isinstance(it, (list, tuple))
                and len(it) == 2
                and isinstance(it[0], str)
                and isinstance(it[1], dict)
            ):
                rows.append((it[0], 0.0, it[1]))
            elif isinstance(it, dict) and "id" in it:
                rows.append((str(it["id"]), 0.0, it))

        return rows

    # ----------------- internal: detail expansion -----------------

    async def _expand_idea_detail(
        self,
        topic: str,
        title: str,
        description: str,
        score: float,
        confidence: float,
    ) -> dict:
        """
        Ask the LLM to turn a leaderboard row into a richer plan.

        Returns a dict with stable keys; if the LLM misbehaves, we raise or
        return a minimal dict (no local pseudo-content).
        """
        system_msg = (
            "You are helping prepare a final research / strategy report. "
            "You receive a single candidate idea (title + short description + score). "
            "Your job is to expand it into a concise but concrete plan.\n\n"
            "Be structured and practical. Do NOT invent new topics; stick to the given one."
        )

        user_msg = (
            f"Overall research topic:\n{topic}\n\n"
            f"Idea title:\n{title}\n\n"
            f"Idea description:\n{description}\n\n"
            f"Panel score (0..1): {score:.2f}\n"
            f"Confidence: {confidence:.2f}\n\n"
            "Return a JSON object with:\n"
            '- "title": refined title (<= 80 chars)\n'
            '- "refined_description": 2–4 sentences, clearer and more concrete\n'
            '- "rationale": 2–3 sentences explaining WHY this is promising\n'
            '- "next_experiment": 1–2 sentences describing the very next test to run\n'
            '- "risk": short phrase for main risk (e.g. "low adoption")\n'
            '- "expected_impact": 1–2 sentences on potential upside if it works\n'
        )

        schema_hint = """
        {
          "title": "string",
          "refined_description": "string",
          "rationale": "string",
          "next_experiment": "string",
          "risk": "string",
          "expected_impact": "string"
        }
        """

        # IMPORTANT: use the function that actually exists in utils_llm
        detail = await utils_llm._call_llm_json(
            system_instructions=system_msg,
            user_prompt=user_msg,
            schema_hint=schema_hint,
            context={
                "agent": "ReportAgent",
                "instance_id": getattr(self, "instance_id", "unknown"),
                "operation": "expand_idea_detail",
                "topic": topic,
                "title": title,
                "score": score,
                "confidence": confidence,
                "audit_path": getattr(self, "audit_path", None),
            },
        )

        if not isinstance(detail, dict):
            # Minimal shape; caller can still use original title/description
            return {
                "title": title,
                "refined_description": description,
                "rationale": "",
                "next_experiment": "",
                "risk": "",
                "expected_impact": "",
                "score": float(score),
                "confidence": float(confidence),
            }

        normalized = {
            "title": detail.get("title", title),
            "refined_description": detail.get(
                "refined_description", description
            ),
            "rationale": detail.get("rationale", ""),
            "next_experiment": detail.get("next_experiment", ""),
            "risk": detail.get("risk", ""),
            "expected_impact": detail.get("expected_impact", ""),
            "score": float(score),
            "confidence": float(confidence),
        }
        return normalized

    # ----------------- public: final report -----------------

    @action
    async def generate_final_report(
        self,
        topic: str,
        top_n: int | None = None,
        detail: bool = True,
    ) -> str:
        """
        Build a detailed report that:
        - Lists all original ideas and their scores.
        - Explains the review & selection process.
        - Expands the top ideas into concrete plans.
        """

        if not self._tournament:
            raise RuntimeError("ReportAgent: tournament handle not set")

        leaderboard = await self._get_leaderboard_rows()
        if not leaderboard:
            # Fallback to raw hypotheses list (still "original ideas")
            leaderboard = await self._get_all_hypotheses_rows()

        # Sort by score descending
        leaderboard = sorted(leaderboard, key=lambda r: r[1], reverse=True)

        if top_n is None or top_n <= 0:
            top_n = len(leaderboard)

        top_rows = leaderboard[:top_n]

        # Build a plain-text table of original ideas
        original_lines = []
        for hid, score, payload in leaderboard:
            title = _text(
                payload.get("title")
                or payload.get("name")
                or f"Idea {hid}"
            )
            desc = _text(
                payload.get("description") or payload.get("text") or ""
            )
            original_lines.append(
                f"- [{hid}] score={score:.3f} :: {title}\n  {desc}"
            )

        original_block = (
            "\n".join(original_lines) if original_lines else "(no ideas)"
        )

        # Ask meta-agent (if configured) for a high-level synthesis of selection
        selection_process_text = ""
        if self._meta and hasattr(self._meta, "synthesize_portfolio"):
            try:
                selection_process_text = await self._meta.synthesize_portfolio(
                    k=top_n
                )
            except Exception as e:
                self.logger.exception(
                    "meta_synthesis_failed",
                    extra={"error": repr(e)},
                )

        # Expand top ideas with more detail (LLM)
        detailed_sections: list[str] = []
        if detail:
            for rank, (hid, score, payload) in enumerate(top_rows, start=1):
                title = _text(
                    payload.get("title")
                    or payload.get("name")
                    or f"Idea {hid}"
                )
                desc = _text(
                    payload.get("description") or payload.get("text") or ""
                )
                confidence = float(payload.get("confidence", 0.0) or 0.0)

                enriched = await self._expand_idea_detail(
                    topic=topic,
                    title=title,
                    description=desc,
                    score=score,
                    confidence=confidence,
                )

                sec = [
                    f"### #{rank}: {enriched['title']} (score={score:.3f}, confidence={confidence:.2f})",
                    "",
                    f"{enriched['refined_description']}",
                    "",
                    f"**Rationale:** {enriched['rationale']}",
                    f"**Next experiment:** {enriched['next_experiment']}",
                    f"**Main risk:** {enriched['risk']}",
                    f"**Expected impact:** {enriched['expected_impact']}",
                ]
                detailed_sections.append("\n".join(sec))

        detailed_block = (
            "\n\n".join(detailed_sections) if detailed_sections else ""
        )

        # Final assembly – this is what the user sees
        report_parts = [
            "# Co-Scientist Report\n",
            "## Topic",
            topic,
            "",
            "## Methodology and Selection Process",
            (
                selection_process_text
                or "Ideas were generated, reviewed by multiple agents, scored, "
                "and ranked based on aggregate score and reviewer confidence."
            ),
            "",
            "## Original Idea Set",
            original_block,
            "",
            "## Top Recommendations and Detailed Plans",
            detailed_block or "(No detailed plans available.)",
        ]

        final_report = "\n\n".join(report_parts)
        log_action(
            self.logger,
            "generate_final_report",
            {"topic": topic, "top_n": top_n, "detail": detail},
            {"report_len": len(final_report)},
        )
        return final_report
