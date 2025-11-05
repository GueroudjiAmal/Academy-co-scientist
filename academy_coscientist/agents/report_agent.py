# academy_coscientist/agents/report_agent.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from academy.agent import Agent, action
from academy_coscientist.utils.utils_logging import make_struct_logger, log_action
from academy_coscientist.utils import utils_llm


def _text(s: Any) -> str:
    return "" if s is None else str(s)


class ReportAgent(Agent):
    """
    Builds the final technical report.

    Improvements:
    - If leaderboard empty, fallback to raw hypotheses so you never get "Top 0".
    - Optional `detail=True` expands each idea into a full plan via LLM (or a deterministic fallback).
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
        log_action(self.logger, "set_handles",
                   {"tournament": str(tournament), "meta": str(meta)}, {"ok": True})

    # ----------------- internal: fetch rows -----------------

    async def _get_leaderboard_rows(self) -> List[Tuple[str, float, Dict[str, Any]]]:
        rows: List[Tuple[str, float, Dict[str, Any]]] = []
        if self._tournament and hasattr(self._tournament, "get_leaderboard"):
            leaderboard = await self._tournament.get_leaderboard()
            for row in leaderboard or []:
                if isinstance(row, (list, tuple)) and len(row) == 3 and isinstance(row[0], str) and isinstance(row[2], dict):
                    try:
                        score = float(row[1])
                    except Exception:
                        score = 0.0
                    rows.append((row[0], score, row[2]))
        return rows

    async def _get_all_hypotheses_rows(self) -> List[Tuple[str, float, Dict[str, Any]]]:
        rows: List[Tuple[str, float, Dict[str, Any]]] = []
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
            if isinstance(it, (list, tuple)) and len(it) == 2 and isinstance(it[0], str) and isinstance(it[1], dict):
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

        Returns a dict with stable keys, even if the LLM misbehaves
        and returns plain text instead of JSON.
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
            '- "risk": short phrase for main risk (e.g. \"low adoption\", \"poor signal\")\n'
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

        # Ask the LLM for structured JSON
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
                # If you already set self.audit_path in __init__, this helps log LLM calls:
                "audit_path": getattr(self, "audit_path", None),
            },
        )

        # ---- Robust normalization layer ----
        # utils_llm._call_llm_json *should* return a dict,
        # but it may sometimes return a string or other shape.
        if isinstance(detail, str):
            # We only got freeform text; wrap it so the rest of the code
            # always sees a dict with stable keys.
            return {
                "title": title,
                "refined_description": description,
                "rationale": detail,
                "next_experiment": "",
                "risk": "",
                "expected_impact": "",
                "score": float(score),
                "confidence": float(confidence),
            }

        if not isinstance(detail, dict):
            # Totally unexpected shape -> safe fallback
            return {
                "title": title,
                "refined_description": description,
                "rationale": f"Unstructured LLM output: {detail!r}",
                "next_experiment": "",
                "risk": "",
                "expected_impact": "",
                "score": float(score),
                "confidence": float(confidence),
            }

        # Some models/ prompts might nest the actual object under a key like "detail"
        if "detail" in detail and isinstance(detail["detail"], dict):
            detail = detail["detail"]

        # Merge with defaults and keep score/confidence explicitly
        normalized = {
            "title": detail.get("title", title),
            "refined_description": detail.get("refined_description", description),
            "rationale": detail.get("rationale", ""),
            "next_experiment": detail.get("next_experiment", ""),
            "risk": detail.get("risk", ""),
            "expected_impact": detail.get("expected_impact", ""),
            "score": float(score),
            "confidence": float(confidence),
        }

        return normalized