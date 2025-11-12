# academy_coscientist/agents/supervisor_agent.py
from __future__ import annotations

import inspect
from typing import Any

from academy.agent import action, Agent
from academy_coscientist.utils import utils_llm
from academy_coscientist.utils.utils_logging import log_action, make_struct_logger


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

async def _call_maybe_async(fn, *args, **kwargs):
    """Call a function that may or may not be async."""
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    return fn(*args, **kwargs)


async def _try_call_variants(obj, method_name: str, variants: tuple[tuple[tuple, dict], ...]):
    """Try several argument signatures when invoking a method dynamically."""
    if not hasattr(obj, method_name):
        return False, None
    method = getattr(obj, method_name)
    if not callable(method):
        return False, None
    for args, kwargs in variants:
        try:
            res = await _call_maybe_async(method, *args, **kwargs)
            return True, res
        except TypeError:
            continue
    return False, None


# ---------------------------------------------------------------------------
# SupervisorAgent
# ---------------------------------------------------------------------------

class SupervisorAgent(Agent):
    """
    Main orchestrator for the Academy Co-Scientist pipeline.

    Coordinates:
        Generation → Review A/B → Tournament → Meta → Report.

    Extended with optional ResearchVectorDBAgent integration so that
    hypothesis generation can use retrieval-augmented generation (RAG),
    and an optional LiteratureAgent to summarize retrieved literature
    before meta-review.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logger = make_struct_logger("SupervisorAgent")

        # agent handles
        self.gen = None
        self.rev_a = None
        self.rev_b = None
        self.tournament = None
        self.meta = None
        self.reporter = None
        self.vectordb = None  # optional FAISS RAG backend
        self.literature = None  # optional LiteratureAgent

        # run parameters
        self._topic: str = "(unspecified topic)"
        self._last_n_ideas: int = 0
        self._review_k: int | None = None

        self.logger.debug("SupervisorAgent initialized", extra={})

    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------

    @action
    async def set_topic(self, topic: str) -> None:
        self._topic = str(topic)
        log_action(self.logger, "set_topic", {"topic": self._topic}, {"ok": True})

    @action
    async def set_counts(self, hypotheses: int, review_k: int | None = None) -> None:
        self._last_n_ideas = int(hypotheses)
        self._review_k = int(review_k) if review_k is not None else None
        log_action(
            self.logger,
            "set_counts",
            {"hypotheses": self._last_n_ideas, "review_k": self._review_k},
            {"ok": True},
        )

    @action
    async def set_handles(
        self,
        gen,
        rev_a,
        rev_b,
        tournament,
        meta,
        reporter,
        vectordb=None,
        literature=None,
    ) -> None:
        """
        Connect subordinate agents.

        Parameters
        ----------
        gen, rev_a, rev_b, tournament, meta, reporter:
            Core pipeline agents.
        vectordb:
            Optional ResearchVectorDBAgent handle, used for RAG.
        literature:
            Optional LiteratureAgent handle, already configured in launcher.
        """
        self.gen = gen
        self.rev_a = rev_a
        self.rev_b = rev_b
        self.tournament = tournament
        self.meta = meta
        self.reporter = reporter
        self.vectordb = vectordb
        self.literature = literature

        # Propagate vectordb to generator if available
        if self.gen and self.vectordb and hasattr(self.gen, "set_vectordb"):
            await self.gen.set_vectordb(self.vectordb)
            log_action(
                self.logger,
                "vectordb_linked",
                {"gen": str(self.gen), "vectordb": str(self.vectordb)},
                {"ok": True},
            )

        log_action(
            self.logger,
            "set_handles",
            {
                "gen": str(gen),
                "rev_a": str(rev_a),
                "rev_b": str(rev_b),
                "tournament": str(tournament),
                "meta": str(meta),
                "reporter": str(reporter),
                "vectordb": str(vectordb),
                "literature": str(literature),
            },
            {"ok": True},
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    async def _review_with_fallback(self, reviewer, tournament) -> None:
        """Try review_all, then review_top(k), falling back gracefully."""
        ok, _ = await _try_call_variants(reviewer, "review_all", (((tournament,), {}), ((), {})))
        if ok:
            log_action(
                self.logger,
                "review_dispatch",
                {"reviewer": str(reviewer), "method": "review_all"},
                {"ok": True},
            )
            return

        k = (
            self._review_k
            if self._review_k is not None
            else self._last_n_ideas
            if self._last_n_ideas > 0
            else None
        )
        variants = (((k,), {}), ((tournament, k), {})) if k is not None else (((), {}),)
        ok, _ = await _try_call_variants(reviewer, "review_top", variants)
        if ok:
            log_action(
                self.logger,
                "review_dispatch",
                {"reviewer": str(reviewer), "method": "review_top", "k": k},
                {"ok": True},
            )
        else:
            log_action(
                self.logger,
                "review_dispatch",
                {"reviewer": str(reviewer)},
                {"warning": "No usable review_* found"},
            )

    async def _collect_reviews(self) -> dict[str, dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}

        async def _pull(reviewer):
            ok, out = await _try_call_variants(reviewer, "get_reviews", (((), {}),))
            return out if (ok and isinstance(out, dict)) else {}

        ra = await _pull(self.rev_a)
        rb = await _pull(self.rev_b)

        def _accumulate(src: dict[str, dict[str, Any]]):
            for hid, rec in src.items():
                crit = rec.get("critique", {})
                score = crit.get("score")
                conf = crit.get("confidence")
                title = rec.get("title")
                if hid not in merged:
                    merged[hid] = {"title": title, "scores": [], "confs": []}
                if isinstance(score, (int, float)):
                    merged[hid]["scores"].append(float(score))
                if isinstance(conf, (int, float)):
                    merged[hid]["confs"].append(float(conf))

        _accumulate(ra)
        _accumulate(rb)

        for hid, agg in merged.items():
            scores, confs = agg["scores"], agg["confs"]
            agg["score"] = sum(scores) / len(scores) if scores else 0.0
            agg["confidence"] = sum(confs) / len(confs) if confs else 0.0

        log_action(
            self.logger,
            "collect_reviews",
            {"ra_count": len(ra), "rb_count": len(rb)},
            {"merged_count": len(merged)},
        )
        return merged

    async def _push_reviews_to_tournament(self, merged_reviews: dict[str, dict[str, Any]]) -> None:
        payload = {
            hid: {
                "score": v.get("score", 0.0),
                "confidence": v.get("confidence", 0.0),
                "title": v.get("title"),
            }
            for hid, v in merged_reviews.items()
        }
        methods = ["apply_reviews", "set_scores", "update_scores", "ingest_reviews", "add_scores"]
        for m in methods:
            ok, _ = await _try_call_variants(self.tournament, m, (((payload,), {}),))
            if ok:
                log_action(
                    self.logger,
                    "tournament_score_inject",
                    {"method": m, "count": len(payload)},
                    {"ok": True},
                )
                return
        log_action(
            self.logger,
            "tournament_score_inject",
            {"methods_tried": methods, "count": len(payload)},
            {"warning": "No scoring hook on tournament"},
        )

    # -----------------------------------------------------------------------
    # Main orchestration
    # -----------------------------------------------------------------------

    @action
    async def run_full_cycle(self) -> str:
        """Run the entire end-to-end pipeline."""
        # commentary = await utils_llm.agent_self_commentary(
        #     "SupervisorAgent",
        #     {"phase": "start", "topic": self._topic},
        #     context={"agent": "SupervisorAgent", "action": "self_commentary"},
        # )
        # self.logger.debug("self_commentary", extra={"sample": commentary[:200]})

        if self._last_n_ideas <= 0:
            raise RuntimeError("set_counts(hypotheses=...) must be positive")
        # 1. Generation (RAG if vectordb linked)
        if self.gen:
            await self.gen.propose_hypotheses(self._last_n_ideas)
        # 2. Reviews
        if self.rev_a:
            await self._review_with_fallback(self.rev_a, self.tournament)
        if self.rev_b:
            await self._review_with_fallback(self.rev_b, self.tournament)

        # 3. Merge + push reviews
        merged = await self._collect_reviews()
        if merged:
            await self._push_reviews_to_tournament(merged)


        # 4. Tournament
        ok, _ = await _try_call_variants(self.tournament, "run_tournament", (((), {}),))
        if not ok:
            await _try_call_variants(self.tournament, "run", (((), {}),))
        # 5. Literature summarization (if wired)
        literature_summary: str | None = None
        if self.literature is not None:
            try:
                literature_summary = await self.literature.review_with_vector_context()
                log_action(
                    self.logger,
                    "literature_summary_generated",
                    {"topic": self._topic},
                    {"chars": len(literature_summary)},
                )
            except Exception as e:
                log_action(
                    self.logger,
                    "literature_summary_failed",
                    {"topic": self._topic, "error": str(e)},
                    {"ok": False},
                )

        # 6. Meta-analysis (with optional literature context)
        leaderboard = await self.tournament.get_leaderboard()
        meta_variants = (
            ((leaderboard, literature_summary), {}),
            ((leaderboard,), {}),
            ((), {}),
        )
        #await _try_call_variants(self.meta, "compute", meta_variants)
        await self.meta.compute(leaderboard, literature_summary)

        # 7. Final report
        top_n = self._last_n_ideas if self._last_n_ideas > 0 else None
        ok, out = await _try_call_variants(
            self.reporter,
            "generate_final_report",
            (
                ((self._topic, top_n, True), {}),
                ((self._topic, top_n), {}),
                ((self._topic,), {}),
            ),
        )
        final_report = out if (ok and isinstance(out, str)) else "Final report unavailable."
        log_action(
            self.logger,
            "run_full_cycle",
            {
                "topic": self._topic,
                "n_ideas": self._last_n_ideas,
                "review_k": self._review_k,
                "rag_enabled": bool(self.vectordb),
                "literature_used": bool(literature_summary),
            },
            {"final_report_len": len(final_report)},
        )
        return final_report
