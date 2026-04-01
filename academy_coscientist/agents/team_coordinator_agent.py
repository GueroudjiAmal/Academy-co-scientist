# academy_coscientist/agents/team_coordinator_agent.py
"""
TeamCoordinatorAgent — orchestrates one co-scientist team.

A team consists of:
  - One HypothesisGenerationAgent  (brainstorm or elaborate hypotheses)
  - One HypothesisRefinerAgent     (improve hypotheses given critique)
  - One or more ReviewAgents       (score hypotheses on multiple dimensions)

The coordinator runs a @loop that executes the generate → review → refine cycle:

  1. Generate N hypotheses (or pull from an upstream tournament if this is
     a downstream team receiving work from a prior team).
  2. For each hypothesis, run the review-refine loop:
       a. Collect reviews from all team reviewers.
       b. Average the scores.
       c. If avg_score >= score_threshold  → hypothesis is VALIDATED; push to
          the downstream tournament and log the handoff.
       d. Else if rounds < max_refinement_rounds → ask the refiner to improve
          the hypothesis and repeat from (a).
       e. Else (max rounds exhausted) → if avg_score >= min_pass_score, push
          anyway; otherwise discard with a log message.

Multi-team chaining:
  - Team 1 (first team): set_incoming(None), generates from scratch.
  - Team 2+ (downstream): set_incoming(staging_tournament), reads validated
    hypotheses produced by the prior team instead of brainstorming.
  - All teams share the same outgoing (global) tournament as their final
    destination.  Intermediate staging tournaments are separate.

This design means every hypothesis reaching the global tournament has been
reviewed and iteratively refined by at least one full team.
"""

from __future__ import annotations

import asyncio
import traceback
import uuid
from typing import Any

from academy.agent import Agent, action, loop

from academy_coscientist.utils.utils_logging import log_action, make_struct_logger


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _merge_critiques(critiques: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge multiple reviewer critiques into one combined critique dict."""
    scores: list[float] = []
    confs: list[float] = []
    weaknesses: list[str] = []
    risks: list[str] = []
    strengths: list[str] = []
    reasonings: list[str] = []
    recommendations: list[str] = []

    for c in critiques:
        if isinstance(c.get("score"), (int, float)):
            scores.append(float(c["score"]))
        if isinstance(c.get("confidence"), (int, float)):
            confs.append(float(c["confidence"]))
        weaknesses.extend(c.get("weaknesses") or [])
        risks.extend(c.get("risks") or [])
        strengths.extend(c.get("strengths") or [])
        if c.get("reasoning"):
            reasonings.append(str(c["reasoning"]))
        if c.get("recommendation"):
            recommendations.append(str(c["recommendation"]))

    # Majority-vote recommendation: most common value, defaulting to "revise"
    recommendation = max(set(recommendations), key=recommendations.count) if recommendations else "revise"

    return {
        "score":          _avg(scores),
        "confidence":     _avg(confs),
        "weaknesses":     list(dict.fromkeys(weaknesses)),   # deduplicate, preserve order
        "risks":          list(dict.fromkeys(risks)),
        "strengths":      list(dict.fromkeys(strengths)),
        "reasoning":      " | ".join(reasonings),
        "recommendation": recommendation,
    }


class TeamCoordinatorAgent(Agent):
    """
    Coordinates a co-scientist team through the generate → review → refine loop.

    Parameters
    ----------
    team_name : str
        Human-readable team identifier (e.g. "team_alpha", "team_beta").
    score_threshold : float
        Minimum average review score for a hypothesis to be considered validated
        and forwarded to the downstream tournament.
    min_pass_score : float
        Minimum score to still forward a hypothesis after max_refinement_rounds
        have been exhausted without reaching score_threshold.
    max_refinement_rounds : int
        Maximum number of refine-then-re-review cycles per hypothesis.
    hypotheses_per_cycle : int
        Number of hypotheses to generate (or pull from incoming) per loop cycle.
    loop_interval : float
        Seconds between team loop cycles.
    loop_start_delay : float
        Seconds to wait before the first cycle (for staggered startup).
    """

    def __init__(
        self,
        team_name: str = "team",
        score_threshold: float = 0.65,
        min_pass_score: float = 0.40,
        max_refinement_rounds: int = 3,
        hypotheses_per_cycle: int = 3,
        loop_interval: float = 120.0,
        loop_start_delay: float = 0.0,
        max_cycles: int | None = None,
    ) -> None:
        super().__init__()
        self.logger = make_struct_logger(f"TeamCoordinator:{team_name}")
        self._team_name = team_name
        self._score_threshold = float(score_threshold)
        self._min_pass_score = float(min_pass_score)
        self._max_rounds = int(max_refinement_rounds)
        self._n_hypotheses = int(hypotheses_per_cycle)
        self._loop_interval = float(loop_interval)
        self._loop_start_delay = float(loop_start_delay)
        self._max_cycles: int | None = int(max_cycles) if max_cycles is not None else None

        # Agent handles — set via actions before the loop starts
        self._generator = None
        self._refiner = None
        self._reviewers: list = []
        self._incoming_tournament = None   # upstream staging (None for first team)
        self._outgoing_tournament = None   # global tournament (all teams push here)
        self._chain_outgoing = None        # staging for next team (non-final teams only)
        self._topic: str = ""

        # Stats
        self._total_validated: int = 0
        self._total_discarded: int = 0
        self._total_refined: int = 0
        self._done: bool = False

    # ------------------------------------------------------------------
    # Configuration actions
    # ------------------------------------------------------------------

    @action
    async def set_topic(self, topic: str) -> None:
        self._topic = str(topic)
        log_action(self.logger, "set_topic", {"topic": topic}, {"ok": True})
        # Propagate to already-registered reviewers and refiner so they
        # evaluate/refine within the correct research domain.
        for reviewer in self._reviewers:
            if hasattr(reviewer, "set_topic"):
                try:
                    await reviewer.set_topic(topic)
                except Exception:
                    pass
        if self._refiner is not None and hasattr(self._refiner, "set_topic"):
            try:
                await self._refiner.set_topic(topic)
            except Exception:
                pass

    @action
    async def set_generator(self, generator) -> None:
        self._generator = generator
        log_action(self.logger, "set_generator", {"generator": str(generator)}, {"ok": True})

    @action
    async def set_refiner(self, refiner) -> None:
        self._refiner = refiner
        # Push current topic immediately so the refiner is domain-aware.
        if self._topic and hasattr(refiner, "set_topic"):
            try:
                await refiner.set_topic(self._topic)
            except Exception:
                pass
        log_action(self.logger, "set_refiner", {"refiner": str(refiner)}, {"ok": True})

    @action
    async def add_reviewer(self, reviewer) -> None:
        self._reviewers.append(reviewer)
        # Push current topic immediately so the reviewer is domain-aware.
        if self._topic and hasattr(reviewer, "set_topic"):
            try:
                await reviewer.set_topic(self._topic)
            except Exception:
                pass
        log_action(
            self.logger,
            "add_reviewer",
            {"reviewer": str(reviewer), "total": len(self._reviewers)},
            {"ok": True},
        )

    @action
    async def set_incoming(self, tournament) -> None:
        """Upstream staging tournament (None = this team generates from scratch)."""
        self._incoming_tournament = tournament
        log_action(
            self.logger, "set_incoming",
            {"incoming": str(tournament) if tournament else "None"},
            {"ok": True},
        )

    @action
    async def set_outgoing(self, tournament) -> None:
        """Global tournament — all validated hypotheses are pushed here."""
        self._outgoing_tournament = tournament
        log_action(
            self.logger, "set_outgoing",
            {"outgoing": str(tournament)},
            {"ok": True},
        )

    @action
    async def set_chain_outgoing(self, tournament) -> None:
        """Staging tournament for the next team (non-final teams only).

        When set, validated hypotheses are pushed to BOTH the global tournament
        (set_outgoing) AND this staging tournament so the next team can pick
        them up and refine them further.
        """
        self._chain_outgoing = tournament
        log_action(
            self.logger, "set_chain_outgoing",
            {"chain_outgoing": str(tournament)},
            {"ok": True},
        )

    @action
    async def is_done(self) -> bool:
        """Returns True once this team has completed all its cycles."""
        return self._done

    @action
    async def resubmit_with_poc_feedback(
        self,
        hyp: dict[str, Any],
        poc_feedback: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Re-run the review-refine loop on a hypothesis that received an INCONCLUSIVE
        PoC verdict.

        The PoC feedback (verdict, interpretation, stdout/stderr snippets, next_steps)
        is embedded into the hypothesis under ``meta.poc_feedback`` so reviewers and the
        refiner can see the experimental evidence when reassessing the hypothesis.

        Returns a status dict ``{"status": "resubmitted" | "no_reviewers", "hyp_id": ...}``.
        """
        enriched = dict(hyp)
        meta = dict(enriched.get("meta") or {})
        meta["poc_feedback"] = poc_feedback
        enriched["meta"] = meta

        hyp_id = enriched.get("id", "unknown")
        log_action(
            self.logger,
            "poc_resubmit",
            {"hyp_id": hyp_id, "team": self._team_name,
             "verdict": poc_feedback.get("verdict")},
            {"ok": True},
        )
        print(
            f"[{self._team_name}] PoC INCONCLUSIVE for hyp={hyp_id} — "
            "resubmitting to review-refine loop with PoC feedback.",
            flush=True,
        )
        if not self._reviewers:
            return {"status": "no_reviewers", "hyp_id": hyp_id}

        await self._run_review_refine_loop(enriched)
        return {"status": "resubmitted", "hyp_id": hyp_id}

    @action
    async def get_stats(self) -> dict[str, Any]:
        return {
            "team": self._team_name,
            "validated": self._total_validated,
            "discarded": self._total_discarded,
            "refined": self._total_refined,
            "chained": self._chain_outgoing is not None,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_hypotheses(self) -> list[dict[str, Any]]:
        """Generate or pull hypotheses for this cycle."""
        if self._incoming_tournament is not None:
            # Downstream team: pull validated hypotheses from upstream staging
            try:
                pairs = await self._incoming_tournament.get_all_hypotheses()
                return [hyp for _, hyp in pairs] if pairs else []
            except Exception as e:
                self.logger.error("incoming_pull_failed", extra={"error": repr(e)})
                return []

        if self._generator is None:
            self.logger.warning("no_generator_no_incoming", extra={})
            return []

        try:
            await self._generator.propose_hypotheses(self._n_hypotheses)
            ideas = await self._generator.get_ideas()
            return list(ideas) if ideas else []
        except Exception as e:
            self.logger.error("generate_failed", extra={"error": repr(e)})
            return []

    async def _collect_reviews(
        self, hyp_id: str, hyp: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], float]:
        """Ask all reviewers for critiques; return (critiques, avg_score)."""
        critiques: list[dict[str, Any]] = []
        for reviewer in self._reviewers:
            try:
                review = await reviewer.review_one(hyp_id, hyp)
                crit = review.get("critique", {})
                critiques.append(crit)
            except Exception as e:
                self.logger.error(
                    "review_one_failed",
                    extra={"reviewer": str(reviewer), "hyp_id": hyp_id, "error": repr(e)},
                )
        scores = [
            float(c["score"])
            for c in critiques
            if isinstance(c.get("score"), (int, float))
        ]
        avg = _avg(scores)
        return critiques, avg

    async def _push_validated(self, hyp: dict[str, Any], avg_score: float) -> None:
        """Push a validated hypothesis to the global tournament (and staging if chained)."""
        if self._outgoing_tournament is None:
            self.logger.warning("no_outgoing_tournament", extra={})
            return

        # Embed score and originating team so provenance is queryable from the tournament.
        hyp_with_score = {**hyp, "score": avg_score, "validated_by_team": self._team_name}

        try:
            await self._outgoing_tournament.add(hyp_with_score)
            self._total_validated += 1
            log_action(
                self.logger,
                "hypothesis_validated",
                {"hyp_id": hyp.get("id"), "team": self._team_name},
                {"avg_score": round(avg_score, 3), "total_validated": self._total_validated},
            )
        except Exception as e:
            self.logger.error(
                "push_validated_failed",
                extra={"hyp_id": hyp.get("id"), "error": repr(e)},
            )
            return

        # Also feed the next team's staging tournament (non-final teams only).
        if self._chain_outgoing is not None:
            try:
                await self._chain_outgoing.add(hyp_with_score)
                log_action(
                    self.logger,
                    "hypothesis_chained",
                    {"hyp_id": hyp.get("id"), "team": self._team_name},
                    {"avg_score": round(avg_score, 3)},
                )
            except Exception as e:
                self.logger.error(
                    "push_chain_failed",
                    extra={"hyp_id": hyp.get("id"), "error": repr(e)},
                )

    async def _refine_hypothesis(
        self,
        hyp: dict[str, Any],
        merged_critique: dict[str, Any],
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Call the refiner agent with current critique and full review history."""
        if self._refiner is None:
            return hyp
        try:
            improved = await self._refiner.refine(hyp, merged_critique, history=history)
            self._total_refined += 1
            return improved
        except Exception as e:
            self.logger.error(
                "refine_call_failed",
                extra={"hyp_id": hyp.get("id"), "error": repr(e)},
            )
            return hyp

    async def _run_review_refine_loop(self, hyp: dict[str, Any]) -> None:
        """
        Run the review-refine cycle for a single hypothesis.

        The hypothesis is refined up to max_refinement_rounds times.
        Each refinement call receives both the current round's merged critique
        AND the full history of all previous rounds so the LLM can identify
        recurring issues and track the score trajectory.

        Once the average score reaches score_threshold the hypothesis is
        forwarded to the outgoing tournament.  If all rounds are exhausted
        it is forwarded only when avg_score >= min_pass_score, otherwise
        it is discarded.
        """
        hyp_id = hyp.get("id") or uuid.uuid4().hex[:8]
        hyp = dict(hyp)
        hyp.setdefault("id", hyp_id)

        if not self._reviewers:
            # No reviewers configured — push directly
            await self._push_validated(hyp, avg_score=0.0)
            return

        # Track the best version seen so far; refinement can worsen the score
        # so we always push (and refine from) the highest-scoring version.
        best_hyp: dict[str, Any] = hyp
        best_score: float = -1.0

        # Accumulates one entry per completed review round:
        # {"round": int, "avg_score": float, "merged": dict, "critiques": list[dict]}
        review_history: list[dict[str, Any]] = []

        for round_n in range(self._max_rounds + 1):
            critiques, avg_score = await self._collect_reviews(hyp_id, hyp)
            merged = _merge_critiques(critiques)

            # Keep the best-scoring version
            if avg_score > best_score:
                best_score = avg_score
                best_hyp = hyp
            elif avg_score < best_score:
                print(
                    f"[{self._team_name}] hyp={hyp_id} score regressed "
                    f"{avg_score:.3f} < best={best_score:.3f} — keeping best version",
                    flush=True,
                )
                log_action(
                    self.logger,
                    "score_regression",
                    {"hyp_id": hyp_id, "round": round_n, "team": self._team_name},
                    {"avg_score": round(avg_score, 3), "best_score": round(best_score, 3)},
                )

            # Record this round before deciding what to do
            review_history.append({
                "round":      round_n,
                "avg_score":  round(avg_score, 3),
                "best_score": round(best_score, 3),
                "merged":     merged,
                "critiques":  critiques,
            })

            log_action(
                self.logger,
                "review_round",
                {"hyp_id": hyp_id, "round": round_n, "team": self._team_name},
                {
                    "avg_score":   round(avg_score, 3),
                    "best_score":  round(best_score, 3),
                    "threshold":   self._score_threshold,
                    "reviewers":   len(critiques),
                    "history_len": len(review_history),
                },
            )

            if best_score >= self._score_threshold:
                print(
                    f"[{self._team_name}] hyp={hyp_id} VALIDATED "
                    f"best_score={best_score:.3f} >= {self._score_threshold} "
                    f"after {round_n} refinement(s)",
                    flush=True,
                )
                await self._push_validated(best_hyp, best_score)
                return

            if round_n < self._max_rounds:
                # Refine from the best-known version so regression doesn't compound.
                print(
                    f"[{self._team_name}] hyp={hyp_id} best_score={best_score:.3f} "
                    f"< {self._score_threshold} "
                    f"— refining from best (round {round_n + 1}/{self._max_rounds})",
                    flush=True,
                )
                hyp = await self._refine_hypothesis(
                    best_hyp,
                    merged,
                    history=review_history,
                )
            else:
                # Max rounds exhausted — push the best version we have
                if best_score >= self._min_pass_score:
                    print(
                        f"[{self._team_name}] hyp={hyp_id} max rounds exhausted, "
                        f"best_score={best_score:.3f} >= min_pass={self._min_pass_score} — forwarding best",
                        flush=True,
                    )
                    await self._push_validated(best_hyp, best_score)
                else:
                    self._total_discarded += 1
                    print(
                        f"[{self._team_name}] hyp={hyp_id} DISCARDED "
                        f"best_score={best_score:.3f} < min_pass={self._min_pass_score}",
                        flush=True,
                    )
                    log_action(
                        self.logger,
                        "hypothesis_discarded",
                        {"hyp_id": hyp_id, "team": self._team_name},
                        {"best_score": round(best_score, 3), "total_discarded": self._total_discarded},
                    )

    # ------------------------------------------------------------------
    # Autonomous loop
    # ------------------------------------------------------------------

    @loop
    async def team_loop(self, shutdown: asyncio.Event) -> None:
        """
        Main team loop: generate/pull → review-refine → forward.

        Runs until the shutdown event is set.
        """
        if self._loop_start_delay > 0:
            print(
                f"[{self._team_name}] waiting {self._loop_start_delay}s before first cycle",
                flush=True,
            )
            await asyncio.sleep(self._loop_start_delay)

        cycle = 0    # counts productive cycles (hypotheses actually processed)
        attempt = 0  # counts all loop iterations including idle waits
        while not shutdown.is_set():
            attempt += 1
            print(
                f"[{self._team_name}] attempt {attempt} — "
                f"generating/pulling hypotheses (n={self._n_hypotheses})",
                flush=True,
            )

            try:
                hypotheses = await self._get_hypotheses()
            except Exception as e:
                print(f"[{self._team_name}] ERROR fetching hypotheses: {e}", flush=True)
                print(traceback.format_exc(), flush=True)
                await asyncio.sleep(self._loop_interval)
                continue

            if not hypotheses:
                print(
                    f"[{self._team_name}] no hypotheses available "
                    f"(attempt {attempt}), waiting…",
                    flush=True,
                )
                await asyncio.sleep(self._loop_interval)
                continue

            # We have real work — this counts as a productive cycle.
            cycle += 1
            print(
                f"[{self._team_name}] cycle {cycle}: "
                f"reviewing {len(hypotheses)} hypothesis/es",
                flush=True,
            )

            for hyp in hypotheses:
                if shutdown.is_set():
                    break
                try:
                    await self._run_review_refine_loop(hyp)
                except Exception as e:
                    print(
                        f"[{self._team_name}] ERROR in review-refine loop for "
                        f"hyp={hyp.get('id')}: {e}",
                        flush=True,
                    )
                    print(traceback.format_exc(), flush=True)

            log_action(
                self.logger,
                "team_cycle_done",
                {"cycle": cycle, "attempt": attempt, "team": self._team_name,
                 "processed": len(hypotheses)},
                {
                    "validated": self._total_validated,
                    "discarded": self._total_discarded,
                    "refined": self._total_refined,
                },
            )

            if self._max_cycles is not None and cycle >= self._max_cycles:
                print(
                    f"[{self._team_name}] max_cycles={self._max_cycles} reached "
                    f"after {attempt} attempt(s) — stopping.",
                    flush=True,
                )
                log_action(
                    self.logger,
                    "team_max_cycles_reached",
                    {"team": self._team_name, "cycle": cycle, "attempt": attempt},
                    {"max_cycles": self._max_cycles},
                )
                self._done = True
                return

            await asyncio.sleep(self._loop_interval)
