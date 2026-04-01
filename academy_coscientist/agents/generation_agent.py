# academy_coscientist/agents/generation_agent.py
from __future__ import annotations

import asyncio
import re
import threading
import traceback
from typing import Any, List, Optional

from academy.agent import Agent, action, loop
from academy_coscientist.utils import utils_llm
from academy_coscientist.utils.utils_logging import log_action, make_struct_logger

# ---------------------------------------------------------------------------
# Module-level shared title registry — written and read by ALL generator
# instances in this process so no two agents ever propose the same hypothesis.
#
# _GLOBAL_TITLES_NORM  : set of normalised titles for fast O(1) dedup checks
# _GLOBAL_TITLES_LIST  : ordered list of original-cased titles fed to the LLM prompt
# ---------------------------------------------------------------------------
_GLOBAL_TITLES_NORM: set[str] = set()
_GLOBAL_TITLES_LIST: list[str] = []
_GLOBAL_TITLES_LOCK = threading.Lock()


def _norm(title: str) -> str:
    """Lowercase + strip non-alphanumeric for fuzzy dedup."""
    return re.sub(r'[^a-z0-9 ]', '', title.lower()).strip()


def _register_titles(titles: list[str]) -> None:
    """Add titles to the global registry (idempotent — ignores exact duplicates)."""
    with _GLOBAL_TITLES_LOCK:
        for t in titles:
            n = _norm(t)
            if n and n not in _GLOBAL_TITLES_NORM:
                _GLOBAL_TITLES_NORM.add(n)
                _GLOBAL_TITLES_LIST.append(t)


def _get_all_known_titles() -> list[str]:
    """Return all original-cased titles seen by any generator instance."""
    with _GLOBAL_TITLES_LOCK:
        return list(_GLOBAL_TITLES_LIST)


def _is_known(title: str) -> bool:
    """True if a normalised form of `title` is already registered."""
    with _GLOBAL_TITLES_LOCK:
        return _norm(title) in _GLOBAL_TITLES_NORM


class HypothesisGenerationAgent(Agent):
    """
    Agent responsible for proposing candidate ideas / hypotheses for a topic.

    Integrates optionally with a VectorDB agent (FAISS) to provide contextual abstracts.

    Autonomous mode: set loop_enabled=True in constructor to activate the @loop.
    The agent will then periodically propose hypotheses without external orchestration.
    """

    def __init__(
        self,
        loop_enabled: bool = False,
        loop_interval: float = 60.0,
        loop_start_delay: float = 0.0,
        loop_n_hypotheses: int = 3,
        rag_top_k: int = 5,
    ) -> None:
        super().__init__()
        self.logger = make_struct_logger("HypothesisGenerationAgent")
        self._topic: Optional[str] = None
        self._tournament = None
        self._vectordb = None
        self._ideas: List[dict[str, Any]] = []
        self._team_name: str = ""
        # Per-instance list kept only for the add_known_titles action interface;
        # actual dedup uses the module-level _GLOBAL_TITLES shared across all instances.
        self._generated_titles: List[str] = []
        # Loop configuration
        self._loop_enabled = loop_enabled
        self._loop_interval = loop_interval
        self._loop_start_delay = loop_start_delay
        self._loop_n_hypotheses = loop_n_hypotheses
        self._rag_top_k = max(1, rag_top_k)

    # ------------------- configuration -------------------

    @action
    async def set_topic(self, topic: str) -> None:
        self._topic = topic
        log_action(self.logger, "set_topic", {"topic": topic}, {"ok": True})

    @action
    async def set_team_name(self, name: str) -> None:
        self._team_name = str(name)
        log_action(self.logger, "set_team_name", {"team_name": name}, {"ok": True})

    @action
    async def add_known_titles(self, titles: list[str]) -> None:
        """Seed the shared global registry with externally known titles."""
        before = len(_GLOBAL_TITLES_LIST)
        _register_titles(titles)
        added = len(_GLOBAL_TITLES_LIST) - before
        log_action(self.logger, "add_known_titles", {"count": len(titles)}, {"added": added})

    @action
    async def set_tournament(self, tournament) -> None:
        self._tournament = tournament
        log_action(self.logger, "set_tournament", {"tournament": str(tournament)}, {"ok": True})

    @action
    async def set_vectordb(self, vectordb_agent) -> None:
        self._vectordb = vectordb_agent
        log_action(self.logger, "set_vectordb", {"vectordb": str(vectordb_agent)}, {"ok": True})

    # ------------------- main behavior -------------------

    @action
    async def propose_hypotheses(self, n: int | None = None) -> None:
        if not self._topic:
            raise RuntimeError("HypothesisGenerationAgent: topic not set")
        if n is None:
            raise RuntimeError("propose_hypotheses requires 'n'")

        topic = self._topic
        rag_context = ""

        # --- Query vector DB for relevant abstracts ---
        if self._vectordb:
            try:
                abstracts = await self._vectordb.query_texts(topic, k=self._rag_top_k)
                if abstracts:
                    rag_context = "\n\n".join(
                        f"[{i+1}] {a}" for i, a in enumerate(abstracts[:5])
                    )
                    self.logger.info(
                        "vectordb_context_retrieved",
                        extra={"topic": topic, "n_docs": len(abstracts)},
                    )
            except Exception as e:
                self.logger.warning(
                    "vectordb_query_failed",
                    extra={"error": repr(e), "topic": topic},
                )

        # --- Combine topic and abstracts for reasoning ---
        if rag_context:
            topic_for_llm = (
                f"{topic}\n\n"
                "Relevant research abstracts:\n"
                f"{rag_context}\n\n"
                "Formulate diverse, falsifiable hypotheses informed by these abstracts."
            )
        else:
            topic_for_llm = topic

        # --- Build exclude list from global registry (all agents, all cycles) ---
        # _get_all_known_titles() returns original-cased titles from every generator
        # instance in this process. Also pull any tournament titles not yet registered.
        exclude: list[str] = _get_all_known_titles()
        registered_norm: set[str] = {_norm(t) for t in exclude}

        if self._tournament:
            try:
                board = await self._tournament.get_leaderboard()
                for _, _, payload in board:
                    t = payload.get("title", "") if isinstance(payload, dict) else ""
                    if t and _norm(t) not in registered_norm:
                        exclude.append(t)
                        registered_norm.add(_norm(t))
            except Exception as e:
                self.logger.warning("tournament_title_fetch_failed", extra={"error": repr(e)})

        async def _run_llm():
            return await utils_llm.brainstorm_hypotheses(
                topic=topic_for_llm,
                n=int(n),
                context={
                    "agent": "HypothesisGenerationAgent",
                    "action": "brainstorm",
                    "team": self._team_name or None,
                    "instance_id": getattr(self, "instance_id", None),
                    "audit_path": getattr(self, "audit_path", None),
                },
                exclude_titles=exclude if exclude else None,
            )

        try:
            ideas = await _run_llm()
        except Exception as e:
            self.logger.exception(
                "brainstorm_failed",
                extra={"error": repr(e), "topic": topic_for_llm, "n_requested": n},
            )
            raise

        if not isinstance(ideas, list):
            raise RuntimeError(f"LLM brainstorm returned non-list: {type(ideas)!r}")

        self._ideas = ideas

        # Record titles in global shared registry AND per-instance list.
        new_titles = [idea.get("title", "") for idea in ideas if idea.get("title")]
        _register_titles(new_titles)
        existing = set(self._generated_titles)
        for t in new_titles:
            if t not in existing:
                self._generated_titles.append(t)
                existing.add(t)
        if len(self._generated_titles) > 120:
            self._generated_titles = self._generated_titles[-120:]

        added = 0

        # --- Push generated ideas into the tournament ---
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
            {"topic": topic, "n_requested": n, "rag_used": bool(rag_context)},
            {"n_generated": len(self._ideas), "n_added_to_tournament": added},
        )

    # ------------------- accessors -------------------

    @action
    async def get_ideas(self) -> list[dict[str, Any]]:
        return list(self._ideas)

    # ------------------- autonomous loop -------------------

    @loop
    async def generation_loop(self, shutdown: asyncio.Event) -> None:
        """Autonomous loop: periodically proposes hypotheses when loop_enabled=True."""
        if not self._loop_enabled:
            return
        if self._loop_start_delay > 0:
            print(f"[GenerationAgent] Waiting {self._loop_start_delay}s before first cycle...", flush=True)
            await asyncio.sleep(self._loop_start_delay)
        cycle = 0
        while not shutdown.is_set():
            if not self._topic:
                print("[GenerationAgent] WARNING: no topic set, skipping cycle.", flush=True)
            elif not self._tournament:
                print("[GenerationAgent] WARNING: no tournament handle set, skipping cycle.", flush=True)
            else:
                cycle += 1
                print(f"[GenerationAgent] Cycle {cycle}: generating {self._loop_n_hypotheses} hypotheses for {self._topic!r}...", flush=True)
                try:
                    await self.propose_hypotheses(self._loop_n_hypotheses)
                    print(f"[GenerationAgent] Cycle {cycle}: OK — {len(self._ideas)} hypotheses in pool.", flush=True)
                    log_action(
                        self.logger,
                        "generation_loop_cycle",
                        {"topic": self._topic, "n": self._loop_n_hypotheses, "cycle": cycle},
                        {"ok": True, "total_ideas": len(self._ideas)},
                    )
                except Exception as e:
                    print(f"[GenerationAgent] ERROR in cycle {cycle}: {e}", flush=True)
                    print(traceback.format_exc(), flush=True)
                    self.logger.error("generation_loop_error", extra={"error": repr(e), "cycle": cycle})
            await asyncio.sleep(self._loop_interval)
