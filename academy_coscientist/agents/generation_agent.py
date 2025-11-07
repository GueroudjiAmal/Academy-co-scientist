# academy_coscientist/agents/generation_agent.py
from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from academy.agent import Agent, action
from academy_coscientist.utils import utils_llm
from academy_coscientist.utils.utils_logging import log_action, make_struct_logger


class HypothesisGenerationAgent(Agent):
    """
    Agent responsible for proposing candidate ideas / hypotheses for a topic.

    Integrates optionally with a VectorDB agent (FAISS) to provide contextual abstracts.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logger = make_struct_logger("HypothesisGenerationAgent")
        self._topic: Optional[str] = None
        self._tournament = None
        self._vectordb = None
        self._ideas: List[dict[str, Any]] = []

    # ------------------- configuration -------------------

    @action
    async def set_topic(self, topic: str) -> None:
        self._topic = topic
        log_action(self.logger, "set_topic", {"topic": topic}, {"ok": True})

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
                results = await self._vectordb.query(topic, k=5)
                abstracts = [
                    r.get("abstract", "").strip()
                    for r in results
                    if isinstance(r, dict) and r.get("abstract")
                ]
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

        # --- Run the LLM safely in its own thread event loop ---
        async def _run_llm():
            return await utils_llm.brainstorm_hypotheses(
                topic=topic_for_llm,
                n=int(n),
                context={
                    "agent": "HypothesisGenerationAgent",
                    "action": "brainstorm",
                    "instance_id": getattr(self, "instance_id", None),
                    "audit_path": getattr(self, "audit_path", None),
                },
            )

        def _thread_entry():
            return asyncio.run(_run_llm())

        try:
            ideas = await asyncio.to_thread(_thread_entry)
        except Exception as e:
            self.logger.exception(
                "brainstorm_failed",
                extra={"error": repr(e), "topic": topic_for_llm, "n_requested": n},
            )
            raise

        if not isinstance(ideas, list):
            raise RuntimeError(f"LLM brainstorm returned non-list: {type(ideas)!r}")

        self._ideas = ideas
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
