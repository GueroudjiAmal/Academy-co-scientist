# academy_coscientist/agents/tournament_agent.py
from __future__ import annotations

import asyncio
import math
import re
import traceback
import uuid
from typing import Any

from academy.agent import action, loop
from academy.agent import Agent

from academy_coscientist.utils.utils_logging import log_action
from academy_coscientist.utils.utils_logging import make_struct_logger


class TournamentAgent(Agent):
    """Minimal, robust tournament that stores hypotheses, accepts scores, and produces a leaderboard."""

    def __init__(
        self,
        loop_enabled: bool = False,
        loop_interval: float = 30.0,
        loop_start_delay: float = 0.0,
        max_cycles: int | None = None,
    ) -> None:
        super().__init__()
        self.logger = make_struct_logger('TournamentAgent')
        self._hyps: dict[str, dict[str, Any]] = {}
        # title-norm → existing hid, for deduplication on insert
        self._title_index: dict[str, str] = {}
        # Loop configuration
        self._loop_enabled = loop_enabled
        self._loop_interval = loop_interval
        self._loop_start_delay = loop_start_delay
        self._max_cycles: int | None = int(max_cycles) if max_cycles is not None else None
        self.logger.debug('TournamentAgent init', extra={'count': 0})

    def _normalize_idea(self, idea: dict[str, Any]) -> dict[str, Any]:
        title = idea.get('title') or idea.get('name') or 'Untitled'
        desc = idea.get('description') or idea.get('text') or ''
        meta = {
            k: v
            for k, v in idea.items()
            if k not in ('title', 'name', 'description', 'text', 'score', 'confidence')
        }
        return {'title': str(title), 'description': str(desc), 'meta': meta}

    def _assign_id(self) -> str:
        return uuid.uuid4().hex[:12]

    @staticmethod
    def _norm_title(title: str) -> str:
        return re.sub(r'[^a-z0-9 ]', '', title.lower()).strip()

    @action
    async def add(self, idea: dict[str, Any]) -> str:
        hyp = self._normalize_idea(idea)
        title_norm = self._norm_title(hyp['title'])
        incoming_score = float(idea.get('score', 0.0) or 0.0)
        incoming_conf  = float(idea.get('confidence', 0.0) or 0.0)

        # Deduplicate by title: if an identical hypothesis already exists, keep
        # whichever version has the higher score rather than adding a duplicate.
        if title_norm in self._title_index:
            existing_hid = self._title_index[title_norm]
            existing_rec = self._hyps[existing_hid]
            if incoming_score > existing_rec.get('score', 0.0):
                existing_rec['score'] = incoming_score
                existing_rec['confidence'] = incoming_conf
                log_action(self.logger, 'add_dedup_update',
                           {'id': existing_hid, 'title': hyp['title']},
                           {'new_score': incoming_score})
            else:
                log_action(self.logger, 'add_dedup_skip',
                           {'title': hyp['title']},
                           {'kept_id': existing_hid})
            return existing_hid

        hid = idea.get('id') or self._assign_id()
        rec = {
            'id': hid,
            **hyp,
            'score': incoming_score,
            'confidence': incoming_conf,
        }
        self._hyps[hid] = rec
        self._title_index[title_norm] = hid
        log_action(self.logger, 'add', {'id': hid}, {'count': len(self._hyps)})
        return hid

    @action
    async def add_hypotheses(self, ideas: list[dict[str, Any]]) -> list[str]:
        ids: list[str] = []
        for idea in ideas or []:
            hid = await self.add(idea)
            ids.append(hid)
        log_action(
            self.logger, 'add_hypotheses', {'n': len(ideas or [])}, {'total': len(self._hyps)}
        )
        return ids

    @action
    async def get_all_hypotheses(self) -> list[tuple[str, dict[str, Any]]]:
        out = [
            (hid, {k: v for k, v in rec.items() if k != 'id'}) for hid, rec in self._hyps.items()
        ]
        log_action(self.logger, 'get_all_hypotheses', {'requested': True}, {'count': len(out)})
        return out

    @action
    async def get_top_hypotheses(self, k: int = 0) -> list[tuple[str, dict[str, Any]]]:
        rows = sorted(
            self._hyps.values(),
            key=lambda r: (r.get('score', 0.0), r.get('confidence', 0.0)),
            reverse=True,
        )
        if k and k > 0:
            rows = rows[:k]
        out = [(r['id'], {k: v for k, v in r.items() if k != 'id'}) for r in rows]
        log_action(self.logger, 'get_top_hypotheses', {'k': k}, {'returned': len(out)})
        return out

    @action
    async def set_scores(self, scores: dict[str, dict[str, Any]]) -> None:
        updated = 0
        missing = 0
        for hid, payload in (scores or {}).items():
            rec = self._hyps.get(hid)
            if not rec:
                missing += 1
                continue
            try:
                if 'score' in payload and payload['score'] is not None:
                    rec['score'] = float(payload['score'])
                if 'confidence' in payload and payload['confidence'] is not None:
                    rec['confidence'] = float(payload['confidence'])
                if payload.get('title'):
                    rec['title'] = str(payload['title'])
                updated += 1
            except Exception:
                continue
        log_action(
            self.logger,
            'set_scores',
            {'incoming': len(scores or {})},
            {'updated': updated, 'missing': missing},
        )

    @action
    async def run_tournament(self) -> None:
        for rec in self._hyps.values():
            s = rec.get('score', 0.0)
            if s is None or math.isnan(s):
                rec['score'] = 0.0
            else:
                rec['score'] = float(s)
        log_action(
            self.logger, 'run_tournament', {'op': 'normalize_scores'}, {'count': len(self._hyps)}
        )

    @action
    async def get_leaderboard(self) -> list[tuple[str, float, dict[str, Any]]]:
        rows = sorted(
            self._hyps.values(),
            key=lambda r: (r.get('score', 0.0), r.get('confidence', 0.0)),
            reverse=True,
        )
        out: list[tuple[str, float, dict[str, Any]]] = []
        for r in rows:
            idea = {k: v for k, v in r.items() if k not in ('id', 'score', 'confidence')}
            out.append((r['id'], float(r.get('score', 0.0) or 0.0), idea))
        log_action(self.logger, 'get_leaderboard', {'requested': True}, {'count': len(out)})
        return out

    # ------------------- autonomous loop -------------------

    @loop
    async def tournament_loop(self, shutdown: asyncio.Event) -> None:
        """Autonomous loop: periodically re-runs tournament ranking when loop_enabled=True."""
        if not self._loop_enabled:
            return
        if self._loop_start_delay > 0:
            await asyncio.sleep(self._loop_start_delay)
        cycle = 0
        while not shutdown.is_set():
            if self._hyps:
                cycle += 1
                try:
                    await self.run_tournament()
                    top = sorted(self._hyps.values(), key=lambda r: r.get('score', 0.0), reverse=True)
                    top_title = top[0].get('title', '?')[:60] if top else '—'
                    print(f"[TournamentAgent] Cycle {cycle}: ranked {len(self._hyps)} hypotheses. Top: {top_title!r}", flush=True)
                    log_action(
                        self.logger,
                        "tournament_loop_cycle",
                        {"n_hyps": len(self._hyps), "cycle": cycle},
                        {"ok": True},
                    )
                except Exception as e:
                    print(f"[TournamentAgent] ERROR in cycle {cycle}: {e}", flush=True)
                    print(traceback.format_exc(), flush=True)
                    self.logger.error("tournament_loop_error", extra={"error": repr(e), "cycle": cycle})
            if self._max_cycles is not None and cycle >= self._max_cycles:
                print(f"[TournamentAgent] max_cycles={self._max_cycles} reached — stopping.", flush=True)
                return
            await asyncio.sleep(self._loop_interval)
