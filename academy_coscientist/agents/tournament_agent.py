# academy_coscientist/agents/tournament_agent.py
from __future__ import annotations

import math
import uuid
from typing import Any

from academy.agent import action
from academy.agent import Agent

from academy_coscientist.utils.utils_logging import log_action
from academy_coscientist.utils.utils_logging import make_struct_logger


class TournamentAgent(Agent):
    """Minimal, robust tournament that stores hypotheses, accepts scores, and produces a leaderboard."""

    def __init__(self) -> None:
        super().__init__()
        self.logger = make_struct_logger('TournamentAgent')
        self._hyps: dict[str, dict[str, Any]] = {}
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

    @action
    async def add(self, idea: dict[str, Any]) -> str:
        hyp = self._normalize_idea(idea)
        hid = idea.get('id') or self._assign_id()
        rec = {
            'id': hid,
            **hyp,
            'score': float(idea.get('score', 0.0) or 0.0),
            'confidence': float(idea.get('confidence', 0.0) or 0.0),
        }
        self._hyps[hid] = rec
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
    async def apply_reviews(self, scores: dict[str, dict[str, Any]]) -> None:
        await self.set_scores(scores)

    @action
    async def update_scores(self, scores: dict[str, dict[str, Any]]) -> None:
        await self.set_scores(scores)

    @action
    async def ingest_reviews(self, scores: dict[str, dict[str, Any]]) -> None:
        await self.set_scores(scores)

    @action
    async def add_scores(self, scores: dict[str, dict[str, Any]]) -> None:
        await self.set_scores(scores)

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
