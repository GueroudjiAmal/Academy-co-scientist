# academy_coscientist/agents/review_agent.py
from __future__ import annotations

import asyncio
import inspect
import traceback
from typing import Any

from academy.agent import action, loop
from academy.agent import Agent

from academy_coscientist.utils import utils_llm
from academy_coscientist.utils.utils_logging import log_action
from academy_coscientist.utils.utils_logging import make_struct_logger


async def _call_maybe_async(fn, *args, **kwargs):
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    return fn(*args, **kwargs)


async def _try_call(obj, name: str, variants: tuple[tuple[tuple, dict], ...]) -> tuple[bool, Any]:
    if not hasattr(obj, name):
        return False, None
    fn = getattr(obj, name)
    if not callable(fn):
        return False, None
    for args, kwargs in variants:
        try:
            res = await _call_maybe_async(fn, *args, **kwargs)
            return True, res
        except TypeError:
            continue
        except Exception:
            raise
    return False, None


class ReviewAgent(Agent):
    """Reviews hypotheses using vector-store context + an LLM critique.

    Actions:
      - set_vector_store(vs_handle)
      - set_tournament(handle)          # optional wire-up
      - review_all(tournament=None)
      - review_top(k: Optional[int]=None)  # if k None, infer from tournament (no magic numbers)
      - get_reviews()
    """

    def __init__(
        self,
        name: str = 'reviewer',
        loop_enabled: bool = False,
        loop_interval: float = 45.0,
        loop_start_delay: float = 0.0,
        rag_top_k: int = 0,
    ) -> None:
        super().__init__()
        self.logger = make_struct_logger('ReviewAgent')
        self.name = name
        self._team_name: str = ""
        self._topic: str = ""
        self._vector_store = None
        self._tournament = None
        self._reviews: dict[str, dict[str, Any]] = {}
        # Loop configuration
        self._loop_enabled = loop_enabled
        self._loop_interval = loop_interval
        self._loop_start_delay = loop_start_delay
        self._rag_top_k = max(1, rag_top_k)
        self.logger.debug('ReviewAgent init', extra={'agent_name': self.name})

    @action
    async def set_topic(self, topic: str) -> None:
        self._topic = str(topic)
        log_action(self.logger, 'set_topic', {'topic': topic}, {'ok': True})

    @action
    async def set_team_name(self, name: str) -> None:
        self._team_name = str(name)
        log_action(self.logger, 'set_team_name', {'team_name': name}, {'ok': True})

    @action
    async def set_vector_store(self, vector_store_handle) -> None:
        self._vector_store = vector_store_handle
        log_action(
            self.logger,
            'set_vector_store',
            {'vector_store': str(vector_store_handle)},
            {'ok': True},
        )

    @action
    async def set_tournament(self, tournament_handle) -> None:
        self._tournament = tournament_handle
        log_action(
            self.logger, 'set_tournament', {'tournament': str(tournament_handle)}, {'ok': True}
        )

    async def _fetch_all_hypotheses(self, tournament) -> list[tuple[str, dict[str, Any]]]:
        if tournament is None:
            return []

        ok, res = await _try_call(tournament, 'get_all_hypotheses', (((), {}),))
        if ok and isinstance(res, list):
            out = []
            for it in res:
                if (
                    isinstance(it, (list, tuple))
                    and len(it) == 2
                    and isinstance(it[0], str)
                    and isinstance(it[1], dict)
                ):
                    out.append((it[0], it[1]))
                elif isinstance(it, dict) and 'id' in it:
                    out.append((str(it['id']), it))
            if out:
                return out

        ok, res = await _try_call(tournament, 'get_top_hypotheses', (((999999,), {}), ((), {})))
        if ok and isinstance(res, list):
            out = []
            for it in res:
                if (
                    isinstance(it, (list, tuple))
                    and len(it) == 2
                    and isinstance(it[0], str)
                    and isinstance(it[1], dict)
                ):
                    out.append((it[0], it[1]))
                elif isinstance(it, dict) and 'id' in it:
                    out.append((str(it['id']), it))
            if out:
                return out

        ok, res = await _try_call(tournament, 'get_leaderboard', (((), {}),))
        if ok and isinstance(res, list):
            out = []
            for row in res:
                if (
                    isinstance(row, (list, tuple))
                    and len(row) == 3
                    and isinstance(row[0], str)
                    and isinstance(row[2], dict)
                ):
                    out.append((row[0], row[2]))
            if out:
                return out

        return []

    async def _fetch_top_hypotheses_local(
        self, tournament, k: int
    ) -> list[tuple[str, dict[str, Any]]]:
        if tournament is None or k <= 0:
            return []

        ok, res = await _try_call(tournament, 'get_top_hypotheses', (((k,), {}),))
        if ok and isinstance(res, list):
            out = []
            for it in res[:k]:
                if (
                    isinstance(it, (list, tuple))
                    and len(it) == 2
                    and isinstance(it[0], str)
                    and isinstance(it[1], dict)
                ):
                    out.append((it[0], it[1]))
                elif isinstance(it, dict) and 'id' in it:
                    out.append((str(it['id']), it))
            if out:
                return out[:k]

        ok, res = await _try_call(tournament, 'get_leaderboard', (((), {}),))
        if ok and isinstance(res, list):
            out = []
            for row in res[:k]:
                if (
                    isinstance(row, (list, tuple))
                    and len(row) == 3
                    and isinstance(row[0], str)
                    and isinstance(row[2], dict)
                ):
                    out.append((row[0], row[2]))
            if out:
                return out[:k]

        ok, res = await _try_call(tournament, 'get_all_hypotheses', (((), {}),))
        if ok and isinstance(res, list):
            out = []
            for it in res[:k]:
                if (
                    isinstance(it, (list, tuple))
                    and len(it) == 2
                    and isinstance(it[0], str)
                    and isinstance(it[1], dict)
                ):
                    out.append((it[0], it[1]))
                elif isinstance(it, dict) and 'id' in it:
                    out.append((str(it['id']), it))
            if out:
                return out[:k]

        return []

    async def _vector_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        vs = self._vector_store
        if vs is None:
            return []

        candidates = [
            ('search', ((query, top_k), {}), ((query,), {})),
            ('query', ((query, top_k), {}), ((query,), {})),
            ('find', ((query, top_k), {}), ((query,), {})),
            ('query_similar', ((query, top_k), {}), ((query,), {})),
        ]

        for method, *variants in candidates:
            ok, res = await _try_call(vs, method, tuple(variants))
            if ok and res is not None:
                if isinstance(res, list):
                    out = []
                    for item in res:
                        if isinstance(item, dict):
                            out.append(item)
                        elif isinstance(item, (list, tuple)) and len(item) == 2:
                            obj, score = item
                            if isinstance(obj, dict):
                                d = dict(obj)
                            else:
                                d = {'text': str(obj)}
                            try:
                                d.setdefault('score', float(score))
                            except Exception:
                                d.setdefault('score', 0.0)
                            out.append(d)
                        else:
                            out.append({'text': str(item)})
                    return out
                return [{'text': str(res)}]

        return []

    async def _review_single(self, hyp_id: str, hyp: dict[str, Any]) -> dict[str, Any]:
        title = hyp.get('title') or hyp.get('name') or hyp.get('summary') or f'Hypothesis {hyp_id}'
        desc = hyp.get('description') or hyp.get('detail') or hyp.get('text') or ''
        query = f'{title}\n\n{desc}'.strip() or title

        ctx = await self._vector_search(query, top_k=self._rag_top_k)

        try:
            if hasattr(utils_llm, 'review_hypothesis'):
                critique = await utils_llm.review_hypothesis(
                    hypothesis={'id': hyp_id, 'title': title, 'description': desc},
                    retrieved=ctx,
                    topic=self._topic,
                    context={
                        'agent': 'ReviewAgent',
                        'reviewer': self.name,
                        'team': self._team_name or None,
                    },
                )
            else:
                critique = {
                    'strengths': ['Well-motivated'] if len(desc) > 40 else ['Concise'],
                    'weaknesses': ['Needs more detail'] if len(desc) < 100 else [],
                    'risks': [],
                    'score': 0.6,
                    'confidence': 0.5,
                    'recommendation': 'revise',
                    'notes': 'utils_llm.review_hypothesis not found; heuristic fallback.',
                }
        except Exception as e:
            print(
                f"[ReviewAgent:{self.name}] LLM ERROR reviewing '{title}': {e!r}",
                flush=True,
            )
            import traceback as _tb
            print(_tb.format_exc(), flush=True)
            critique = {
                'strengths': [],
                'weaknesses': [f'LLM error: {e!r}'],
                'risks': [],
                'score': 0.0,
                'confidence': 0.0,
                'recommendation': 'defer',
                'notes': 'LLM call failed.',
            }

        result = {'id': hyp_id, 'title': title, 'context_used': ctx[:5], 'critique': critique}
        log_action(
            self.logger,
            'review_single',
            {
                'hypothesis_id': hyp_id,
                'title': title,
                'desc_len': len(desc),
                'retrieved_k': len(ctx),
            },
            {
                'score': critique.get('score'),
                'confidence': critique.get('confidence'),
                'recommendation': critique.get('recommendation'),
                'reasoning': critique.get('reasoning') or '',
            },
        )
        return result

    @action
    async def review_all(self, tournament=None) -> None:
        if tournament is None:
            tournament = self._tournament
        try:
            hyps = await self._fetch_all_hypotheses(tournament)
        except Exception as e:
            log_action(
                self.logger, 'review_all', {'tournament': str(tournament)}, {'error': repr(e)}
            )
            return

        outputs: list[dict[str, Any]] = []
        for hid, h in hyps:
            rev = await self._review_single(hid, h)
            self._reviews[hid] = rev
            outputs.append({'id': hid, 'score': rev['critique'].get('score')})

        log_action(self.logger, 'review_all', {'count': len(hyps)}, {'reviewed': len(outputs)})

    @action
    async def review_top(self, k: int | None = None) -> None:
        tournament = self._tournament

        inferred_k = None
        if k is None and tournament is not None:
            ok, res = await _try_call(tournament, 'get_top_hypotheses', (((999999,), {}), ((), {})))
            if ok and isinstance(res, list):
                inferred_k = len(res)
            else:
                ok, res = await _try_call(tournament, 'get_leaderboard', (((), {}),))
                if ok and isinstance(res, list):
                    inferred_k = len(res)
                else:
                    ok, res = await _try_call(tournament, 'get_all_hypotheses', (((), {}),))
                    if ok and isinstance(res, list):
                        inferred_k = len(res)

        use_k = int(k if k is not None else (inferred_k or 0))
        try:
            top = await self._fetch_top_hypotheses_local(tournament, use_k) if use_k > 0 else []
        except Exception as e:
            log_action(
                self.logger, 'review_top', {'k': k, 'inferred_k': inferred_k}, {'error': repr(e)}
            )
            return

        outputs: list[dict[str, Any]] = []
        for hid, h in top:
            rev = await self._review_single(hid, h)
            self._reviews[hid] = rev
            outputs.append({'id': hid, 'score': rev['critique'].get('score')})

        log_action(
            self.logger,
            'review_top',
            {'k': k, 'inferred_k': inferred_k, 'found': len(top)},
            {'reviewed': len(outputs)},
        )

    @action
    async def review_one(self, hyp_id: str, hyp: dict[str, Any]) -> dict[str, Any]:
        """Review a single hypothesis by id and payload; return the full review record.

        Used by TeamCoordinatorAgent for per-hypothesis iterative review-refine loops.
        The result is also stored in self._reviews for later retrieval via get_reviews().
        """
        result = await self._review_single(hyp_id, hyp)
        self._reviews[hyp_id] = result
        return result

    @action
    async def get_reviews(self) -> dict[str, dict[str, Any]]:
        log_action(self.logger, 'get_reviews', {'requested': True}, {'count': len(self._reviews)})
        return dict(self._reviews)

    async def _push_scores_to_tournament(self) -> int:
        """
        Push the scores from self._reviews into the tournament.

        In the autonomous simulator there is no SupervisorAgent calling
        _push_reviews_to_tournament(), so the ReviewAgent does it itself
        after each review cycle.

        Returns the number of scores pushed.
        """
        if not self._tournament or not self._reviews:
            return 0

        payload: dict[str, dict[str, Any]] = {}
        for hid, rec in self._reviews.items():
            crit = rec.get('critique', {})
            score = crit.get('score')
            confidence = crit.get('confidence')
            title = rec.get('title')
            payload[hid] = {
                'score': float(score) if isinstance(score, (int, float)) else 0.0,
                'confidence': float(confidence) if isinstance(confidence, (int, float)) else 0.0,
                'title': title,
            }

        try:
            await self._tournament.set_scores(payload)
            log_action(
                self.logger,
                'scores_pushed_to_tournament',
                {'count': len(payload)},
                {'ok': True},
            )
            print(
                f"[ReviewAgent:{self.name}] Pushed {len(payload)} scores to tournament.",
                flush=True,
            )
            return len(payload)
        except Exception as e:
            print(
                f"[ReviewAgent:{self.name}] WARNING: could not push scores — {e!r}",
                flush=True,
            )
            return 0

    # ------------------- autonomous loop -------------------

    @loop
    async def review_loop(self, shutdown: asyncio.Event) -> None:
        """Autonomous loop: reviews all hypotheses then pushes scores to tournament."""
        if not self._loop_enabled:
            return
        if self._loop_start_delay > 0:
            await asyncio.sleep(self._loop_start_delay)
        cycle = 0
        while not shutdown.is_set():
            if not self._tournament:
                print(f"[ReviewAgent:{self.name}] WARNING: no tournament set, skipping cycle.", flush=True)
            else:
                cycle += 1
                print(f"[ReviewAgent:{self.name}] Cycle {cycle}: reviewing hypotheses...", flush=True)
                try:
                    await self.review_all()
                    pushed = await self._push_scores_to_tournament()
                    print(
                        f"[ReviewAgent:{self.name}] Cycle {cycle}: OK — "
                        f"{len(self._reviews)} reviews, {pushed} scores pushed to tournament.",
                        flush=True,
                    )
                    log_action(
                        self.logger,
                        "review_loop_cycle",
                        {"reviewer": self.name, "cycle": cycle},
                        {"ok": True, "total_reviews": len(self._reviews), "scores_pushed": pushed},
                    )
                except Exception as e:
                    print(f"[ReviewAgent:{self.name}] ERROR in cycle {cycle}: {e}", flush=True)
                    print(traceback.format_exc(), flush=True)
                    self.logger.error("review_loop_error", extra={"error": repr(e), "cycle": cycle})
            await asyncio.sleep(self._loop_interval)
