# academy_coscientist/agents/datatypes.py
from __future__ import annotations

import uuid
from dataclasses import dataclass
from dataclasses import field
from typing import Any


def _new_id() -> str:
    return str(uuid.uuid4())[:8]


@dataclass
class Hypothesis:
    id: str
    text: str
    origin: str
    metadata: dict[str, Any] | None = field(default_factory=dict)
    elo: float = 1200.0
    history: list[str] = field(default_factory=list)


@dataclass
class Review:
    hypothesis_id: str
    reviewer: str
    strengths: str
    weaknesses: str
    risk: str
    novelty: str
    experimental_plan: str
    score: float


@dataclass
class MatchResult:
    winner_id: str
    loser_id: str
    reasoning: str


@dataclass
class LeaderboardEntry:
    hypothesis_id: str
    elo: float
    summary: str
