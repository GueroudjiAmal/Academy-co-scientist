# academy_coscientist/agents/datatypes.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any
import uuid


def _new_id() -> str:
    return str(uuid.uuid4())[:8]


@dataclass
class Hypothesis:
    id: str
    text: str
    origin: str
    metadata: Dict[str, Any] | None = field(default_factory=dict)
    elo: float = 1200.0
    history: List[str] = field(default_factory=list)


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
