# academy_coscientist/utils/config.py
from __future__ import annotations

import os
from typing import Any

import yaml

_CONFIG: dict[str, Any] = {}
_CONFIG_PATH: str | None = None


def load_config(path: str) -> None:
    """Load YAML config into memory."""
    global _CONFIG, _CONFIG_PATH
    _CONFIG_PATH = path
    with open(path, encoding='utf-8') as f:
        _CONFIG = yaml.safe_load(f) or {}


def get_config() -> dict[str, Any]:
    global _CONFIG, _CONFIG_PATH
    if not _CONFIG:
        path = os.environ.get("ACADEMY_CONFIG_PATH", "")
        if path:
            _CONFIG_PATH = path
            with open(path, encoding="utf-8") as _f:
                _CONFIG = yaml.safe_load(_f) or {}
    return _CONFIG


def get_model(role: str, default: str | None = None) -> str:
    """Return a model string by logical role. Example roles:
    - "reasoning"
    - "writing"
    - "embedding"
    """
    models = get_config().get('models', {})
    if models.get(role):
        return str(models[role])
    # sensible fallbacks if not configured
    defaults = {
        # generic fallbacks
        'reasoning':     "o4-mini",
        'writing':       "gpt-4o",
        # per-agent-type keys
        'generation':    "o4-mini",
        'review':        "o4-mini",
        'refinement':    "o4-mini",
        'poc':           "claude-opus-4-6",
        'poc_codegen':   "claude-opus-4-6",
        'poc_fix':       "claude-opus-4-6",
        'poc_interpret': "claude-sonnet-4-6",
        'report':        "claude-sonnet-4-6",
        'commentary':    "gpt-4o-mini",
        'embedding':     "local-all-MiniLM-L6-v2",
        'argo':          "gpt-4o",
    }
    return default or defaults.get(role, 'gpt-4o-mini')


def get_temperature(role: str, default: float | None = None) -> float | None:
    """Return a temperature float by logical role. Example roles:
    - "writing"     (creative text, brainstorming)
    - "commentary"  (agent self-commentary)
    - "summary"     (summarisation tasks)
    Returns None for roles that don't use temperature (e.g. reasoning models).
    """
    temperatures = get_config().get('temperatures', {})
    if role in temperatures:
        val = temperatures[role]
        return None if val is None else float(val)
    defaults: dict[str, float | None] = {
        'generation':   0.7,   # HypothesisGenerationAgent — brainstorming
        'review':       0.1,   # ReviewAgent — scoring and critiquing
        'refinement':   0.3,   # HypothesisRefinerAgent — refining
        'poc_codegen':  1.0,   # PoC code generation
        'poc_fix':      0.3,   # PoC error fixing
        'poc_interpret': 0.2,  # PoC result interpretation
        'report':       0.3,   # PDF report narratives
        'commentary':   0.5,   # Agent self-commentary
        # legacy aliases kept for backwards compat
        'writing':      0.7,
        'summary':      0.4,
    }
    return default if default is not None else defaults.get(role, 0.7)


def get_path(key: str, default: str | None = None) -> str:
    """Return a filesystem path from config.paths.<key>."""
    paths = get_config().get('paths', {})
    return str(paths.get(key, default or ''))


def get_launch_param(key: str, default: Any = None) -> Any:
    """Return value from config.launch.<key>."""
    launch = get_config().get('launch', {})
    return launch.get(key, default)


def maybe_override(value: Any, override: Any | None) -> Any:
    """Helper: if CLI override is provided (not None), use it; else use config value."""
    return override if override is not None else value
