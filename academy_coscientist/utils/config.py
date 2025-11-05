# academy_coscientist/utils/config.py
from __future__ import annotations

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
    return _CONFIG


def get_model(role: str, default: str | None = None) -> str:
    """Return a model string by logical role. Example roles:
    - "reasoning"
    - "writing"
    - "embedding"
    """
    models = _CONFIG.get('models', {}) if _CONFIG else {}
    if models.get(role):
        return str(models[role])
    # sensible fallbacks if not configured
    defaults = {
        'reasoning': 'gpt-5-reasoning',
        'writing': 'gpt-5o-mini',
        'embedding': 'text-embedding-3-small',
    }
    return default or defaults.get(role, 'gpt-5o-mini')


def get_path(key: str, default: str | None = None) -> str:
    """Return a filesystem path from config.paths.<key>."""
    paths = _CONFIG.get('paths', {}) if _CONFIG else {}
    return str(paths.get(key, default or ''))


def get_launch_param(key: str, default: Any = None) -> Any:
    """Return value from config.launch.<key>."""
    launch = _CONFIG.get('launch', {}) if _CONFIG else {}
    return launch.get(key, default)


def maybe_override(value: Any, override: Any | None) -> Any:
    """Helper: if CLI override is provided (not None), use it; else use config value."""
    return override if override is not None else value
