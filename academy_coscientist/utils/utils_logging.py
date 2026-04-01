# academy_coscientist/utils/utils_logging.py
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

__all__ = [
    'get_actions_log_path',
    'get_llm_audit_path',
    'get_run_dir',
    'get_run_id',
    'init_run_context',
    'log_action',
    'make_struct_logger',
    'record_llm_call',
    'register_action_hook',
    'unregister_action_hook',
    'register_llm_hook',
    'unregister_llm_hook',
]

# ---------------------------------------------------------------------------
# Plugin hooks — callables registered here are invoked after the built-in
# JSONL logging so external plugins (e.g. FlowCept) can observe every event
# without modifying agent code.
# ---------------------------------------------------------------------------

# Each element: callable(logger, action, input_payload, output_payload)
_ACTION_HOOKS: list = []
# Each element: callable(payload: dict)
_LLM_HOOKS: list = []


def register_action_hook(fn) -> None:
    """Register a callable invoked after every log_action() call."""
    if fn not in _ACTION_HOOKS:
        _ACTION_HOOKS.append(fn)


def unregister_action_hook(fn) -> None:
    """Remove a previously registered action hook (no-op if not found)."""
    try:
        _ACTION_HOOKS.remove(fn)
    except ValueError:
        pass


def register_llm_hook(fn) -> None:
    """Register a callable invoked after every record_llm_call() call."""
    if fn not in _LLM_HOOKS:
        _LLM_HOOKS.append(fn)


def unregister_llm_hook(fn) -> None:
    """Remove a previously registered LLM hook (no-op if not found)."""
    try:
        _LLM_HOOKS.remove(fn)
    except ValueError:
        pass

# -------------------- module-wide run context --------------------

_LOGS_ROOT_ENV = 'ACADEMY_LOGS_DIR'
_DEFAULT_LOGS_ROOT = 'runs'
_RUN_ID_ENV = 'ACADEMY_RUN_ID'
_RUN_DIR_ENV = 'ACADEMY_RUN_DIR'

_RUN_ID: str | None = None
_RUN_DIR: str | None = None
_LOGS_ROOT: str | None = None
_INIT_DONE: bool = False


def _timestamp() -> str:
    # UTC for reproducibility; format YYYYMMDD-HHMMSS
    return time.strftime('%Y%m%d-%H%M%S', time.gmtime())


def _short_uid(n: int = 8) -> str:
    return uuid.uuid4().hex[:n]


def get_run_id() -> str:
    if not _RUN_ID:
        raise RuntimeError('Run context not initialized. Call init_run_context() first.')
    return _RUN_ID


def get_run_dir() -> str:
    if not _RUN_DIR:
        raise RuntimeError('Run context not initialized. Call init_run_context() first.')
    return _RUN_DIR


def get_actions_log_path() -> str:
    return os.path.join(get_run_dir(), 'actions.jsonl')


def get_llm_audit_path() -> str:
    return os.path.join(get_run_dir(), 'llm_calls.jsonl')


# -------------------- initialization & logger setup --------------------


def init_run_context() -> tuple[str, str]:
    """Create a unique logs directory for this run and configure root logging.

    Returns:
        (run_id, run_dir)
    """
    global _RUN_ID, _RUN_DIR, _LOGS_ROOT, _INIT_DONE

    if _INIT_DONE and _RUN_ID and _RUN_DIR:
        # idempotent; return existing
        return _RUN_ID, _RUN_DIR

    # Subprocess workers inherit these env vars from the main process so all
    # agents log into the same run directory instead of creating their own.
    env_run_id = os.environ.get(_RUN_ID_ENV)
    env_run_dir = os.environ.get(_RUN_DIR_ENV)
    if env_run_id and env_run_dir:
        _RUN_ID = env_run_id
        _RUN_DIR = env_run_dir
        os.makedirs(_RUN_DIR, exist_ok=True)
    else:
        _LOGS_ROOT = os.environ.get(_LOGS_ROOT_ENV, _DEFAULT_LOGS_ROOT)
        os.makedirs(_LOGS_ROOT, exist_ok=True)
        _RUN_ID = f'{_timestamp()}-{_short_uid(8)}'
        _RUN_DIR = os.path.join(_LOGS_ROOT, _RUN_ID)
        os.makedirs(_RUN_DIR, exist_ok=True)
        # Publish for any child processes spawned later
        os.environ[_RUN_ID_ENV] = _RUN_ID
        os.environ[_RUN_DIR_ENV] = _RUN_DIR

    # Configure root logger only once
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if someone calls init twice
    if not any(
        isinstance(h, logging.FileHandler) and getattr(h, '_academy_handler', False)
        for h in root.handlers
    ):
        # File handler captures everything (DEBUG+)
        fh = logging.FileHandler(os.path.join(_RUN_DIR, 'events.log'), encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh._academy_handler = True  # type: ignore[attr-defined]
        fh.setFormatter(
            logging.Formatter(
                fmt='%(asctime)s %(levelname)s [%(name)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
            )
        )
        root.addHandler(fh)

    if not any(
        isinstance(h, logging.StreamHandler) and getattr(h, '_academy_handler', False)
        for h in root.handlers
    ):
        # Console handler is less noisy (INFO)
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh._academy_handler = True  # type: ignore[attr-defined]
        sh.setFormatter(
            logging.Formatter(
                fmt='%(levelname)s [%(name)s] %(message)s',
            )
        )
        root.addHandler(sh)

    # Touch jsonl logs so paths exist
    for p in (get_actions_log_path(), get_llm_audit_path()):
        if not os.path.exists(p):
            with open(p, 'w', encoding='utf-8') as f:
                pass

    _INIT_DONE = True

    # Print a single line for discoverability (to console)
    logging.getLogger('launcher').info('Run logs at %s (run id %s)', _RUN_DIR, _RUN_ID)

    return _RUN_ID, _RUN_DIR


def make_struct_logger(name: str) -> logging.Logger:
    """Get a logger by name. Handlers/formatters are attached to root in init_run_context()."""
    if not _INIT_DONE:
        # Ensure context exists even if someone forgot
        init_run_context()
    return logging.getLogger(name)


# -------------------- structured JSONL writers --------------------


def _append_jsonl(path: str, row: dict[str, Any]) -> None:
    # Ensure JSON-serializable & compact
    row = {
        'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'run_id': get_run_id(),
        **row,
    }
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def log_action(
    logger: logging.Logger,
    action: str,
    input_payload: dict[str, Any] | None,
    output_payload: dict[str, Any] | None,
) -> None:
    """Write a structured action record to actions.jsonl AND emit a concise debug line.

    IMPORTANT: Do NOT pass 'name' inside extra kwargs to logging methods—'name' is reserved
    in LogRecord. Use this helper instead to avoid KeyError('overwrite name in LogRecord').
    """
    # JSONL
    _append_jsonl(
        get_actions_log_path(),
        {
            'type': 'action',
            'logger': logger.name,
            'action': action,
            'input': input_payload or {},
            'output': output_payload or {},
        },
    )
    # Human-readable
    logger.debug(
        'action=%s input=%s output=%s',
        action,
        list(input_payload or {}),
        list(output_payload or {}),
    )
    # Plugin hooks (e.g. FlowCept) — errors are swallowed so they never
    # interrupt normal agent operation.
    for _hook in _ACTION_HOOKS:
        try:
            _hook(logger, action, input_payload, output_payload)
        except Exception:
            pass


def record_llm_call(payload: dict[str, Any], mirror_to: str | None = None) -> None:
    """Append a structured LLM call record to llm_calls.jsonl (or a provided path).
    Payload should already be JSON-serializable.
    """
    path = mirror_to or get_llm_audit_path()
    _append_jsonl(path, {'type': 'llm_call', **(payload or {})})
    # Plugin hooks (e.g. FlowCept) — errors are swallowed.
    for _hook in _LLM_HOOKS:
        try:
            _hook(payload or {})
        except Exception:
            pass
