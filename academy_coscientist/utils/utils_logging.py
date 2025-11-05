# academy_coscientist/utils/utils_logging.py
from __future__ import annotations

import os
import json
import time
import uuid
import logging
from typing import Any, Dict, Tuple

__all__ = [
    "init_run_context",
    "make_struct_logger",
    "log_action",
    "record_llm_call",
    "get_llm_audit_path",
    "get_actions_log_path",
    "get_run_id",
    "get_run_dir",
]

# -------------------- module-wide run context --------------------

_LOGS_ROOT_ENV = "ACADEMY_LOGS_DIR"
_DEFAULT_LOGS_ROOT = "logs"

_RUN_ID: str | None = None
_RUN_DIR: str | None = None
_LOGS_ROOT: str | None = None
_INIT_DONE: bool = False


def _timestamp() -> str:
    # UTC for reproducibility; format YYYYMMDD-HHMMSS
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def _short_uid(n: int = 8) -> str:
    return uuid.uuid4().hex[:n]


def get_run_id() -> str:
    if not _RUN_ID:
        raise RuntimeError("Run context not initialized. Call init_run_context() first.")
    return _RUN_ID


def get_run_dir() -> str:
    if not _RUN_DIR:
        raise RuntimeError("Run context not initialized. Call init_run_context() first.")
    return _RUN_DIR


def get_actions_log_path() -> str:
    return os.path.join(get_run_dir(), "actions.jsonl")


def get_llm_audit_path() -> str:
    return os.path.join(get_run_dir(), "llm_calls.jsonl")


# -------------------- initialization & logger setup --------------------

def init_run_context() -> Tuple[str, str]:
    """
    Create a unique logs directory for this run and configure root logging.

    Returns:
        (run_id, run_dir)
    """
    global _RUN_ID, _RUN_DIR, _LOGS_ROOT, _INIT_DONE

    if _INIT_DONE and _RUN_ID and _RUN_DIR:
        # idempotent; return existing
        return _RUN_ID, _RUN_DIR

    _LOGS_ROOT = os.environ.get(_LOGS_ROOT_ENV, _DEFAULT_LOGS_ROOT)
    os.makedirs(_LOGS_ROOT, exist_ok=True)

    _RUN_ID = f"{_timestamp()}-{_short_uid(8)}"
    _RUN_DIR = os.path.join(_LOGS_ROOT, _RUN_ID)
    os.makedirs(_RUN_DIR, exist_ok=True)

    # Configure root logger only once
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if someone calls init twice
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "_academy_handler", False)
               for h in root.handlers):
        # File handler captures everything (DEBUG+)
        fh = logging.FileHandler(os.path.join(_RUN_DIR, "events.log"), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh._academy_handler = True  # type: ignore[attr-defined]
        fh.setFormatter(logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root.addHandler(fh)

    if not any(isinstance(h, logging.StreamHandler) and getattr(h, "_academy_handler", False)
               for h in root.handlers):
        # Console handler is less noisy (INFO)
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh._academy_handler = True  # type: ignore[attr-defined]
        sh.setFormatter(logging.Formatter(
            fmt="%(levelname)s [%(name)s] %(message)s"
        ))
        root.addHandler(sh)

    # Touch jsonl logs so paths exist
    for p in (get_actions_log_path(), get_llm_audit_path()):
        if not os.path.exists(p):
            with open(p, "w", encoding="utf-8") as f:
                pass

    _INIT_DONE = True

    # Print a single line for discoverability (to console)
    logging.getLogger("launcher").info("Run logs at %s (run id %s)", _RUN_DIR, _RUN_ID)

    return _RUN_ID, _RUN_DIR


def make_struct_logger(name: str) -> logging.Logger:
    """
    Get a logger by name. Handlers/formatters are attached to root in init_run_context().
    """
    if not _INIT_DONE:
        # Ensure context exists even if someone forgot
        init_run_context()
    return logging.getLogger(name)


# -------------------- structured JSONL writers --------------------

def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    # Ensure JSON-serializable & compact
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": get_run_id(),
        **row,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def log_action(
    logger: logging.Logger,
    action: str,
    input_payload: Dict[str, Any] | None,
    output_payload: Dict[str, Any] | None,
) -> None:
    """
    Write a structured action record to actions.jsonl AND emit a concise debug line.

    IMPORTANT: Do NOT pass 'name' inside extra kwargs to logging methods—'name' is reserved
    in LogRecord. Use this helper instead to avoid KeyError('overwrite name in LogRecord').
    """
    # JSONL
    _append_jsonl(
        get_actions_log_path(),
        {
            "type": "action",
            "logger": logger.name,
            "action": action,
            "input": input_payload or {},
            "output": output_payload or {},
        },
    )
    # Human-readable
    logger.debug(
        "action=%s input_keys=%s output_keys=%s",
        action,
        list((input_payload or {}).keys()),
        list((output_payload or {}).keys()),
    )


def record_llm_call(payload: Dict[str, Any], mirror_to: str | None = None) -> None:
    """
    Append a structured LLM call record to llm_calls.jsonl (or a provided path).
    Payload should already be JSON-serializable.
    """
    path = mirror_to or get_llm_audit_path()
    _append_jsonl(path, {"type": "llm_call", **(payload or {})})
