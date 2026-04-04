# academy_coscientist/utils/utils_llm.py

from __future__ import annotations

import json
import os
import random
import logging
from typing import Any
import asyncio

from openai import APIStatusError, AsyncOpenAI, RateLimitError

try:
    import anthropic as _anthropic_module
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _anthropic_module = None  # type: ignore[assignment]
    _ANTHROPIC_AVAILABLE = False

from academy_coscientist.utils.config import get_config, get_model, get_temperature
from academy_coscientist.utils.utils_logging import (
    get_llm_audit_path,
    log_action,
    record_llm_call,
    make_struct_logger,
)

_logger = make_struct_logger("utils_llm")

_last_known_embed_dim: int | None = None


def _use_argo() -> bool:
    """Return True if Argo routing is active.

    Checks both the config flag (works when config is loaded in this process)
    and the ARGO_USER env var (works in subprocesses where config may not be loaded).
    """
    return bool(os.environ.get("ARGO_USER")) or get_config().get("argo_mode", False)


def _get_openai_client() -> AsyncOpenAI:
    """
    Create a fresh AsyncOpenAI client.

    We avoid binding a single client instance to a specific event loop,
    which is what caused 'Event is bound to a different event loop'
    errors when run under different asyncio loops.
    """
    return AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def _get_anthropic_client():
    """Create a fresh Anthropic async client."""
    if not _ANTHROPIC_AVAILABLE:
        raise RuntimeError(
            "anthropic package is not installed. Run: pip install anthropic"
        )
    return _anthropic_module.AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )


# ---------------------------------------------------------------------------
# Argo Gateway helpers
# ---------------------------------------------------------------------------

_ARGO_DEFAULT_URL = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"

# Maps Argo model names → direct OpenAI / Anthropic model IDs used when
# argo_mode is true but the Argo call fails and argo_fallback is enabled.
_ARGO_TO_DIRECT_MODEL: dict[str, str] = {
    "gpt4o":          "gpt-4o",
    "gpt4o-mini":     "gpt-4o-mini",
    "gpt4":           "gpt-4",
    "gpt41":          "gpt-4.1",
    "gpt41mini":      "gpt-4.1-mini",
    "gpt52":          "gpt-4o",
    "gpt54":          "gpt-4o",
    "gpt5":           "gpt-4o",
    "o4mini":         "o4-mini",
    "claudeopus46":   "claude-opus-4-6",
    "claudesonnet46": "claude-sonnet-4-6",
    "claudehaiku45":  "claude-haiku-4-5-20251001",
    "gemini25pro":    "claude-sonnet-4-6",   # no direct Gemini; fallback to Claude
    "llama3":         "gpt-4o-mini",         # no direct llama; fallback to OpenAI
}


def _get_argo_user() -> str:
    user = os.environ.get("ARGO_USER", "") or get_config().get("argo_user", "")
    if not user:
        raise RuntimeError(
            "ARGO_USER environment variable is not set and argo_user is not in config. "
            "Set ARGO_USER=<anl-username> or add argo_user: <username> to your config."
        )
    return user


def _get_argo_base_url() -> str:
    return os.environ.get("ARGO_BASE_URL", _ARGO_DEFAULT_URL)


async def call_argo_llm(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.1,
    top_p: float = 0.9,
    ctx: dict[str, Any] | None = None,
) -> str:
    """
    Call the ANL Argo Gateway and return the response text.

    Drop-in alternative to ``call_reasoning_llm`` / ``call_writing_llm`` /
    ``call_claude_llm`` — same ``(system, user, model, temperature, ctx)``
    interface, backed by the Argo REST API instead of OpenAI/Anthropic.

    The model name is passed straight to Argo (e.g. ``"gpt4o"``, ``"gpt4"``,
    ``"llama3"``).  Defaults to the ``argo`` role from config, then ``"gpt4o"``.

    Env vars
    --------
    ARGO_USER     : ANL username (required)
    ARGO_BASE_URL : override the default gateway URL (optional)
    """
    import aiohttp

    ctx = ctx or {}
    audit_path = ctx.get("audit_path", get_llm_audit_path())
    resolved_model = model or get_model("argo", default="gpt4o")
    url = _get_argo_base_url()
    argo_user = _get_argo_user()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    payload: dict[str, Any] = {
        "user": argo_user,
        "model": resolved_model,
        "messages": messages,
        "stop": [],
        "top_p": top_p,
    }
    # Claude models on Argo do not accept temperature or top_p
    _is_claude = resolved_model.lower().startswith("claude")
    if temperature is not None and not _is_claude:
        payload["temperature"] = temperature
    if _is_claude:
        payload.pop("top_p", None)

    llm_record: dict[str, Any] = {
        "type": "chat_completion",
        "model": resolved_model,
        "temperature": temperature,
        "top_p": top_p,
        "messages": messages,
        "context": ctx,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                status = resp.status
                if status != 200:
                    # Read the body for a useful error message, then raise a
                    # plain RuntimeError so the exception is always picklable
                    # (aiohttp's ClientResponseError carries CIMultiDictProxy
                    # headers which cannot be pickled by the Redis exchange).
                    try:
                        body = await resp.text()
                    except Exception:
                        body = "<unreadable>"
                    raise RuntimeError(
                        f"Argo API returned HTTP {status} for model={resolved_model!r}: {body[:300]}"
                    )
                data = await resp.json()
        content = data.get("response", "")
        # Argo sometimes prepends an auth-warning block to the actual response.
        # The block ends with "** END NOTICE FROM ARGO **"; strip it when present.
        _ARGO_NOTICE_END = "** END NOTICE FROM ARGO **"
        if _ARGO_NOTICE_END in content:
            content = content.split(_ARGO_NOTICE_END, 1)[1].strip()
        llm_record["text"] = content
        llm_record["finish_reason"] = "stop"
        record_llm_call(llm_record, mirror_to=audit_path)
        return content
    except RuntimeError:
        raise  # already clean and picklable
    except Exception as e:
        # Wrap any other aiohttp / network exception in a plain RuntimeError
        # so it can be serialised by the Redis exchange (pickle-safe).
        msg = f"Argo API call failed (model={resolved_model!r}): {e}"
        llm_record["error"] = msg
        record_llm_call(llm_record, mirror_to=audit_path)
        _logger.error("Argo API call failed", extra={"error": repr(e), "model": resolved_model})
        raise RuntimeError(msg) from None


async def call_claude_llm(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 1.0,
    max_tokens: int = 8192,
    ctx: dict[str, Any] | None = None,
) -> str:
    """
    Call the Anthropic Claude API and return the response text.

    The model defaults to the `poc` role from config, then `claude-sonnet-4-6`.
    Logs the call via record_llm_call for FlowCept provenance.
    """
    ctx = ctx or {}
    audit_path = ctx.get("audit_path", get_llm_audit_path())
    resolved_model = model or get_model("poc", default="claude-sonnet-4-6")

    if _use_argo():
        return await call_argo_llm(
            system=system,
            user=user,
            model=resolved_model,
            temperature=temperature,
            ctx=ctx,
        )

    messages = [{"role": "user", "content": user}]
    llm_record: dict[str, Any] = {
        "type": "chat_completion",
        "model": resolved_model,
        "system_instructions": system,
        "user_prompt": user,
        "messages": messages,
        "temperature_requested": temperature,
        "context": ctx,
    }

    try:
        client = _get_anthropic_client()
        resp = await client.messages.create(
            model=resolved_model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            temperature=temperature,
        )
        content = resp.content[0].text if resp.content else ""
        llm_record["text"] = content
        llm_record["model_used"] = resp.model
        llm_record["finish_reason"] = resp.stop_reason
        llm_record["temperature_sent"] = temperature
        if resp.usage:
            llm_record["usage"] = {
                "prompt_tokens": resp.usage.input_tokens,
                "completion_tokens": resp.usage.output_tokens,
                "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
            }
        record_llm_call(llm_record, mirror_to=audit_path)
        return content
    except Exception as e:
        llm_record["error"] = repr(e)
        record_llm_call(llm_record, mirror_to=audit_path)
        _logger.error("Claude API call failed", extra={"error": repr(e), "model": resolved_model})
        raise

def _is_local_embedding_model(name: str) -> bool:
    return str(name).strip().lower().startswith("local-")


def _local_model_name_from_config(name: str) -> str:
    return name.split("local-", 1)[1] if "local-" in name else name


# ---------------------------------------------------------------------------
# Optional local embedding backend (SentenceTransformer)
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]

_LOCAL_EMBEDDER_CACHE: dict[str, Any] = {}


def _get_local_embedder(model_name: str):
    global _LOCAL_EMBEDDER_CACHE, SentenceTransformer

    if model_name in _LOCAL_EMBEDDER_CACHE:
        return _LOCAL_EMBEDDER_CACHE[model_name]

    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers not installed, but a local embedding "
            "model was requested."
        )

    model = SentenceTransformer(model_name)
    _LOCAL_EMBEDDER_CACHE[model_name] = model
    return model


# ---------------------------------------------------------------------------
# Safe JSON serializers (for logging SDK objects)
# ---------------------------------------------------------------------------


def _to_safe_json(x: Any, depth: int = 0, max_depth: int = 4) -> Any:
    if depth > max_depth:
        return str(x)

    if isinstance(x, dict):
        return {k: _to_safe_json(v, depth + 1, max_depth) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_safe_json(v, depth + 1, max_depth) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x

    return repr(x)


# ---------------------------------------------------------------------------
# Chat / reasoning helpers
# ---------------------------------------------------------------------------


async def _chat_completion(
    model: str,
    messages: list[dict[str, str]],
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    ctx: dict[str, Any] | None = None,
) -> str:
    ctx = ctx or {}
    audit_path = ctx.get("audit_path", get_llm_audit_path())

    _NO_TEMPERATURE_MODELS = {"o1", "o1-mini", "o3", "o3-mini", "o4-mini", "gpt-5", "gpt-5-mini"}
    allow_temperature = model not in _NO_TEMPERATURE_MODELS

    temperature_value = temperature if allow_temperature else None

    llm_record: dict[str, Any] = {
        "type": "chat_completion",
        "model": model,
        "temperature": temperature_value,
        "messages": messages,
        "context": ctx,
    }

    try:
        client = _get_openai_client()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens
        if temperature_value is not None:
            kwargs["temperature"] = temperature_value

        resp = await client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""
        llm_record["text"] = content          # actual response text
        llm_record["response"] = _to_safe_json(resp)
        llm_record["model_used"] = getattr(resp, "model", model)
        llm_record["temperature_requested"] = temperature
        llm_record["temperature_sent"] = temperature_value
        llm_record["temperature_suppressed"] = not allow_temperature
        llm_record["finish_reason"] = (
            resp.choices[0].finish_reason if resp.choices else None
        )
        if resp.usage:
            llm_record["usage"] = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        record_llm_call(llm_record, mirror_to=audit_path)
        return content
    except (APIStatusError, RateLimitError) as e:
        llm_record["error"] = repr(e)
        record_llm_call(llm_record, mirror_to=audit_path)
        _logger.error("Chat completion failed", extra={"error": repr(e), "model": model})
        raise RuntimeError(f"OpenAI chat completion error (model={model!r}): {e}") from None


def _is_anthropic_model(name: str) -> bool:
    return str(name).strip().lower().startswith("claude-")


def _resolve_fallback_model(argo_model: str) -> str:
    """
    Resolve a direct OpenAI/Anthropic model name to use when Argo is unavailable.

    Priority:
    1. ``fallback_models.<argo_model>`` in config  (explicit per-model override)
    2. ``_ARGO_TO_DIRECT_MODEL`` table             (built-in mapping)
    3. ``fallback_models.default`` in config       (catch-all override)
    4. ``"gpt-4o"``                                (hard-coded last resort)
    """
    cfg_fallbacks: dict = get_config().get("fallback_models", {}) or {}
    if argo_model in cfg_fallbacks:
        return str(cfg_fallbacks[argo_model])
    if argo_model in _ARGO_TO_DIRECT_MODEL:
        return _ARGO_TO_DIRECT_MODEL[argo_model]
    if "default" in cfg_fallbacks:
        return str(cfg_fallbacks["default"])
    return "gpt-4o"


async def _dispatch_chat(
    system: str,
    user: str,
    model: str,
    temperature: float | None,
    ctx: dict[str, Any] | None,
    max_tokens: int | None = None,
) -> str:
    """Route a chat request to Argo, Anthropic, or OpenAI based on active config and model name."""
    if _use_argo():
        try:
            return await call_argo_llm(
                system=system,
                user=user,
                model=model,
                temperature=temperature if temperature is not None else 0.1,
                ctx=ctx,
            )
        except Exception as argo_err:
            if not get_config().get("argo_fallback", False):
                raise
            fallback_model = _resolve_fallback_model(model)
            _logger.warning(
                "Argo call failed (model=%r), falling back to direct API (fallback=%r): %s",
                model, fallback_model, argo_err,
            )
            model = fallback_model  # fall through to direct-API dispatch below

    if _is_anthropic_model(model):
        return await call_claude_llm(
            system=system,
            user=user,
            model=model,
            temperature=temperature if temperature is not None else 1.0,
            max_tokens=max_tokens or 8192,
            ctx=ctx,
        )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return await _chat_completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        ctx=ctx,
    )


async def call_reasoning_llm(
    system: str,
    user: str,
    ctx: dict[str, Any] | None = None,
    max_completion_tokens: int | None = None,
    temperature: float | None = None,
    model: str | None = None,
) -> str:
    _model = model or get_model("reasoning")
    return await _dispatch_chat(
        system=system,
        user=user,
        model=_model,
        temperature=temperature,
        ctx=ctx,
        max_tokens=max_completion_tokens,
    )


async def call_writing_llm(
    system: str,
    user: str,
    temperature: float | None = None,
    ctx: dict[str, Any] | None = None,
    max_completion_tokens: int | None = None,
    model: str | None = None,
) -> str:
    if temperature is None:
        temperature = get_temperature("writing")
    _model = model or get_model("writing")
    return await _dispatch_chat(
        system=system,
        user=user,
        model=_model,
        temperature=temperature,
        ctx=ctx,
        max_tokens=max_completion_tokens,
    )


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


async def _embed_with_local_model(
    model_alias: str,
    texts: list[str],
    ctx: dict[str, Any],
) -> list[list[float]]:
    global _last_known_embed_dim

    audit_path = ctx.get("audit_path", get_llm_audit_path())
    raw_name = _local_model_name_from_config(model_alias)

    record_llm_call(
        {
            "type": "embed_call_local",
            "model": raw_name,
            "alias": model_alias,
            "texts_sample": texts[:3],
            "num_texts": len(texts),
            "context": ctx,
        },
        mirror_to=audit_path,
    )

    try:
        embedder = _get_local_embedder(raw_name)

        loop = asyncio.get_running_loop()

        def _do_encode() -> list[list[float]]:
            arr = embedder.encode(texts, convert_to_numpy=True)
            return arr.tolist()

        vectors: list[list[float]] = await loop.run_in_executor(None, _do_encode)

        if vectors and isinstance(vectors[0], list):
            _last_known_embed_dim = len(vectors[0])

        record_llm_call(
            {
                "type": "embed_result_local",
                "model": raw_name,
                "alias": model_alias,
                "num_vectors": len(vectors),
                "dim": _last_known_embed_dim,
                "context": ctx,
            },
            mirror_to=audit_path,
        )
        return vectors
    except Exception as e:  # pragma: no cover - defensive
        record_llm_call(
            {
                "type": "embed_error_local",
                "model": raw_name,
                "alias": model_alias,
                "error": repr(e),
                "context": ctx,
            },
            mirror_to=audit_path,
        )
        raise


async def _embed_with_openai_model(
    model_name: str,
    texts: list[str],
    ctx: dict[str, Any],
) -> list[list[float]]:
    global _last_known_embed_dim

    audit_path = ctx.get("audit_path", get_llm_audit_path())
    record_llm_call(
        {
            "type": "embed_call_openai",
            "model": model_name,
            "texts_sample": texts[:3],
            "num_texts": len(texts),
            "context": ctx,
        },
        mirror_to=audit_path,
    )

    try:
        client = _get_openai_client()
        resp = await client.embeddings.create(
            model=model_name,
            input=texts,
        )
    except (APIStatusError, RateLimitError) as e:
        record_llm_call(
            {
                "type": "embed_error_openai",
                "model": model_name,
                "error": repr(e),
                "context": ctx,
            },
            mirror_to=audit_path,
        )
        # APIStatusError carries an httpx Response object which is not picklable;
        # re-raise as a plain RuntimeError so the Redis exchange can serialize it.
        raise RuntimeError(f"OpenAI embedding error (model={model_name!r}): {e}") from None

    vectors: list[list[float]] = [d.embedding for d in resp.data]  # type: ignore[assignment]

    if vectors and isinstance(vectors[0], list):
        _last_known_embed_dim = len(vectors[0])

    record_llm_call(
        {
            "type": "embed_result_openai",
            "model": model_name,
            "num_vectors": len(vectors),
            "dim": _last_known_embed_dim,
            "context": ctx,
        },
        mirror_to=audit_path,
    )
    return vectors


async def embed_texts(
    texts: list[str],
    context: dict[str, Any] | None = None,
) -> list[list[float]]:
    """
    Front-door for embedding calls.

    Uses OpenAI models by default; if the config model name starts with 'local-',
    a local sentence-transformer is used instead.
    """
    ctx = context or {}
    embed_model = get_model("embedding")

    if _is_local_embedding_model(embed_model):
        return await _embed_with_local_model(embed_model, texts, ctx)
    else:
        return await _embed_with_openai_model(embed_model, texts, ctx)


# ---------------------------------------------------------------------------
# Text LLM helpers (JSON)
# ---------------------------------------------------------------------------


def _extract_text_for_commentary(d: Any) -> str:
    texts: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            t = node.get("type")
            if t == "reasoning":
                txt = node.get("text") or node.get("content") or node.get("value")
                if isinstance(txt, str):
                    texts.append(txt)
                elif isinstance(txt, list):
                    segs = [s for s in txt if isinstance(s, str)]
                    if segs:
                        texts.append("\n".join(segs))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(d)
    out = "\n".join(texts).strip()
    return out[:4000] + ("..." if len(out) > 4000 else "")


# ---------------------------------------------------------------------------
# JSON-producing helper used by ReportAgent and others
# ---------------------------------------------------------------------------


async def _call_llm_json(
    system_instructions: str,
    user_prompt: str,
    schema_hint: str,
    context: dict[str, Any] | None = None,
    temperature: float | None = None,
    model_role: str = "reasoning",
) -> Any:
    """
    Ask the reasoning model for STRICT JSON following schema_hint.
    Parses & logs the JSON.

    On malformed or empty JSON, we return `{}` instead of fabricating content.
    """
    ctx = context or {}
    audit_path = ctx.get("audit_path", get_llm_audit_path())
    reasoning_model = get_model(model_role, default=get_model("reasoning"))
    _temperature = temperature if temperature is not None else 0.1

    messages = [
        {
            "role": "system",
            "content": (
                system_instructions.strip()
                + "\n\nYou MUST reply with STRICT JSON only. "
                "Do not include markdown fences or explanations.\n"
                "Intended JSON shape (informal description):\n"
                f"{schema_hint.strip()}\n"
            ),
        },
        {"role": "user", "content": user_prompt.strip()},
    ]

    llm_record: dict[str, Any] = {
        "type": "json_call",
        "model": reasoning_model,
        "messages": messages,
        "schema_hint": schema_hint,
        "context": ctx,
    }

    try:
        llm_text = await _dispatch_chat(
            system=messages[0]["content"],
            user=messages[1]["content"],
            model=reasoning_model,
            temperature=_temperature,
            ctx=ctx,
        )
        llm_record["model_used"] = reasoning_model
        llm_record["finish_reason"] = "stop"
    except Exception as e:
        llm_record["error"] = repr(e)
        record_llm_call(llm_record, mirror_to=audit_path)
        _logger.error(
            "JSON LLM call failed",
            extra={"error": repr(e), "model": reasoning_model},
        )
        raise

    parsed: Any = {}
    fallback_used = False

    s = llm_text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
            s = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(s)
    except Exception:
        start_obj = llm_text.find("{")
        start_arr = llm_text.find("[")
        if start_obj == -1 and start_arr == -1:
            parsed = {}
            fallback_used = True
        else:
            if start_obj == -1:
                start_idx = start_arr
            elif start_arr == -1:
                start_idx = start_obj
            else:
                start_idx = min(start_obj, start_arr)

            end_curly = llm_text.rfind("}")
            end_brack = llm_text.rfind("]")
            end_idx = max(end_curly, end_brack)
            candidate = llm_text[start_idx : end_idx + 1]
            try:
                parsed = json.loads(candidate)
            except Exception:
                parsed = {}
                fallback_used = True

    # Surface reasoning as a top-level key so provenance tools can find it
    # without traversing parsed_response.
    _reasoning = parsed.get("reasoning", "") if isinstance(parsed, dict) else ""
    record_llm_call(
        {
            "type": "parsed_json_result",
            "model": reasoning_model,
            "model_used": llm_record.get("model_used", reasoning_model),
            "temperature": None,          # reasoning model — temperature not sent
            "temperature_suppressed": True,
            "system_instructions": system_instructions,
            "user_prompt": user_prompt,
            "schema_hint": schema_hint,
            "raw_response_text": llm_text,
            "parsed_response": parsed,
            "reasoning": _reasoning,
            "fallback_used": fallback_used,
            "finish_reason": llm_record.get("finish_reason"),
            "usage": llm_record.get("usage"),
            "context": ctx,
        },
        mirror_to=audit_path,
    )

    return parsed


# ---------------------------------------------------------------------------
# Convenience wrapper to maintain old call style
# ---------------------------------------------------------------------------


async def call_llm_json(*args: Any, **kwargs: Any) -> Any:
    """
    Backwards-compatible wrapper that forwards to _call_llm_json.

    It supports two styles:

      - Positional:
          await call_llm_json(system_instructions, user_prompt, schema_hint)

      - Keyword:
          await call_llm_json(
              system_msg="...",
              user_msg="...",
              schema_hint="...",
          )
    """
    system_msg: str | None = None
    user_msg: str | None = None
    schema_hint: str | None = None
    context: dict[str, Any] | None = kwargs.get("context")

    if len(args) == 3:
        system_msg = args[0]
        user_msg = args[1]
        schema_hint = args[2]
    elif len(args) > 0:
        system_msg = args[0]
        if len(args) >= 2:
            user_msg = args[1]
        if len(args) >= 3:
            schema_hint = args[2]

    system_msg = kwargs.get("system_msg", system_msg)
    user_msg = kwargs.get("user_msg", user_msg)
    schema_hint = kwargs.get("schema_hint", schema_hint)

    if user_msg is None:
        user_msg = "No user prompt provided."
    if system_msg is None:
        system_msg = "You are a helpful scientific writing assistant."

    return await _call_llm_json(
        system_instructions=system_msg,
        user_prompt=user_msg,
        schema_hint=schema_hint or "",
        context=context,
    )


# ---------------------------------------------------------------------------
# Higher-level helpers used by agents
# ---------------------------------------------------------------------------


async def extract_search_keywords(topic: str, max_keywords: int = 5) -> str:
    """
    Distil a research topic (which may be a full PDF extract or a verbose
    description) into a short, focused keyword query suitable for paper
    database APIs (Semantic Scholar, OpenAlex, arXiv).

    Uses a fast, deterministic LLM call so the harvester always has a clean,
    concise query regardless of how verbose the original topic is.

    Parameters
    ----------
    topic:
        Any text — a short topic string, a full document extract, etc.
    max_keywords:
        Maximum number of keywords / key phrases to return.

    Returns
    -------
    A comma-separated keyword string, e.g.
    ``"vector database, knowledge graph, HPC memory, embedding retrieval"``
    """
    # Truncate very long inputs (PDF text) to keep the prompt cheap
    topic_snippet = topic[:3000] if len(topic) > 3000 else topic

    system = (
        "You are a scientific literature search assistant. "
        "Your sole task is to extract the most specific and informative technical "
        f"keywords from the given research topic for querying academic paper databases. "
        f"Return exactly {max_keywords} keywords or short key-phrases, "
        "comma-separated on a single line. No explanations, no numbering, no extra text."
    )
    user = f"Research topic:\n\n{topic_snippet}"

    def _topic_fallback(t: str) -> str:
        """Return the first sentence (or first 200 chars) as a plain keyword query."""
        first = t.split(".")[0].strip()
        return first[:200] if first else t[:200]

    try:
        raw = await call_writing_llm(
            system=system,
            user=user,
            temperature=0.1,
            model=get_model("commentary"),  # lightweight model
            ctx={"task": "keyword_extraction"},
        )
        # Normalise: strip markdown fences, take only the first line
        cleaned = raw.strip().strip("`").split("\n")[0].strip()
        # Guard: if the result looks like an error/notice rather than keywords,
        # fall back (catches cases where Argo warning stripping was incomplete).
        _BAD_MARKERS = ("NOTICE", "WARNING", "AUTHENTICATION", "ERROR", "<!DOCTYPE", "⚠")
        if any(m in cleaned for m in _BAD_MARKERS) or len(cleaned) > 300:
            return _topic_fallback(topic)
        return cleaned or _topic_fallback(topic)
    except Exception:
        return _topic_fallback(topic)


async def review_hypothesis(
    hypothesis: dict[str, Any],
    retrieved: list[dict[str, Any]],
    topic: str = "",
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Use the reasoning LLM to score a hypothesis on five dimensions and return a
    structured critique with a final composite score in [0, 1].

    Dimensions scored 1-10 each:
      - novelty        : how original / non-obvious the idea is
      - rigor          : how precise, falsifiable, and well-defined it is
      - feasibility    : how testable / implementable it is with current methods
      - impact         : potential scientific or practical significance
      - clarity        : how clearly the hypothesis is stated

    The composite score = weighted mean of the five dimensions, normalized to [0, 1].
    Weights: novelty=0.25, rigor=0.20, feasibility=0.20, impact=0.25, clarity=0.10
    """
    ctx = context or {}
    hyp_id    = hypothesis.get('id', 'unknown')
    title     = hypothesis.get('title', 'Untitled')
    desc      = hypothesis.get('description', '')

    # Summarise retrieved context for the prompt (keep it short)
    context_snippets: list[str] = []
    for doc in (retrieved or [])[:5]:
        snippet = (
            doc.get('abstract') or doc.get('text') or doc.get('title') or ''
        )
        if snippet:
            context_snippets.append(snippet[:300])
    context_block = '\n\n'.join(context_snippets) if context_snippets else '(none)'

    topic_line = f' within the research domain: "{topic}"' if topic else ""
    system = (
        f"You are a rigorous scientific peer reviewer evaluating hypotheses"
        f"{topic_line}.\n\n"
        "Your job is to evaluate a research hypothesis critically and fairly. "
        "When scoring NOVELTY, assess originality relative to the specific research domain "
        "stated above — not in the abstract. A hypothesis is novel if it introduces a new "
        "mechanistic angle, experimental approach, or theoretical framing that has not been "
        "well-explored within this domain. Do NOT penalise a hypothesis for being 'off-topic' "
        "or suggest the authors pivot to a different domain."
    )

    schema = """
    {
      "reasoning":     <string — step-by-step chain of thought before scoring>,
      "novelty":       <int 1-10>,
      "rigor":         <int 1-10>,
      "feasibility":   <int 1-10>,
      "impact":        <int 1-10>,
      "clarity":       <int 1-10>,
      "confidence":    <float 0-1, how confident you are in these scores>,
      "strengths":     [<string>, ...],
      "weaknesses":    [<string>, ...],
      "risks":         [<string>, ...],
      "recommendation": "accept" | "revise" | "reject",
      "summary":       <string, 1-2 sentences>
    }
    """

    topic_context = f"Research topic: {topic}\n\n" if topic else ""
    user = (
        f"{topic_context}"
        f"Hypothesis title: {title}\n\n"
        f"Description:\n{desc}\n\n"
        f"Relevant literature context:\n{context_block}\n\n"
        "First write your step-by-step reasoning in the \"reasoning\" field, then score the "
        "hypothesis on the following dimensions (1 = very poor, 10 = excellent):\n"
        "  novelty      – how original/non-obvious is the idea within this research domain?\n"
        "  rigor        – how precise, falsifiable, and well-defined is it?\n"
        "  feasibility  – how testable/implementable with current methods?\n"
        "  impact       – potential scientific or practical significance?\n"
        "  clarity      – how clearly is the hypothesis stated?\n"
        "  confidence   – your confidence in these scores (0.0 = very uncertain, 1.0 = very certain)\n\n"
        "Also list 1-3 strengths, 1-3 weaknesses, major risks, a recommendation "
        "(accept / revise / reject), and a 1-2-sentence summary.\n"
        "Return STRICT JSON only (no markdown, no prose outside JSON)."
    )

    raw = await call_reasoning_llm(
        system=system,
        user=user,
        model=get_model("review"),
        temperature=get_temperature("review", 0.1) or 0.1,
        ctx={**ctx, 'call_type': 'review_hypothesis', 'hyp_id': hyp_id},
        max_completion_tokens=4000,
    )

    # Parse JSON — tolerate markdown fences
    parsed: dict[str, Any] = {}
    s = raw.strip()
    if s.startswith('```'):
        lines = s.splitlines()
        s = '\n'.join(lines[1:-1]).strip()
    try:
        parsed = json.loads(s)
    except Exception:
        start = min(
            (raw.find(c) for c in ('{', '[') if raw.find(c) != -1),
            default=-1,
        )
        if start != -1:
            end = max(raw.rfind('}'), raw.rfind(']'))
            try:
                parsed = json.loads(raw[start: end + 1])
            except Exception:
                parsed = {}

    def _clamp(v: Any, lo: float = 1.0, hi: float = 10.0) -> float:
        try:
            return max(lo, min(hi, float(v)))
        except (TypeError, ValueError):
            return (lo + hi) / 2.0

    # Some models nest scores under a "scores" sub-object; flatten it.
    scores_src = parsed.get('scores') if isinstance(parsed.get('scores'), dict) else {}
    def _get_dim(key: str, default: Any = 5) -> Any:
        return parsed.get(key, scores_src.get(key, default))

    novelty     = _clamp(_get_dim('novelty'))
    rigor       = _clamp(_get_dim('rigor'))
    feasibility = _clamp(_get_dim('feasibility'))
    impact      = _clamp(_get_dim('impact'))
    clarity     = _clamp(_get_dim('clarity'))

    # Weighted composite, normalised to [0, 1]
    composite = (
        0.25 * novelty +
        0.20 * rigor +
        0.20 * feasibility +
        0.25 * impact +
        0.10 * clarity
    ) / 10.0

    # Confidence: prefer model-supplied value; fall back to 1 - normalised std-dev
    import statistics
    dims = [novelty, rigor, feasibility, impact, clarity]
    try:
        std = statistics.stdev(dims)
        consistency_confidence = max(0.0, 1.0 - std / 9.0)
    except Exception:
        consistency_confidence = 0.5

    raw_confidence = parsed.get('confidence', scores_src.get('confidence'))
    if raw_confidence is not None:
        try:
            confidence = max(0.0, min(1.0, float(raw_confidence)))
        except (TypeError, ValueError):
            confidence = consistency_confidence
    else:
        confidence = consistency_confidence

    critique = {
        'novelty':        novelty,
        'rigor':          rigor,
        'feasibility':    feasibility,
        'impact':         impact,
        'clarity':        clarity,
        'score':          round(composite, 4),
        'confidence':     round(confidence, 4),
        'reasoning':      parsed.get('reasoning', ''),
        'strengths':      parsed.get('strengths',  []),
        'weaknesses':     parsed.get('weaknesses', []),
        'risks':          parsed.get('risks',      []),
        'recommendation': parsed.get('recommendation', 'revise'),
        'notes':          parsed.get('summary', ''),
    }

    _logger.info(
        'review_hypothesis_done',
        extra={
            'hyp_id':     hyp_id,
            'score':      critique['score'],
            'confidence': critique['confidence'],
        },
    )
    return critique


def _normalize_title(title: str) -> str:
    """Lowercase + strip punctuation/whitespace for fuzzy title comparison."""
    import re
    return re.sub(r'[^a-z0-9 ]', '', title.lower()).strip()


def _parse_hypotheses_raw(parsed: Any) -> list[dict[str, Any]]:
    """Extract a list of hypothesis dicts from a raw LLM-parsed response."""
    raw: list | None = None
    if isinstance(parsed, dict):
        raw = parsed.get("hypotheses")
    elif isinstance(parsed, list):
        raw = parsed
    if not isinstance(raw, list):
        return []
    out = []
    for i, item in enumerate(raw):
        # Always assign our own deterministic id — never trust the LLM to generate it,
        # as it produces inconsistent formats and sometimes omits or blanks the field.
        our_id = f"h_{i+1}"
        if isinstance(item, dict):
            out.append({
                "id":          our_id,
                "title":       item.get("title", f"Hypothesis {i+1}"),
                "description": item.get("description", str(item)),
                "rationale":   item.get("rationale", ""),
            })
        else:
            out.append({
                "id":          our_id,
                "title":       f"Hypothesis {i+1}",
                "description": str(item),
                "rationale":   "",
            })
    return out


async def brainstorm_hypotheses(
    topic: str,
    n: int,
    context: dict[str, Any] | None = None,
    exclude_titles: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Generate `n` unique hypotheses for `topic`.

    Titles in `exclude_titles` are forbidden.  If the LLM returns duplicates
    despite the instruction, filtered results are collected and the LLM is
    re-called for the missing count — up to 2 retry rounds.
    """
    ctx = context or {}
    system = (
        "You are a careful, critical scientist. Generate diverse, "
        "non-redundant hypotheses for the given research topic."
    )
    schema_tpl = """
{{
  "reasoning": <string — step-by-step chain of thought before proposing the hypotheses>,
  "hypotheses": [
    {{
      "title":       <string, concise hypothesis title>,
      "description": <string, 2-4 sentences describing the hypothesis>,
      "rationale":   <string, why this hypothesis is worth investigating>
    }},
    ... (exactly {n} items)
  ]
}}
"""

    # Normalised set of already-known titles (exclude_titles + collected so far)
    excluded_norm: set[str] = {_normalize_title(t) for t in (exclude_titles or []) if t}
    collected: list[dict[str, Any]] = []
    remaining = n
    MAX_RETRIES = 2

    for attempt in range(MAX_RETRIES + 1):
        all_excluded = list(exclude_titles or []) + [h["title"] for h in collected]
        instructions = (
            f"Propose exactly {remaining} distinct, falsifiable hypotheses. "
            "Think step by step in the 'reasoning' field before listing hypotheses. "
            "Each hypothesis must be substantially different from the others."
        )
        if all_excluded:
            instructions += (
                " The following hypothesis titles have ALREADY been generated — "
                "do NOT produce any hypothesis that is similar or overlapping with them: "
                + "; ".join(f'"{t}"' for t in all_excluded)
                + "."
            )
        user = json.dumps(
            {"topic": topic, "n": remaining, "instructions": instructions},
            indent=2,
        )
        parsed = await _call_llm_json(
            system_instructions=system,
            user_prompt=user,
            schema_hint=schema_tpl.format(n=remaining),
            context={**ctx, "call_type": "brainstorm_hypotheses", "attempt": attempt},
            temperature=get_temperature("generation", 0.7) or 0.7,
            model_role="generation",
        )
        # Log brainstorm reasoning to actions.jsonl so it appears in provenance.
        _brainstorm_reasoning = parsed.get("reasoning", "") if isinstance(parsed, dict) else ""
        if _brainstorm_reasoning:
            log_action(
                _logger,
                "brainstorm_reasoning",
                {"topic": topic, "attempt": attempt, "n_requested": remaining},
                {"reasoning": _brainstorm_reasoning},
            )
        batch = _parse_hypotheses_raw(parsed)

        # Filter duplicates from this batch
        for hyp in batch:
            norm = _normalize_title(hyp["title"])
            if norm and norm not in excluded_norm:
                excluded_norm.add(norm)
                collected.append(hyp)
                if len(collected) >= n:
                    break

        remaining = n - len(collected)
        if remaining <= 0:
            break

    return collected[:n]


async def agent_self_commentary(
    agent_name: str,
    payload: Any,
    context: dict[str, Any] | None = None,
) -> str:
    ctx = context or {}
    system = (
        "You are a meta-cognitive assistant describing the reasoning process of "
        "another agent for debugging and transparency."
    )
    user = json.dumps(
        {
            "agent": agent_name,
            "payload": payload,
        },
        indent=2,
    )
    out = await call_writing_llm(
        system=system,
        user=user,
        model=get_model("commentary"),
        temperature=get_temperature("commentary"),
        ctx={**ctx, "call_type": "agent_self_commentary"},
        max_completion_tokens=800,
    )
    return out


def _build_history_block(history: list[dict[str, Any]]) -> str:
    """
    Format review history into a compact prompt block.

    Each entry in *history* is expected to have:
      round, avg_score, merged (dict with weaknesses/risks/strengths/reasoning).

    Also surfaces recurring weaknesses — issues that appeared in 2+ rounds —
    so the LLM knows which problems have NOT been fixed yet.
    """
    if not history:
        return ""

    lines: list[str] = ["## Review history (all previous rounds)\n"]

    # Collect weakness frequency across rounds to highlight recurring ones
    weakness_count: dict[str, int] = {}
    for entry in history:
        for w in (entry.get("merged") or {}).get("weaknesses") or []:
            weakness_count[w] = weakness_count.get(w, 0) + 1

    recurring = [w for w, cnt in weakness_count.items() if cnt >= 2]

    for entry in history:
        rnd   = entry.get("round", "?")
        score = entry.get("avg_score", "?")
        m     = entry.get("merged") or {}
        wk    = "; ".join(m.get("weaknesses") or []) or "none"
        rk    = "; ".join(m.get("risks") or []) or "none"
        st    = "; ".join(m.get("strengths") or []) or "none"
        rec   = m.get("recommendation", "revise")
        rsn   = (m.get("reasoning") or "")[:300]

        lines.append(
            f"Round {rnd} | avg_score={score} | recommendation={rec}\n"
            f"  Strengths : {st}\n"
            f"  Weaknesses: {wk}\n"
            f"  Risks     : {rk}\n"
            f"  Reasoning : {rsn}"
        )

    if recurring:
        lines.append(
            "\n⚠️  RECURRING ISSUES (appeared in 2+ rounds — MUST be fixed):\n"
            + "\n".join(f"  • {w}" for w in recurring)
        )

    return "\n\n".join(lines)


async def refine_hypothesis(
    hypothesis: dict[str, Any],
    critique: dict[str, Any],
    topic: str,
    history: list[dict[str, Any]] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Use the reasoning LLM to improve a hypothesis given reviewer critique
    and the full history of all previous review rounds.

    Parameters
    ----------
    hypothesis:
        Current hypothesis dict (id, title, description, rationale, …).
    critique:
        Merged critique from the *current* review round (score, weaknesses,
        risks, strengths, reasoning, recommendation).
    topic:
        Research topic string for context.
    history:
        All previous review rounds for this hypothesis.  Each entry:
        ``{"round": int, "avg_score": float, "merged": dict, "critiques": list}``.
        When provided, the prompt explicitly flags recurring weaknesses —
        issues that were not fixed in prior refinements — and shows the
        full score trajectory so the LLM understands the improvement arc.
    context:
        Optional logging/provenance context dict.

    Returns
    -------
    dict
        Improved hypothesis with the same id, updated fields, plus
        ``revision_notes`` and ``refinement_round``.
    """
    ctx    = context or {}
    hyp_id = hypothesis.get("id", "unknown")
    title  = hypothesis.get("title", "Untitled")
    desc   = hypothesis.get("description", "")
    rationale = hypothesis.get("rationale", "")
    current_round = hypothesis.get("refinement_round", 0)

    score          = critique.get("score", 0.0)
    recommendation = critique.get("recommendation", "revise")
    reasoning      = critique.get("reasoning", "")
    weaknesses     = critique.get("weaknesses") or []
    risks          = critique.get("risks") or []
    strengths      = critique.get("strengths") or []

    # Individual dimension scores (1-10) — extracted so the prompt can
    # highlight which specific dimensions need the most improvement.
    novelty     = critique.get("novelty")
    rigor       = critique.get("rigor")
    feasibility = critique.get("feasibility")
    impact      = critique.get("impact")
    clarity     = critique.get("clarity")

    # Build history block (empty string when no history yet)
    history_block = _build_history_block(history or [])

    # Score trajectory for the prompt header
    if history:
        scores = [e.get("avg_score", 0) for e in history]
        trajectory = " → ".join(f"{s:.3f}" for s in scores)
        trajectory_line = f"Score trajectory: {trajectory}"
    else:
        trajectory_line = f"Score: {score:.3f}/1.0 (first review)"

    # Build dimension score line (only include dimensions the reviewer supplied)
    _dim_parts = []
    for _name, _val in [
        ("novelty", novelty), ("rigor", rigor), ("feasibility", feasibility),
        ("impact", impact), ("clarity", clarity),
    ]:
        if _val is not None:
            _dim_parts.append(f"{_name}={_val}/10")
    _dim_line = "  " + "  ".join(_dim_parts) if _dim_parts else ""

    # Flag dimensions that are clearly below par (< 6/10) so the refiner
    # knows exactly where to focus.
    _low_dims = [
        n for n, v in [
            ("novelty", novelty), ("rigor", rigor), ("feasibility", feasibility),
            ("impact", impact), ("clarity", clarity),
        ]
        if v is not None and float(v) < 6.0
    ]
    _low_line = (
        "  ⚠️  Dimensions scoring below 6/10 (priority targets): "
        + ", ".join(_low_dims)
        if _low_dims else ""
    )

    current_feedback = (
        f"## Current round feedback (round {current_round + 1})\n"
        f"Composite score: {score:.3f}/1.0\n"
        + (f"Dimension scores:\n{_dim_line}\n" if _dim_line else "")
        + (_low_line + "\n" if _low_line else "")
        + f"Recommendation: {recommendation}\n"
        f"Reviewer reasoning: {reasoning}\n"
        f"Strengths : {'; '.join(strengths) if strengths else 'none'}\n"
        f"Weaknesses: {'; '.join(weaknesses) if weaknesses else 'none'}\n"
        f"Risks     : {'; '.join(risks) if risks else 'none'}"
    )

    schema = """
{
  "reasoning":      <string — step-by-step analysis of which dimensions are weakest, what needs to change and why>,
  "title":          <string — improved, concise title>,
  "description":    <string — improved description, more precise and falsifiable>,
  "rationale":      <string — why this hypothesis is scientifically worth pursuing>,
  "revision_notes": <string — what changed, which weaknesses and low-scoring dimensions were addressed, and how>
}
"""

    system = (
        f"You are a scientific hypothesis refiner working strictly within the research "
        f"domain: \"{topic}\".\n\n"
        "HARD CONSTRAINTS — violating any of these makes the output invalid:\n"
        "  • The refined hypothesis MUST remain squarely within the above research domain. "
        "Do NOT change the scientific subject matter, application area, or system being studied.\n"
        "  • Do NOT introduce a new topic, organism, disease, material, algorithm, or domain "
        "that was absent from the original hypothesis.\n\n"
        "Your job is to improve the hypothesis based on structured peer-reviewer feedback "
        "from one or more review rounds. Make it more precise, falsifiable, and rigorous "
        "WITHOUT drifting from the topic.\n\n"
        "Pay special attention to:\n"
        "  1. Dimensions that scored below 6/10 — especially NOVELTY: if novelty is low, "
        "introduce a genuinely new mechanistic angle, experimental design, or theoretical "
        "framing WITHIN the same domain that distinguishes it from existing work. "
        "Novelty means a new idea about the same subject, not a new subject.\n"
        "  2. Weaknesses that have recurred across multiple rounds — these are the issues "
        "that previous refinements failed to fix."
    )

    user = (
        f"Research topic: {topic}\n\n"
        f"## Current hypothesis (refinement round {current_round + 1})\n"
        f"Title      : {title}\n"
        f"Description: {desc}\n"
        f"Rationale  : {rationale}\n\n"
        f"{trajectory_line}\n\n"
        f"{current_feedback}\n\n"
        + (f"{history_block}\n\n" if history_block else "")
        + "Produce a revised hypothesis that:\n"
          f"1. Stays strictly within the research domain \"{topic}\" — the subject matter, "
          "application area, and system being studied must not change.\n"
          "2. Directly addresses EVERY weakness and risk listed above.\n"
          "3. Fixes any recurring issues that appeared in multiple rounds.\n"
          "4. Retains the strengths identified by reviewers.\n"
          "5. Increases scientific rigour, specificity, and falsifiability.\n"
          "6. Improves low-scoring dimensions — especially NOVELTY: novelty means a new "
          "mechanistic angle, experimental approach, or theoretical framing within the "
          f"same domain (\"{topic}\"), NOT switching to a different topic.\n\n"
          "Return STRICT JSON only — no markdown fences, no prose outside JSON."
    )

    parsed = await _call_llm_json(
        system_instructions=system,
        user_prompt=user,
        schema_hint=schema,
        context={**ctx, "call_type": "refine_hypothesis", "hyp_id": hyp_id},
        temperature=get_temperature("refinement", 0.3) or 0.3,
        model_role="refinement",
    )

    if not isinstance(parsed, dict) or not parsed.get("title"):
        return hypothesis  # fallback: return original unchanged

    refined = dict(hypothesis)
    for field in ("title", "description", "rationale", "revision_notes", "reasoning"):
        if field in parsed and parsed[field]:
            refined[field] = parsed[field]
    refined["refinement_round"] = current_round + 1

    _logger.info(
        "refine_hypothesis_done",
        extra={
            "hyp_id":       hyp_id,
            "score_before": score,
            "round":        refined["refinement_round"],
            "history_len":  len(history) if history else 0,
        },
    )
    return refined


async def generate_poc_code(
    hypothesis: dict[str, Any],
    topic: str,
    context: dict[str, Any] | None = None,
) -> str:
    """
    Generate a self-contained Python proof-of-concept script that tests the hypothesis.

    The script must be a real experiment — real libraries, real data, real metrics —
    not a simulation.  Claude chooses the appropriate tools for the hypothesis domain.

    Returns the Python source code as a string.
    """
    ctx = context or {}
    title       = hypothesis.get("title", "Untitled")
    description = hypothesis.get("description", "")
    rationale   = hypothesis.get("rationale", "")
    weaknesses  = hypothesis.get("weaknesses") or []
    risks       = hypothesis.get("risks") or []
    reasoning   = hypothesis.get("reasoning", "")
    team        = hypothesis.get("validated_by_team", "")

    system = """\
You are an expert Python programmer and scientist. Write a self-contained Python \
proof-of-concept (PoC) script that empirically tests the given research hypothesis.

CORE REQUIREMENT — REAL EXPERIMENT, NOT A SIMULATION:
The script must perform a genuine computational experiment:
- Use real algorithms and real libraries (not hand-rolled approximations).
- Use a small but semantically meaningful dataset crafted for the domain \
  (not random numbers or lorem ipsum).
- Measure real outcomes that directly test the hypothesis claim \
  (accuracy, latency, error rate, throughput, statistical difference, etc.).
- Compare at least two conditions (e.g. baseline vs proposed) so the result is meaningful.
- The verdict must be computed from the measured data — not hardcoded.

What is NOT acceptable:
- Sleeping to "simulate" latency.
- Generating random vectors and calling it a benchmark.
- Computing metrics on data that was designed to guarantee the result.
- Skipping real library calls with toy fallbacks that bypass the actual algorithm.

STRUCTURE (adapt as needed for the domain):
1. Build a small, domain-appropriate dataset or environment.
2. Implement or instantiate the components the hypothesis is about.
3. Run the experiment — measure real outcomes under at least two conditions.
4. Print a clear results table or summary.
5. State whether the data supports or refutes the hypothesis.

The script must complete within 90 seconds.
End with exactly one line:  RESULT: <concise factual summary of what was measured>

Return ONLY raw Python source code — no markdown fences, no prose explanations.\
"""

    # Build a detailed hypothesis block so the LLM has full context
    hyp_block_parts = [
        f"Hypothesis title: {title}",
        f"Description: {description}",
    ]
    if rationale:
        hyp_block_parts.append(f"Rationale: {rationale}")
    if reasoning:
        hyp_block_parts.append(f"Review reasoning: {reasoning}")
    if weaknesses:
        hyp_block_parts.append("Known weaknesses:\n" + "\n".join(f"  - {w}" for w in weaknesses))
    if risks:
        hyp_block_parts.append("Identified risks:\n" + "\n".join(f"  - {r}" for r in risks))
    if team:
        hyp_block_parts.append(f"Validated by: {team}")
    hyp_block = "\n".join(hyp_block_parts)

    user = (
        f"Research topic: {topic}\n\n"
        f"{hyp_block}\n\n"
        "Using the full context above (description, rationale, review reasoning, "
        "weaknesses and risks identified by reviewers), write a real proof-of-concept "
        "experiment that directly tests the core claim of this hypothesis.\n"
        "The experiment must address the weaknesses and risks listed — do not ignore them.\n"
        "Choose the libraries, data structures, algorithms, and metrics that best fit "
        "this specific domain — do not force an architecture that does not naturally apply.\n"
        "The result must come from actual computation on real data.\n"
        "End with a RESULT: line."
    )

    poc_model = get_model("poc_codegen", default=get_model("poc", default="claudeopus46"))
    code_text = await call_claude_llm(
        system=system,
        user=user,
        model=poc_model,
        temperature=get_temperature("poc_codegen", 1.0) or 1.0,
        ctx={**ctx, "call_type": "generate_poc_code", "hyp_id": hypothesis.get("id")},
    )

    # Strip markdown fences if the model wrapped the code despite instructions.
    code_text = code_text.strip()
    if code_text.startswith("```"):
        lines = code_text.splitlines()
        start = 1
        end = len(lines) - 1
        if lines[-1].strip() == "```":
            end = len(lines) - 1
        code_text = "\n".join(lines[start:end]).strip()

    return code_text


async def interpret_poc_results(
    hypothesis: dict[str, Any],
    code: str,
    stdout: str,
    stderr: str,
    returncode: int,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Use the LLM to interpret the PoC execution results in relation to the hypothesis.

    Returns a dict with keys: verdict, confidence, interpretation, next_steps.
    """
    ctx = context or {}
    title = hypothesis.get("title", "Untitled")

    schema = """{
  "verdict":        <"SUPPORTED" | "PARTIALLY_SUPPORTED" | "INCONCLUSIVE" | "REFUTED">,
  "confidence":     <float 0-1>,
  "interpretation": <string — what the output shows and why it supports/refutes the hypothesis>,
  "next_steps":     <string — what a researcher should do next to strengthen or falsify this>
}"""

    output_block = (
        f"Return code: {returncode}\n"
        f"STDOUT:\n{stdout[:3000]}\n"
        f"STDERR:\n{stderr[:1000]}"
    )

    system = (
        "You are a scientific reviewer interpreting the results of a computational "
        "proof-of-concept experiment. Assess whether the output supports the hypothesis, "
        "explain your reasoning, and suggest next steps. "
        "Reply with STRICT JSON matching the schema below — no markdown fences, no prose.\n"
        f"Schema:\n{schema}"
    )
    user = (
        f"Hypothesis: {title}\n\n"
        f"Execution output:\n{output_block}\n\n"
        "Return STRICT JSON only."
    )

    poc_model = get_model("poc_interpret", default=get_model("poc", default="claudeopus46"))
    raw = await call_claude_llm(
        system=system,
        user=user,
        model=poc_model,
        temperature=get_temperature("poc_interpret", 0.2) or 0.2,
        ctx={**ctx, "call_type": "interpret_poc_results", "hyp_id": hypothesis.get("id")},
    )

    # Parse JSON from Claude's response.
    parsed = None
    s = raw.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        s = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(s)
    except Exception:
        start = min(
            (s.find("{") if s.find("{") != -1 else len(s)),
            (s.find("[") if s.find("[") != -1 else len(s)),
        )
        end = max(s.rfind("}"), s.rfind("]"))
        if 0 <= start < end:
            try:
                parsed = json.loads(s[start:end + 1])
            except Exception:
                pass

    if not isinstance(parsed, dict):
        return {
            "verdict": "INCONCLUSIVE",
            "confidence": 0.0,
            "interpretation": "LLM interpretation failed to return valid JSON.",
            "next_steps": "",
        }
    return parsed


async def fix_poc_code(
    code: str,
    stderr: str,
    stdout: str,
    returncode: int,
    context: dict[str, Any] | None = None,
) -> str:
    """
    Ask Claude to fix a PoC script that failed to execute correctly.

    Provides the original code, the error output, and asks for a corrected version.
    Returns the fixed Python source code as a string.
    """
    ctx = context or {}
    system = """\
You are an expert Python debugger. You will be given a Python script that failed to run, \
along with its error output. Fix all bugs so the script runs successfully to completion.

Rules:
- Return ONLY the complete fixed Python source code — no markdown fences, no explanations.
- Do not change the experiment logic, only fix the bugs.
- Preserve the overall structure and intent of the script.
- The script must end with a line starting with "RESULT:".
- The script must complete within 90 seconds.\
"""
    error_block = f"Return code: {returncode}\nSTDERR:\n{stderr}\nSTDOUT (partial):\n{stdout[:500]}"
    user = (
        f"The following Python script failed:\n\n```python\n{code}\n```\n\n"
        f"Error output:\n{error_block}\n\n"
        "Return the complete fixed Python script."
    )
    poc_model = get_model("poc_fix", default=get_model("poc", default="claudeopus46"))
    fixed = await call_claude_llm(
        system=system,
        user=user,
        model=poc_model,
        temperature=get_temperature("poc_fix", 0.3) or 0.3,
        ctx={**ctx, "call_type": "fix_poc_code"},
    )
    fixed = fixed.strip()
    if fixed.startswith("```"):
        lines = fixed.splitlines()
        fixed = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
    return fixed
