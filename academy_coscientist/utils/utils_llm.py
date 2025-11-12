# academy_coscientist/utils/utils_llm.py

from __future__ import annotations

import json
import os
import random
import logging
from typing import Any
import asyncio

from openai import APIStatusError, AsyncOpenAI, RateLimitError

from academy_coscientist.utils.config import get_model
from academy_coscientist.utils.utils_logging import (
    get_llm_audit_path,
    record_llm_call,
    make_struct_logger,
)

_logger = make_struct_logger("utils_llm")

_last_known_embed_dim: int | None = None


def _get_openai_client() -> AsyncOpenAI:
    """
    Create a fresh AsyncOpenAI client.

    We avoid binding a single client instance to a specific event loop,
    which is what caused 'Event is bound to a different event loop'
    errors when run under different asyncio loops.
    """
    return AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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

    allow_temperature = True
    if model in {"o4-mini"}:
        allow_temperature = False

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
        llm_record["response"] = _to_safe_json(resp)
        record_llm_call(llm_record, mirror_to=audit_path)
        return content
    except (APIStatusError, RateLimitError) as e:
        llm_record["error"] = repr(e)
        record_llm_call(llm_record, mirror_to=audit_path)
        _logger.error("Chat completion failed", extra={"error": repr(e), "model": model})
        raise


async def call_reasoning_llm(
    system: str,
    user: str,
    ctx: dict[str, Any] | None = None,
    max_completion_tokens: int | None = None,
) -> str:
    model = get_model("reasoning")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return await _chat_completion(
        model=model,
        messages=messages,
        temperature=None,
        max_completion_tokens=max_completion_tokens,
        ctx=ctx,
    )


async def call_writing_llm(
    system: str,
    user: str,
    temperature: float = 0.7,
    ctx: dict[str, Any] | None = None,
    max_completion_tokens: int | None = None,
) -> str:
    model = get_model("writing")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return await _chat_completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        ctx=ctx,
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
        raise

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
) -> Any:
    """
    Ask the reasoning model for STRICT JSON following schema_hint.
    Parses & logs the JSON.

    On malformed or empty JSON, we return `{}` instead of fabricating content.
    """
    ctx = context or {}
    audit_path = ctx.get("audit_path", get_llm_audit_path())
    reasoning_model = get_model("reasoning")

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
        client = _get_openai_client()
        resp = await client.chat.completions.create(
            model=reasoning_model,
            messages=messages,
        )
        llm_text = resp.choices[0].message.content or ""
        llm_record["raw_response"] = _to_safe_json(resp)
    except (APIStatusError, RateLimitError) as e:
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

    record_llm_call(
        {
            "type": "parsed_json_result",
            "model": reasoning_model,
            "temperature": None,
            "system_instructions": system_instructions,
            "user_prompt": user_prompt,
            "schema_hint": schema_hint,
            "raw_response_text": llm_text,
            "parsed_response": parsed,
            "fallback_used": fallback_used,
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


async def brainstorm_hypotheses(
    topic: str,
    n: int,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    ctx = context or {}
    system = (
        "You are a careful, critical scientist. Generate diverse, "
        "non-redundant hypotheses for the given research topic."
    )
    user = json.dumps(
        {
            "topic": topic,
            "n": n,
            "instructions": (
                "Propose distinct, falsifiable hypotheses. "
                "Include a short description and rationale for each."
            ),
        },
        indent=2,
    )
    raw = await call_reasoning_llm(
        system=system,
        user=user,
        ctx={**ctx, "call_type": "brainstorm_hypotheses"},
        max_completion_tokens=2000,
    )

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            out = []
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    out.append(
                        {
                            "id": item.get("id", f"h_{i+1}"),
                            "title": item.get("title", f"Hypothesis {i+1}"),
                            "description": item.get("description", str(item)),
                        }
                    )
                else:
                    out.append(
                        {
                            "id": f"h_{i+1}",
                            "title": f"Hypothesis {i+1}",
                            "description": str(item),
                        }
                    )
            return out
    except Exception:
        pass

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    hypotheses: list[dict[str, Any]] = []
    current: list[str] = []
    for ln in lines:
        if ln[0].isdigit() and ln[1:3] in (". ", ") "):
            if current:
                hypotheses.append(
                    {
                        "id": f"h_{len(hypotheses)+1}",
                        "title": current[0],
                        "description": "\n".join(current),
                    }
                )
                current = []
            current.append(ln)
        else:
            current.append(ln)
    if current:
        hypotheses.append(
            {
                "id": f"h_{len(hypotheses)+1}",
                "title": current[0],
                "description": "\n".join(current),
            }
        )

    return hypotheses


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
        temperature=0.3,
        ctx={**ctx, "call_type": "agent_self_commentary"},
        max_completion_tokens=800,
    )
    return out


async def rewrite_meta_summary_with_llm(
    raw_summary: str,
    context: Any | None = None,
) -> str:
    """
    Use the writing model to rewrite and polish a meta-review summary.

    Parameters
    ----------
    raw_summary : str
        The basic meta-review draft (possibly bullet-style).
    context : Any | None
        Optional context (e.g., tournament data) to provide additional hints.

    Returns
    -------
    str
        A refined, formal paragraph ready for inclusion in the final report.
    """
    try:
        context_text = f"\n\nContext:\n{context}" if context else ""
        system = "You are a careful scientific summarizer."
        user = (
            "You are a scientific meta-review assistant.\n"
            "Rewrite the following raw meta-review summary into a clear, formal, and concise paragraph.\n"
            "Keep the tone analytical, neutral, and factual.\n"
            "Avoid introducing new claims or results not present in the text.\n\n"
            f"Raw summary:\n{raw_summary}{context_text}"
        )

        rewritten = await call_writing_llm(
            system=system,
            user=user,
            temperature=0.3,
            ctx={"call_type": "rewrite_meta_summary"},
            max_completion_tokens=500,
        )

        _logger.info(
            "rewrite_meta_summary_success",
            extra={"length": len(rewritten)},
        )
        return rewritten.strip()
    except Exception as e:
        _logger.error("rewrite_meta_summary_failed", extra={"error": str(e)})
        return raw_summary


async def summarize_portfolio_with_llm(
    portfolio_text: str,
    context: Any | None = None,
) -> str:
    """
    Use the writing model to synthesize a formal, concise summary of the
    research portfolio and methodology.

    Parameters
    ----------
    portfolio_text : str
        Description of how hypotheses were generated, reviewed, and selected.
    context : Any | None
        Optional metadata or invocation context.

    Returns
    -------
    str
        A polished, human-readable synthesis suitable for inclusion in the
        'Methodology and Selection Process' section of the report.
    """
    try:
        context_text = f"\n\nContext:\n{context}" if context else ""
        system = "You are a careful scientific summarizer."
        user = (
            "You are a scientific meta-review assistant helping draft the "
            "Methodology and Selection Process section of a research report.\n"
            "Based on the description below, write 1–3 concise paragraphs that:\n"
            "1) Summarize how candidate hypotheses were generated,\n"
            "2) Explain how they were reviewed and scored, and\n"
            "3) Describe in general terms how the final set was selected.\n"
            "Keep the tone formal, analytical, and neutral.\n"
            "Do NOT fabricate specific study results or numerical metrics.\n\n"
            f"Pipeline / portfolio description:\n{portfolio_text}{context_text}"
        )

        summary = await call_writing_llm(
            system=system,
            user=user,
            temperature=0.4,
            ctx={"call_type": "summarize_portfolio"},
            max_completion_tokens=800,
        )

        _logger.info(
            "summarize_portfolio_success",
            extra={"length": len(summary)},
        )
        return summary.strip()
    except Exception as e:
        _logger.error("summarize_portfolio_failed", extra={"error": str(e)})
        return (
            "The methodology and selection process followed the standard co-scientist "
            "pipeline of generation, review, tournament ranking, and meta-review."
        )
