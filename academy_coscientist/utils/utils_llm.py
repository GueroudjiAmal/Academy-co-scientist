# academy_coscientist/utils/utils_llm.py
from __future__ import annotations

import json
from typing import Any, Iterable, List, Optional

from openai import AsyncOpenAI
import openai
from academy_coscientist.utils.config import get_model
from academy_coscientist.utils.utils_logging import make_struct_logger

_client = AsyncOpenAI()
_logger = make_struct_logger("utils_llm")


# ---------------------------------------------------------------------------
# Low-level chat + JSON helpers
# ---------------------------------------------------------------------------


async def _chat(
    purpose: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.4,
    max_tokens: int = 2048,
    context: Optional[dict[str, Any]] = None,
) -> str:
    """
    Unified, version-safe chat wrapper for all OpenAI models.

    Handles:
      - reasoning models (o1, o3, o4-mini, etc.)
      - legacy chat models (gpt-4o, gpt-3.5-turbo)
      - parameter compatibility (`max_tokens` vs `max_completion_tokens`)
      - temperature restrictions
    """
    model = get_model(purpose)
    ctx = context or {}

    _logger.debug(
        "llm_chat_start",
        extra={"purpose": purpose, "model": model, "temperature": temperature, "max_tokens": max_tokens},
    )

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]

    # Detect if the model is a reasoning model
    is_reasoning = any(tag in model for tag in ("o1", "o3", "o4", "mini", "reasoning"))

    # Build kwargs dynamically
    kwargs = dict(model=model, messages=messages)

    # Reasoning models: temperature must be omitted
    if not is_reasoning:
        kwargs["temperature"] = temperature

    # Try the new-style param first
    try:
        resp = await _client.chat.completions.create(
            **kwargs, max_completion_tokens=max_tokens
        )
    except openai.BadRequestError as e:
        msg = str(e).lower()
        if "unsupported_parameter" in msg or "max_tokens" in msg:
            # fallback to legacy parameter name
            resp = await _client.chat.completions.create(
                **kwargs, max_tokens=max_tokens
            )
        elif "temperature" in msg or "unsupported_value" in msg:
            # retry without temperature for reasoning models
            kwargs.pop("temperature", None)
            resp = await _client.chat.completions.create(
                **kwargs, max_completion_tokens=max_tokens
            )
        else:
            raise
    except TypeError:
        # compatibility fallback for old SDKs
        resp = await _client.chat.completions.create(
            **kwargs, max_tokens=max_tokens
        )

    text = resp.choices[0].message.content or ""
    out = text.strip()

    _logger.debug(
        "llm_chat_done",
        extra={"purpose": purpose, "model": model, "context": ctx, "output_preview": out[:200]},
    )

    return out



def _extract_json_block(text: str) -> str:
    """
    Best-effort JSON extraction.

    We do *not* invent content here; we simply try to trim away markdown
    fences or stray commentary so that the remaining text is valid JSON.
    If this fails, the caller is responsible for handling the parse error.
    """
    text = text.strip()

    # Strip ```json fences if present.
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
            text = "\n".join(lines[1:-1]).strip()

    # Already-valid JSON?
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # Try slice from first { / [ to last } / ]
    first = len(text)
    for ch in ("{", "["):
        idx = text.find(ch)
        if idx != -1:
            first = min(first, idx)

    last = -1
    for ch in ("}", "]"):
        idx = text.rfind(ch)
        if idx != -1:
            last = max(last, idx)

    if first >= last or last == -1 or first == len(text):
        return text

    return text[first : last + 1].strip()


async def _call_llm_json(
    system_instructions: str,
    user_prompt: str,
    schema_hint: str,
    *,
    purpose: str = "reasoning",
    temperature: float = 0.3,
    max_tokens: int = 2048,
    context: Optional[dict[str, Any]] = None,
) -> Any:
    """
    Shared JSON helper.

    IMPORTANT: there is no deterministic JSON fabrication here.
    If the model output cannot be parsed as JSON even after light cleanup,
    json.loads(...) will raise and that exception is propagated.
    """
    sys_prompt = (
        system_instructions.strip()
        + "\n\nYou MUST reply with STRICT JSON only. "
        "Do not include markdown code fences, prose, or explanations.\n"
        "Intended JSON shape (natural language description):\n"
        f"{schema_hint.strip()}\n"
    )

    raw = await _chat(
        purpose=purpose,
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        context=context,
    )

    json_text = _extract_json_block(raw)
    return json.loads(json_text)


async def call_llm_json(
    system_instructions: str,
    user_prompt: str,
    schema_hint: str,
    *,
    purpose: str = "reasoning",
    temperature: float = 0.3,
    max_tokens: int = 2048,
    context: Optional[dict[str, Any]] = None,
) -> Any:
    """
    Public wrapper used by downstream agents (e.g. ReportAgent).

    This just forwards to `_call_llm_json` with explicit keyword arguments,
    so signatures remain stable and easy to audit.
    """
    return await _call_llm_json(
        system_instructions=system_instructions,
        user_prompt=user_prompt,
        schema_hint=schema_hint,
        purpose=purpose,
        temperature=temperature,
        max_tokens=max_tokens,
        context=context,
    )


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

_embedder_model = None


async def embed_texts(
    texts: Iterable[str],
    *,
    context: Optional[dict[str, Any]] = None,
) -> List[List[float]]:
    """
    Compute embeddings for a batch of texts.

    Behaviour:
    - If config.models.embedding starts with 'local-', we use a SMALL local
      sentence-transformer model (e.g. all-MiniLM-L6-v2).
    - Otherwise we call OpenAI's embeddings endpoint with that model name.

    This aligns with the constraint:
      - use OpenAI for LLM calls
      - only use a small local model for embeddings if needed.
    """
    from typing import cast

    global _embedder_model
    ctx = context or {}
    embed_model = get_model("embedding")

    texts_list = [str(t) for t in texts]

    # Local path
    if embed_model.startswith("local-"):
        model_name = embed_model[len("local-") :] or "all-MiniLM-L6-v2"
        if _embedder_model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            _logger.info(
                "embed_init_local",
                extra={"model": model_name},
            )
            _embedder_model = SentenceTransformer(model_name)
        vecs = _embedder_model.encode(
            texts_list,
            convert_to_numpy=False,
            show_progress_bar=False,
        )
        return [list(map(float, v)) for v in vecs]

    # Remote OpenAI embeddings
    resp = await _client.embeddings.create(
        model=embed_model,
        input=texts_list,
    )
    out: List[List[float]] = [
        cast(List[float], d.embedding) for d in resp.data
    ]

    _logger.debug(
        "embed_done",
        extra={"model": embed_model, "n": len(out), "context": ctx},
    )
    return out


# ---------------------------------------------------------------------------
# High-level helpers used by agents
# ---------------------------------------------------------------------------


async def brainstorm_hypotheses(
    topic: str,
    n: int,
    context: Optional[dict[str, Any]] = None,
) -> List[dict[str, Any]]:
    """
    Generate a list of candidate hypotheses / ideas for a topic.

    There is intentionally NO deterministic fallback here: if this call fails,
    the exception will propagate back to the calling agent.
    """
    schema_hint = """
    A JSON array of objects, each with:
      - id: short identifier string (e.g. "H1")
      - title: short, human-readable title
      - description: multi-sentence description of the hypothesis
      - rationale: explanation of why this might be true or worthwhile
      - feasibility: short text on how feasible this is to test
      - novelty: short text on novelty vs typical work
      - potential_impact: description of scientific / societal impact
      - risk_factors: list of strings for key risks or caveats
    """

    user_prompt = f"""
Topic:
{topic}

Generate EXACTLY {n} diverse, non-trivial, testable hypotheses or research
ideas about the topic above. They should:

- be as specific and operationalizable as possible
- cover multiple distinct mechanisms / interventions / perspectives
- be plausible to investigate in practice

Return ONLY JSON as described in the schema hint.
"""

    ideas = await _call_llm_json(
        system_instructions=(
            "You are an expert research scientist in a multi-agent system. "
            "Your job is to propose high-quality hypotheses."
        ),
        user_prompt=user_prompt,
        schema_hint=schema_hint,
        purpose="reasoning",
        temperature=0.6,
        max_tokens=4096,
        context=context,
    )

    if not isinstance(ideas, list):
        raise RuntimeError(f"brainstorm_hypotheses expected list, got {type(ideas)!r}")
    return ideas


async def review_hypothesis(
    hypothesis: dict[str, Any],
    retrieved: Optional[List[dict[str, Any]]] = None,
    context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Peer-review style critique for a single hypothesis.

    This function is explicitly used by ReviewAgent._review_single and is what
    prevents the system from falling back to deterministic, hard-coded
    heuristics when the LLM is available.
    """
    schema_hint = """
    A JSON object with:
      - strengths: list of strings
      - weaknesses: list of strings
      - feasibility_risks: list of strings
      - suggested_improvements: list of strings
      - risks: list of strings
      - score: number in [0, 1]
      - confidence: number in [0, 1]
      - recommendation: one of
          ["strong_accept","accept","weak_accept","borderline",
           "weak_reject","reject"]
      - notes: free-form string with any extra comments
    """

    title = hypothesis.get("title") or hypothesis.get("name") or f"Hypothesis {hypothesis.get('id')}"
    desc = hypothesis.get("description") or hypothesis.get("text") or ""

    if retrieved:
        ctx_block_lines = []
        for i, doc in enumerate(retrieved[:5]):
            text = str(doc.get("text") or "")
            score = doc.get("score")
            header = f"Doc {i+1}"
            if score is not None:
                header += f" (score={score})"
            ctx_block_lines.append(f"{header}:\n{text}\n")
        ctx_block = "\n".join(ctx_block_lines)
    else:
        ctx_block = "No external documents were retrieved for this hypothesis."

    user_prompt = f"""
You are a critical but fair scientific peer reviewer.

Hypothesis JSON:
{json.dumps(hypothesis, ensure_ascii=False, indent=2)}

Retrieved context (may be noisy or incomplete):
{ctx_block}

Provide a detailed, calibrated critique and return it strictly as JSON according
to the schema hint.
"""

    critique = await _call_llm_json(
        system_instructions=(
            "You are reviewing a single scientific hypothesis. Be concrete, "
            "specific, and well-calibrated in your judgments."
        ),
        user_prompt=user_prompt,
        schema_hint=schema_hint,
        purpose="reasoning",
        temperature=0.4,
        max_tokens=3072,
        context=context,
    )

    if not isinstance(critique, dict):
        raise RuntimeError(f"review_hypothesis expected dict, got {type(critique)!r}")

    return critique


async def evolve_hypothesis(
    idea: dict[str, Any],
    critiques: dict[str, Any],
    context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Improve / refine an existing hypothesis using reviews.
    """
    schema_hint = """
    A JSON object describing a refined hypothesis with fields:
      - id
      - title
      - description
      - rationale
      - feasibility
      - novelty
      - potential_impact
      - risk_factors (list of strings)
    """

    user_prompt = f"""
You are an AI co-scientist. Refine this hypothesis using the critiques.

Original idea JSON:
{json.dumps(idea, ensure_ascii=False, indent=2)}

Aggregated critiques JSON:
{json.dumps(critiques, ensure_ascii=False, indent=2)}

Make the refined version more precise, testable, and decision-relevant.
Return ONLY JSON following the schema hint.
"""

    refined = await _call_llm_json(
        system_instructions="Refine hypotheses given critiques while preserving the core idea.",
        user_prompt=user_prompt,
        schema_hint=schema_hint,
        purpose="reasoning",
        temperature=0.5,
        max_tokens=3072,
        context=context,
    )

    if not isinstance(refined, dict):
        raise RuntimeError(f"evolve_hypothesis expected dict, got {type(refined)!r}")
    return refined


async def summarize_background(
    topic: str,
    context: Optional[dict[str, Any]] = None,
) -> str:
    """
    Background literature-style summary for a topic.

    Used by LiteratureAgent. Returns plain text (no markdown).
    """
    user_prompt = f"""
Topic:
{topic}

Write a compact but information-dense background summary (around 800–1200 words)
as if for a scientifically literate collaborator. Cover:

- main sub-areas or facets of the topic
- common methodologies and their limitations
- known sources of bias, toxicity, or failure modes if relevant
- key open questions or promising directions

Plain text only, no markdown headings or bullet syntax.
"""

    summary = await _chat(
        purpose="reasoning",
        system_prompt="You write clear, neutral, and precise scientific background summaries.",
        user_prompt=user_prompt,
        temperature=0.4,
        max_tokens=4096,
        context=context,
    )
    return summary


async def agent_self_commentary(
    agent_name: str,
    state_snapshot: dict[str, Any],
    context: Optional[dict[str, Any]] = None,
) -> str:
    """
    Short reflective note used by SupervisorAgent for logging / introspection.
    """
    compact_state = json.dumps(state_snapshot, ensure_ascii=False)[:2000]

    user_prompt = f"""
You are {agent_name}, an agent in a multi-agent co-scientist pipeline.

Here is a JSON snapshot of your current state:
{compact_state}

Write a very short self-commentary (<= 200 words) about:
- what you are about to do next
- what you should pay particular attention to
- any salient risks or failure modes you want to flag

Plain text only.
"""

    commentary = await _chat(
        purpose="reasoning",
        system_prompt="You are a self-reflective AI component describing its own next steps.",
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=512,
        context=context,
    )
    return commentary
