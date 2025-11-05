# academy_coscientist/utils/utils_llm.py
from __future__ import annotations

import os
import json
import random
from typing import Any, Dict, List, Tuple, Optional
from openai import AsyncOpenAI
from openai import RateLimitError, APIStatusError

from academy_coscientist.utils.utils_logging import record_llm_call, get_llm_audit_path
from academy_coscientist.utils.config import get_model

# ------------------------- OpenAI / Embedding setup ---------------------------

_LOCAL_EMBEDDER_CACHE: Dict[str, Any] = {}

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore

_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

_last_known_embed_dim: Optional[int] = None


def _is_local_embedding_model(name: str) -> bool:
    return str(name).strip().lower().startswith("local-")


def _local_model_name_from_config(name: str) -> str:
    return name.split("local-", 1)[1] if "local-" in name else name


def _get_local_embedder(model_name: str):
    global _LOCAL_EMBEDDER_CACHE, SentenceTransformer
    if model_name in _LOCAL_EMBEDDER_CACHE:
        return _LOCAL_EMBEDDER_CACHE[model_name]
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers not installed, "
            "but a local embedding model was requested."
        )
    model = SentenceTransformer(model_name)
    _LOCAL_EMBEDDER_CACHE[model_name] = model
    return model


# --------------------------- Safe JSON serializers ----------------------------

def _to_safe_json(x, depth: int = 0, max_depth: int = 4):
    """Recursively convert SDK / pydantic objects to plain JSONable structures."""
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if depth >= max_depth:
        s = str(x)
        return s[:2000] + ("..." if len(s) > 2000 else "")
    if isinstance(x, dict):
        return {k: _to_safe_json(v, depth + 1, max_depth) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_to_safe_json(v, depth + 1, max_depth) for v in x]
    # pydantic v2 models in the OpenAI SDK expose model_dump()
    if hasattr(x, "model_dump") and callable(getattr(x, "model_dump")):
        try:
            return _to_safe_json(x.model_dump(), depth + 1, max_depth)
        except Exception:
            pass
    # generic objects
    try:
        d = vars(x)
        return _to_safe_json(d, depth + 1, max_depth)
    except Exception:
        s = str(x)
        return s[:2000] + ("..." if len(s) > 2000 else "")


def _extract_reasoning_text_from_dump(d: dict) -> str:
    """
    Best-effort: walk the dumped response dict to collect any 'reasoning' items' text.
    The Responses API represents output as a list of items; some items have type='reasoning'.
    """
    texts = []

    def walk(node):
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


# ------------------------------ Embeddings ------------------------------------

async def _embed_with_local_model(
    model_alias: str,
    texts: List[str],
    ctx: Dict[str, Any],
) -> List[List[float]]:
    global _last_known_embed_dim

    audit_path = ctx.get("audit_path", get_llm_audit_path())
    raw_name = _local_model_name_from_config(model_alias)

    record_llm_call(
        {
            "type": "embed_request_local",
            "model": raw_name,
            "alias": model_alias,
            "temperature": None,
            "input_texts_preview": [t[:300] for t in texts[:3]],
            "count": len(texts),
            "context": ctx,
        },
        mirror_to=audit_path,
    )

    try:
        st_model = _get_local_embedder(raw_name)
        vectors_np = st_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vectors = [v.tolist() for v in vectors_np]

        if vectors:
            _last_known_embed_dim = len(vectors[0])

        record_llm_call(
            {
                "type": "embed_response_local",
                "model": raw_name,
                "alias": model_alias,
                "temperature": None,
                "count": len(vectors),
                "dim": _last_known_embed_dim,
                "context": ctx,
            },
            mirror_to=audit_path,
        )
        return vectors

    except Exception as e:
        dim = _last_known_embed_dim if _last_known_embed_dim is not None else 256
        fallback_vectors = [[0.0] * dim for _ in texts]

        record_llm_call(
            {
                "type": "embed_error_local",
                "model": raw_name,
                "alias": model_alias,
                "temperature": None,
                "error": repr(e),
                "used_fallback_dim": dim,
                "context": ctx,
            },
            mirror_to=audit_path,
        )
        return fallback_vectors


async def _embed_with_openai_model(
    model_name: str,
    texts: List[str],
    ctx: Dict[str, Any],
) -> List[List[float]]:
    global _last_known_embed_dim

    audit_path = ctx.get("audit_path", get_llm_audit_path())

    record_llm_call(
        {
            "type": "embed_request_openai",
            "model": model_name,
            "temperature": None,
            "input_texts_preview": [t[:300] for t in texts[:3]],
            "count": len(texts),
            "context": ctx,
        },
        mirror_to=audit_path,
    )

    try:
        resp = await _client.embeddings.create(
            model=model_name,
            input=texts,
        )
        vectors = [d.embedding for d in resp.data]

        if vectors and isinstance(vectors[0], list):
            _last_known_embed_dim = len(vectors[0])

        record_llm_call(
            {
                "type": "embed_response_openai",
                "model": model_name,
                "temperature": None,
                "count": len(vectors),
                "dim": _last_known_embed_dim,
                "context": ctx,
            },
            mirror_to=audit_path,
        )
        return vectors

    except Exception as e:
        dim = _last_known_embed_dim if _last_known_embed_dim is not None else 256
        fallback_vectors = [[0.0] * dim for _ in texts]

        record_llm_call(
            {
                "type": "embed_error_openai",
                "model": model_name,
                "temperature": None,
                "error": repr(e),
                "used_fallback_dim": dim,
                "context": ctx,
            },
            mirror_to=audit_path,
        )
        return fallback_vectors


async def embed_texts(
    texts: List[str],
    context: Optional[Dict[str, Any]] = None,
) -> List[List[float]]:
    """
    Front-door for embedding calls.
    """
    ctx = context or {}
    embed_model = get_model("embedding")

    if _is_local_embedding_model(embed_model):
        return await _embed_with_local_model(embed_model, texts, ctx)
    else:
        return await _embed_with_openai_model(embed_model, texts, ctx)


# --------------------------- Text LLM helpers ---------------------------------

async def _call_openai_responses(
    model: str,
    system_instructions: str,
    user_input: str,
    allow_temperature: bool = False,
    temperature_value: float = 0.4,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Low-level Responses API wrapper (text output).
    Fully logged to llm_calls.jsonl with safe, JSON-serializable payloads.
    """
    ctx = context or {}
    audit_path = ctx.get("audit_path", get_llm_audit_path())

    llm_record: Dict[str, Any] = {
        "type": "responses_call",
        "model": model,
        "temperature": temperature_value if allow_temperature else None,
        "system_instructions": system_instructions,
        "user_input": user_input,
        "context": ctx,
    }

    try:
        if allow_temperature:
            resp = await _client.responses.create(
                model=model,
                instructions=system_instructions,
                input=user_input,
                temperature=temperature_value,
            )
        else:
            resp = await _client.responses.create(
                model=model,
                instructions=system_instructions,
                input=user_input,
            )

        text = resp.output_text.strip()

        # Safely serialize for logs
        try:
            dumped = resp.model_dump()  # pydantic v2 on SDK objects
        except Exception:
            dumped = _to_safe_json(resp)  # fallback, still JSONable

        reasoning_text = ""
        try:
            if isinstance(dumped, dict):
                reasoning_text = _extract_reasoning_text_from_dump(dumped)
        except Exception:
            reasoning_text = ""

        llm_record["raw_response_text"] = text
        llm_record["parsed_response"] = text
        llm_record["reasoning"] = {
            "reasoning_text": reasoning_text,
            "response_dump_snippet": _to_safe_json(dumped),
        }
        llm_record["error"] = None

        record_llm_call(llm_record, mirror_to=audit_path)
        return text

    except (RateLimitError, APIStatusError, Exception) as e:
        llm_record["raw_response_text"] = ""
        llm_record["parsed_response"] = ""
        llm_record["reasoning"] = None
        llm_record["error"] = repr(e)
        record_llm_call(llm_record, mirror_to=audit_path)
        return ""


# ---------------------------------------------------------------------------
# Text-only LLM helper with robust fallback
# ---------------------------------------------------------------------------

import json
import logging

logger = logging.getLogger(__name__)

# Make sure this global exists somewhere in your file:
try:
    LLM_DEGRADED
except NameError:
    LLM_DEGRADED = False  # type: ignore[assignment]


async def _call_llm_text(system_msg: str, user_msg: str) -> str:
    """
    Thin wrapper around the OpenAI Responses API that returns plain text.

    If no client/model is available or an error occurs, it falls back to
    returning the user message (deterministic, non-empty).
    """
    global LLM_DEGRADED

    # If the client or model isn't configured, degrade gracefully
    if "_client" not in globals() or "_MODEL" not in globals() or globals().get("_client") is None:
        LLM_DEGRADED = True
        logger.warning(
            "LLM client/model not available in _call_llm_text; "
            "falling back to deterministic text."
        )
        return user_msg

    client = globals()["_client"]
    model = globals()["_MODEL"]

    try:
        resp = await client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )

        # Try to extract text from Responses API output
        try:
            output_item = resp.output[0]
            content_item = output_item.content[0]

            # Newer SDK: content_item.text.value
            text_obj = getattr(content_item, "text", None)
            if text_obj is not None:
                value = getattr(text_obj, "value", None)
                if isinstance(value, str) and value.strip():
                    return value

            # Fallback if it's a dict-like structure
            if isinstance(content_item, dict):
                t = content_item.get("text")
                if isinstance(t, dict) and "value" in t and isinstance(t["value"], str):
                    return t["value"]
                if isinstance(t, str) and t.strip():
                    return t

        except Exception:
            # If parsing fails, fall back to stringifying the whole response
            logger.debug("Failed to parse Responses API text output; using str(resp).")

        # Last-ditch: string representation of response
        s = str(resp)
        return s if s.strip() else user_msg

    except Exception as e:
        LLM_DEGRADED = True
        logger.exception(
            "LLM text call failed in _call_llm_text; using deterministic fallback.",
            extra={"error": repr(e)},
        )
        return user_msg


# ---------------------------------------------------------------------------
# Generic structured expansion helper with deterministic fallback
# ---------------------------------------------------------------------------

async def structured_expand(*args, **kwargs) -> str:
    """
    Generic helper for section-level expansions with LLM + deterministic fallback.

    It is intentionally permissive so it can handle different call styles, e.g.:

        await structured_expand(system_msg, user_msg, schema_hint)
        await structured_expand(system_msg=..., user_msg=..., schema_hint=...)
        await structured_expand(system_msg=..., user_msg=..., fallback_text="...")

    Behavior:
      - If `schema_hint` is provided, it calls `_call_llm_json(...)` and tries
        to convert the result into text.
      - Otherwise, it calls `_call_llm_text(...)`.
      - On *any* failure or empty response, it returns a deterministic,
        non-empty fallback based on `fallback_text` or `user_msg`.
    """
    global LLM_DEGRADED

    system_msg = None
    user_msg = None
    schema_hint = None
    fallback_text = None

    # Positional unpacking: (system_msg, user_msg, schema_hint?)
    if args:
        if len(args) >= 1:
            system_msg = args[0]
        if len(args) >= 2:
            user_msg = args[1]
        if len(args) >= 3:
            schema_hint = args[2]

    # Keyword overrides
    system_msg = kwargs.get("system_msg", system_msg)
    user_msg = kwargs.get("user_msg", user_msg)
    schema_hint = kwargs.get("schema_hint", schema_hint)
    fallback_text = kwargs.get("fallback_text", fallback_text)

    # Ensure we have something deterministic to return if everything else fails
    if user_msg is None:
        user_msg = "No user prompt provided."

    # If system message is missing, still proceed — LLMs can handle that.
    if system_msg is None:
        system_msg = "You are a helpful scientific writing assistant."

    # If no LLM JSON helper is available, skip to text-only mode
    use_json = schema_hint is not None and "_call_llm_json" in globals()

    try:
        if use_json:
            data = await globals()["_call_llm_json"](system_msg, user_msg, schema_hint)

            # Try to interpret the JSON-structured response as text
            if isinstance(data, dict):
                # If the schema has a 'text' or 'content' field, use that
                for key in ("text", "content", "body"):
                    val = data.get(key)
                    if isinstance(val, str) and val.strip():
                        return val
                # Otherwise pretty-print the JSON as a deterministic representation
                return json.dumps(data, indent=2, sort_keys=True)

            if isinstance(data, list):
                # Join list items as bullet points
                lines = [f"- {str(item)}" for item in data]
                return "\n".join(lines)

            # Fallback: string representation
            s = str(data)
            return s if s.strip() else (fallback_text or user_msg)

        # Text-only path
        text = await _call_llm_text(system_msg, user_msg)
        if text and text.strip():
            return text

        # If we reach here, treat as failure and fall through to deterministic fallback
        raise ValueError("Empty LLM response in structured_expand")

    except Exception as e:
        LLM_DEGRADED = True
        logger.exception(
            "structured_expand LLM failed; using deterministic fallback.",
            extra={"error": repr(e)},
        )

        if fallback_text and str(fallback_text).strip():
            return str(fallback_text)

        # Generic deterministic fallback using the prompt itself
        return (
            "LLM expansion unavailable. Deterministic summary based on the prompt:\n\n"
            + str(user_msg)
        )


async def _call_llm_json(
    system_instructions: str,
    user_prompt: str,
    schema_hint: str,
    context: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Ask the reasoning model for STRICT JSON following schema_hint.
    Parse + log; if empty, parsed becomes {} and a fallback can be applied upstream.
    """
    ctx = context or {}
    audit_path = ctx.get("audit_path", get_llm_audit_path())

    reasoning_model = get_model("reasoning")

    formatted_user_prompt = (
        f"{user_prompt}\n\n"
        "You MUST respond with ONLY valid JSON, no markdown fences, "
        "no commentary, and only the requested keys.\n"
        "Target JSON shape:\n"
        f"{schema_hint}\n"
        "Return ONLY that JSON."
    )

    llm_text = await _call_openai_responses(
        model=reasoning_model,
        system_instructions=system_instructions,
        user_input=formatted_user_prompt,
        allow_temperature=False,
        context={
            **ctx,
            "call_type": "json",
            "schema_hint": schema_hint,
        },
    )

    fallback_used = False
    if not llm_text:
        fallback_used = True
        llm_text = "{}"

    # attempt strict JSON recovery
    start_obj = llm_text.find("{")
    start_arr = llm_text.find("[")
    if start_obj == -1 or (start_arr != -1 and start_arr < start_obj):
        start_idx = start_arr
    else:
        start_idx = start_obj

    if start_idx == -1:
        parsed: Any = {}
    else:
        end_curly = llm_text.rfind("}")
        end_brack = llm_text.rfind("]")
        end_idx = max(end_curly, end_brack)
        candidate = llm_text[start_idx : end_idx + 1]
        try:
            parsed = json.loads(candidate)
        except Exception:
            parsed = {}

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


# ---------------- Agent self commentary (used by Supervisor) ------------------

async def agent_self_commentary(
    agent_name: str,
    state_snapshot: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> str:
    reasoning_model = get_model("reasoning")
    compact_state = json.dumps(state_snapshot, ensure_ascii=False)[:2000]

    system_msg = (
        "You are an autonomous research audit assistant. "
        "Your job: briefly summarize an agent's current state for engineers. "
        "Be concise (1-2 sentences), concrete, and factual. "
        "Do not invent data."
    )

    user_msg = (
        f"Agent `{agent_name}` internal snapshot:\n"
        f"{compact_state}\n\n"
        "Write 1-2 sentence status, no bullet points."
    )

    text = await _call_openai_responses(
        model=reasoning_model,
        system_instructions=system_msg,
        user_input=user_msg,
        allow_temperature=False,
        context={
            **(context or {}),
            "call_type": "self_commentary",
            "agent_name": agent_name,
        },
    )

    return text or f"[{agent_name} status unavailable]"


# ---------------------- Local fallbacks (no LLM needed) -----------------------

def _seed_titles(topic: str, n: int) -> List[str]:
    base = [
        "Benchmarking Reliability Under Distribution Shift",
        "Agent Self-Verification and Tool-Use Checks",
        "Uncertainty-Aware Planning and Halting",
        "Counterfactual Data Augmentation for Robustness",
        "Eval Harness for Scientific Claims",
        "Memory Hygiene & Provenance Tracking",
        "Multi-Agent Cross-Examination",
        "Grounded Retrieval with Trust Signals",
        "Safety-Valve: Conservative Decoding on Risk",
        "Rapid Postmortem & Feedback Loops",
        "Human-in-the-Loop Triage for Edge Cases",
        "Automated Reproduction Pipelines",
        "Spec-First Protocol Generation",
        "Open Lab Notebook Automation",
        "Causal Tests for Tool Integration",
    ]
    random.seed(abs(hash(topic)) % (2**32))
    random.shuffle(base)
    return base[:max(10, n)]

def _tshirt(i: int) -> str:
    return ["XS", "S", "M", "L", "XL"][i % 5]

def _timeline(i: int) -> int:
    return [2, 4, 6, 8, 12, 16][i % 6]

def _confidence(i: int) -> float:
    return round(0.5 + 0.05 * ((i % 6) - 2), 2)  # ~0.4..0.6 band

def _metrics(topic: str) -> List[Dict[str, str]]:
    return [
        {"metric": "reproduction_rate", "target": "≥ 90% within 2 attempts", "timeframe": "pilot"},
        {"metric": "hallucination_rate", "target": "≤ 1% on held-out tasks", "timeframe": "R1"},
        {"metric": "verification_coverage", "target": "≥ 85% of steps auto-checked", "timeframe": "R1"},
    ]

def _plan() -> List[str]:
    return [
        "Define scope and success criteria with domain experts",
        "Assemble or synthesize a representative task suite",
        "Implement baseline and instrumentation for traces",
        "Prototype approach; run A/B against baseline",
        "Measure predefined metrics; analyze failures",
        "Harden implementation; write reproducible recipe",
        "Ship doc + code + dashboards",
    ]

def _risks() -> List[str]:
    return [
        "Metric gaming or overfitting to eval set",
        "Tool/API variability across environments",
        "Data leakage or unnoticed shortcuts",
    ]

def _mitigations() -> List[str]:
    return [
        "Holdout rotations and adversarial evals",
        "Version pinning and containerized runners",
        "Provenance checks and red-team probes",
    ]

def _local_idea(topic: str, title: str, i: int) -> Dict[str, Any]:
    return {
        "title": title,
        "rationale": f"Increase reliability for '{topic}' by addressing common failure modes and adding verifiable controls.",
        "mechanism": "Introduce checks, redundancy, and robust training/eval signals to reduce silent failures.",
        "novelty": "Combines verification, provenance, and uncertainty control into a practical pipeline.",
        "dependencies": "Task suite, logging/tracing infra, container build, basic RAG/store.",
        "plan": _plan(),
        "resources": "1 PM, 2 Eng, 1 DS; GPU access; CI/CD; observability stack",
        "metrics": _metrics(topic),
        "risks": _risks(),
        "mitigations": _mitigations(),
        "expected_outcome": "Measurable lift in reproduction_rate and drop in hallucination_rate on target domain.",
        "confidence": _confidence(i),
        "timeline_weeks": _timeline(i),
        "cost_tshirt": _tshirt(i),
    }

def _fmt_metrics(metrics: List[Dict[str, Any]]) -> str:
    lines = []
    for m in metrics or []:
        metric = m.get("metric", "")
        target = m.get("target", "")
        tf = m.get("timeframe", "")
        parts = [p for p in [metric, target, tf] if p]
        if parts:
            lines.append(f"- " + " | ".join(parts))
    return "\n".join(lines) if lines else "- (none specified)"

def _fmt_list(name: str, items: List[str]) -> str:
    if not items:
        return f"{name}:\n- (none)"
    return f"{name}:\n" + "\n".join(f"- {it}" for it in items)

def _local_meta(ideas: List[Tuple[str, float, Dict[str, Any]]]) -> str:
    if not ideas:
        return "(no ideas)"
    themes = {
        "verification": 0,
        "evals": 0,
        "retrieval": 0,
        "planning": 0,
        "tooling": 0,
        "observability": 0,
    }
    for _hid, _elo, idea in ideas:
        t = (idea.get("title","") + " " + idea.get("rationale","")).lower()
        if any(k in t for k in ["verify", "check", "self"]): themes["verification"] += 1
        if "eval" in t or "benchmark" in t: themes["evals"] += 1
        if "retriev" in t or "rag" in t: themes["retrieval"] += 1
        if "plan" in t or "uncertainty" in t: themes["planning"] += 1
        if "tool" in t or "api" in t: themes["tooling"] += 1
        if "log" in t or "trace" in t or "observ" in t: themes["observability"] += 1

    ordered = sorted(themes.items(), key=lambda x: x[1], reverse=True)
    lines = ["**Themes (counts):** " + ", ".join([f"{k}:{v}" for k,v in ordered if v>0])]

    # simple roadmap by ELO, batching
    sorted_ideas = sorted(ideas, key=lambda x: x[1], reverse=True)
    batches = [sorted_ideas[i:i+3] for i in range(0, min(10, len(sorted_ideas)), 3)]
    lines.append("\n**Prioritized roadmap:**")
    for idx, batch in enumerate(batches, start=1):
        ids = ", ".join([f"{hid} (ELO {elo:.0f})" for hid, elo, _ in batch])
        lines.append(f"- Phase {idx}: {ids}")

    return "\n".join(lines)


# ---------------- Enriched agent-level helpers (with fallbacks) ---------------

async def brainstorm_hypotheses(
    topic: str,
    n: int,
    context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    LLM JSON brainstorm; if empty, produce a high-quality local top-10 set.
    """
    n = max(n, 10)
    system_msg = (
        "You are a senior technical program designer. "
        "Propose distinct, testable, HIGH-LEVERAGE ideas for the given topic. "
        "Each idea must include a concrete plan, metrics, timeline, resources, "
        "risks, mitigations, expected outcome and confidence."
    )
    user_msg = (
        f"Generate {n} distinct ideas for:\n{topic}\n\n"
        "For each idea, fill these fields:\n"
        "- title\n- rationale\n- mechanism\n- novelty\n- dependencies\n"
        "- plan (5-8 bullets)\n- resources\n- metrics [{metric,target,timeframe}]\n"
        "- risks (2-3)\n- mitigations (1-2 per risk)\n"
        "- expected_outcome\n- confidence (0..1)\n- timeline_weeks (int)\n- cost_tshirt (XS..XL)\n"
    )
    schema_hint = """
{
  "ideas": [
    {
      "title": "string",
      "rationale": "string",
      "mechanism": "string",
      "novelty": "string",
      "dependencies": "string",
      "plan": ["string", "string"],
      "resources": "string",
      "metrics": [{"metric":"string","target":"string","timeframe":"string"}],
      "risks": ["string","string"],
      "mitigations": ["string","string"],
      "expected_outcome": "string",
      "confidence": 0.0,
      "timeline_weeks": 0,
      "cost_tshirt": "XS"
    }
  ]
}
"""
    data = await _call_llm_json(system_msg, user_msg, schema_hint, context=context)
    ideas: List[Dict[str, Any]] = []
    for it in data.get("ideas", []):
        if isinstance(it, dict) and it.get("title"):
            ideas.append(it)

    if ideas:
        return ideas

    # -------- Local fallback: deterministic top-10 ideas --------
    titles = _seed_titles(topic, n)
    local_ideas = [_local_idea(topic, t, i) for i, t in enumerate(titles)]
    record_llm_call(
        {
            "type": "brainstorm_fallback_local",
            "model": "offline-fallback",
            "temperature": None,
            "raw_response_text": json.dumps(local_ideas)[:5000],
            "parsed_response": None,
            "reasoning": "LLM brainstorm empty; generated deterministic local ideas.",
            "error": None,
            "context": {**(context or {}), "call_type": "brainstorm_local"},
        },
        mirror_to=get_llm_audit_path(),
    )
    return local_ideas


async def critique_hypothesis(
    idea: Dict[str, Any],
    retrieved_contexts: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    evidence_block = ""
    if retrieved_contexts:
        trimmed = [c.strip() for c in retrieved_contexts[:8]]
        joined = "\n\n---\n".join(trimmed)
        evidence_block = "\n\n[EVIDENCE PACK]\n" + joined + "\n"

    system_msg = (
        "You are a technical review panel. Evaluate feasibility, payoff, clarity, "
        "risk, dependencies, and realism of plan and metrics. Offer pointed improvements."
    )
    user_msg = (
        "Evaluate this idea in detail and propose improvements.\n\n"
        f"IDEA JSON:\n{json.dumps(idea, ensure_ascii=False)}\n"
        f"{evidence_block}\n"
        "Return: strengths, weaknesses, feasibility_risks(2-4), "
        "suggested_improvements(3-6), score(0..1)."
    )
    schema_hint = """
{
  "strengths": "string",
  "weaknesses": "string",
  "feasibility_risks": ["string","string"],
  "suggested_improvements": ["string","string","string"],
  "score": 0.0
}
"""
    data = await _call_llm_json(system_msg, user_msg, schema_hint, context=context)
    return {
        "strengths": data.get("strengths", ""),
        "weaknesses": data.get("weaknesses", ""),
        "feasibility_risks": data.get("feasibility_risks", []),
        "suggested_improvements": data.get("suggested_improvements", []),
        "score": float(data.get("score", 0.6)),
    }


async def debate_and_choose(
    i1_text: str,
    i2_text: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    system_msg = (
        "You are a portfolio allocator. Pick the better near-term bet considering "
        "expected value (payoff*probability), time-to-insight, plan realism, and risk."
    )
    user_msg = (
        "Compare proposals and pick a single winner (1 or 2) with 2-4 sentences of reasoning.\n\n"
        f"Proposal 1:\n{i1_text}\n\n"
        f"Proposal 2:\n{i2_text}\n\n"
    )
    schema_hint = """
{
  "winner": 1,
  "reasoning": "string"
}
"""
    data = await _call_llm_json(system_msg, user_msg, schema_hint, context=context)
    raw_winner = data.get("winner", 1)
    try:
        winner_int = int(raw_winner)
        if winner_int not in (1, 2):
            winner_int = 1
    except Exception:
        winner_int = 1
    return {"winner": winner_int, "reasoning": data.get("reasoning", "")}


async def evolve_hypothesis(
    idea: Dict[str, Any],
    critiques: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    system_msg = (
        "You are a refiner. Incorporate critiques to make the plan more specific, "
        "reduce risk, clarify metrics/targets, and improve expected value."
    )
    user_msg = (
        "Refine the idea with these critiques. Keep structure and add specificity where missing.\n\n"
        f"IDEA JSON:\n{json.dumps(idea, ensure_ascii=False)}\n\n"
        f"CRITIQUES JSON:\n{json.dumps(critiques, ensure_ascii=False)}\n\n"
        "Return the full improved IDEA JSON (same keys as input)."
    )
    schema_hint = """
{
  "title": "string",
  "rationale": "string",
  "mechanism": "string",
  "novelty": "string",
  "dependencies": "string",
  "plan": ["string","string"],
  "resources": "string",
  "metrics": [{"metric":"string","target":"string","timeframe":"string"}],
  "risks": ["string","string"],
  "mitigations": ["string","string"],
  "expected_outcome": "string",
  "confidence": 0.0,
  "timeline_weeks": 0,
  "cost_tshirt": "XS"
}
"""
    data = await _call_llm_json(
        system_instructions=system_msg,
        user_prompt=user_msg,
        schema_hint=schema_hint,
        context=context,
    )

    refined = {
        "title": data.get("title", idea.get("title", "")),
        "rationale": data.get("rationale", idea.get("rationale", "")),
        "mechanism": data.get("mechanism", idea.get("mechanism", "")),
        "novelty": data.get("novelty", idea.get("novelty", "")),
        "dependencies": data.get("dependencies", idea.get("dependencies", "")),
        "plan": data.get("plan", idea.get("plan", [])),
        "resources": data.get("resources", idea.get("resources", "")),
        "metrics": data.get("metrics", idea.get("metrics", [])),
        "risks": data.get("risks", idea.get("risks", [])),
        "mitigations": data.get("mitigations", idea.get("mitigations", [])),
        "expected_outcome": data.get("expected_outcome", idea.get("expected_outcome", "")),
        "confidence": float(data.get("confidence", idea.get("confidence", 0.6))),
        "timeline_weeks": int(data.get("timeline_weeks", idea.get("timeline_weeks", 6))),
        "cost_tshirt": data.get("cost_tshirt", idea.get("cost_tshirt", "M")),
    }
    return refined


async def summarize_background(
    topic: str,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    system_msg = (
        "Write a compact background summary for experts: current approaches, "
        "main challenges, open questions, and recent trends. Avoid fluff."
    )
    user_msg = (
        f"Topic: {topic}\nSummarize current understanding, approaches, limitations, and open questions."
    )

    writing_model = get_model("writing")

    text = await _call_openai_responses(
        model=writing_model,
        system_instructions=system_msg,
        user_input=user_msg,
        allow_temperature=True,
        temperature_value=0.4,
        context={
            **(context or {}),
            "call_type": "background_summary",
            "topic": topic,
        },
    )
    return text or f"[OFFLINE SUMMARY for {topic}]"


async def synthesize_meta_report(
    leaderboard: List[Tuple[str, float, Dict[str, Any]]],
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Try LLM meta report; if empty, synthesize locally (and log).
    """
    system_msg = (
        "Synthesize portfolio: group themes, identify cross-cutting risks, "
        "shared dependencies, opportunities to share data/tools, and a short prioritized roadmap."
    )

    slim = []
    for hid, elo, idea in leaderboard:
        slim.append({
            "id": hid,
            "elo": elo,
            "title": idea.get("title", ""),
            "confidence": idea.get("confidence", 0.6),
            "timeline_weeks": idea.get("timeline_weeks", 6),
            "cost_tshirt": idea.get("cost_tshirt", "M"),
            "key_metrics": idea.get("metrics", [])[:2],
        })

    user_msg = (
        "Given this portfolio, write:\n"
        "1) Themes across ideas\n"
        "2) Shared risks/mitigations\n"
        "3) Shared dependencies and data/tools that could be centralized\n"
        "4) A rough prioritized roadmap across the top 10 ideas\n\n"
        f"PORTFOLIO SLIM JSON:\n{json.dumps(slim, ensure_ascii=False)}"
    )

    reasoning_model = get_model("reasoning")

    text = await _call_openai_responses(
        model=reasoning_model,
        system_instructions=system_msg,
        user_input=user_msg,
        allow_temperature=False,
        context={
            **(context or {}),
            "call_type": "meta_report",
        },
    )
    if text:
        return text

    local = _local_meta(leaderboard)
    record_llm_call(
        {
            "type": "meta_report_fallback_local",
            "model": "offline-fallback",
            "temperature": None,
            "raw_response_text": local[:5000],
            "parsed_response": None,
            "reasoning": "LLM meta synthesis empty; built local synthesis.",
            "error": None,
            "context": {**(context or {}), "call_type": "meta_report_local"},
        },
        mirror_to=get_llm_audit_path(),
    )
    return local


# --------- Local report construction fallback (no LLM needed) -----------------

def build_detailed_report_locally(
    research_goal: str,
    ideas_ranked: List[Tuple[str, float, Dict[str, Any]]],
    meta_summary: str,
) -> str:
    out = []
    out.append(f"# Goal\n{research_goal}\n")
    out.append("# Top 10 Ideas\n")
    if not ideas_ranked:
        out.append("(no ideas generated)")
    for rank, (hid, elo, idea) in enumerate(ideas_ranked, start=1):
        out.append(f"## {rank}. {idea.get('title','(untitled)')}  \nID: {hid} · ELO {elo:.1f}")
        out.append(f"**Rationale:** {idea.get('rationale','')}")
        out.append(f"**Mechanism:** {idea.get('mechanism','')}")
        out.append(f"**Novelty:** {idea.get('novelty','')}")
        out.append(f"**Dependencies:** {idea.get('dependencies','')}")
        plan = idea.get("plan", [])
        if plan:
            out.append("**Plan:**")
            for step in plan:
                out.append(f"- {step}")
        else:
            out.append("**Plan:**\n- (none)")
        out.append(f"**Resources:** {idea.get('resources','')}")
        out.append("**Metrics:**\n" + _fmt_metrics(idea.get("metrics", [])))
        out.append(_fmt_list("Risks", idea.get("risks", [])))
        out.append(_fmt_list("Mitigations", idea.get("mitigations", [])))
        out.append(f"**Expected outcome:** {idea.get('expected_outcome','')}")
        out.append(
            f"**Confidence:** {idea.get('confidence', 0.6)} · "
            f"**Timeline (weeks):** {idea.get('timeline_weeks', 6)} · "
            f"**Cost (t-shirt):** {idea.get('cost_tshirt','M')}"
        )
        out.append("")  # spacer

    out.append("# Portfolio Synthesis\n")
    out.append(meta_summary or "(none)")

    return "\n".join(out)


async def draft_final_report(
    research_goal: str,
    ideas_ranked: List[Tuple[str, float, Dict[str, Any]]],
    meta_summary: str,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Try LLM-written report; if empty, build a rich local report.
    Both paths are logged: the failed LLM call is in llm_calls.jsonl.
    """
    system_msg = (
        "Write a technical brief preserving concrete plans and metrics."
        " Include: (a) goal, (b) top 10 ideas with detailed plan/outcomes/confidence,"
        " (c) portfolio synthesis. Avoid generic buzzwords."
    )

    ideas_block = []
    for rank, (hid, elo, idea) in enumerate(ideas_ranked, start=1):
        ideas_block.append({
            "rank": rank,
            "id": hid,
            "elo": round(elo, 1),
            **idea,
        })

    user_msg = (
        f"# Goal\n{research_goal}\n\n"
        "# Top 10 Ideas (preserve details; do NOT over-summarize)\n"
        f"{json.dumps(ideas_block, ensure_ascii=False)}\n\n"
        "# Portfolio Synthesis\n"
        f"{meta_summary}\n\n"
        "Now write the final report."
    )

    writing_model = get_model("writing")

    text = await _call_openai_responses(
        model=writing_model,
        system_instructions=system_msg,
        user_input=user_msg,
        allow_temperature=True,
        temperature_value=0.4,
        context={
            **(context or {}),
            "call_type": "final_report",
            "ideas_count": len(ideas_ranked),
        },
    )

    if text:
        return text

    # Local fallback (no LLM)
    local = build_detailed_report_locally(research_goal, ideas_ranked, meta_summary)
    record_llm_call(
        {
            "type": "final_report_fallback_local",
            "model": "offline-fallback",
            "temperature": None,
            "raw_response_text": local[:5000],
            "parsed_response": None,
            "reasoning": "LLM returned empty; constructed local report.",
            "error": None,
            "context": {
                **(context or {}),
                "call_type": "final_report_local",
                "ideas_count": len(ideas_ranked),
            },
        },
        mirror_to=get_llm_audit_path(),
    )
    return local
