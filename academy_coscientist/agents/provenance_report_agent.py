# academy_coscientist/agents/provenance_report_agent.py
"""
ProvenancePDFReportAgent — comprehensive PDF report for a co-scientist run.

Uses Claude to synthesise narrative sections from collected data.

Sections
--------
  1. Cover page
  2. Executive Summary  (Claude-generated)
  3. Hypothesis Overview table
  4. Hypothesis Details (per-hypothesis: description, rationale,
     all reviewer critiques, all refinement rounds with full content,
     PoC result, Claude narrative summary)
  5. Meta-Review
  6. Proof-of-Concept Results (verdict, interpretation, next steps,
     stdout/stderr, and full PoC code)
  7. Provenance Statistics (agent types, action counts, model usage)
  8. Provenance Performance (overhead timing)
"""

from __future__ import annotations

import asyncio
import csv
import html
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any

from academy.agent import Agent
from academy.agent import action

from academy_coscientist.utils.utils_logging import log_action
from academy_coscientist.utils.utils_logging import make_struct_logger

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
_NAVY    = '#1e3a5f'
_BLUE    = '#2980b9'
_TEAL    = '#1a6a5f'
_GREEN   = '#1a5f3f'
_LTBLUE  = '#ebf5fb'
_LTGRN   = '#eafaf1'
_LTYELL  = '#fef9e7'
_DKGREY  = '#555555'
_MIDGREY = '#888888'


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _esc(text: Any) -> str:
    return html.escape(str(text), quote=False)


def _nl(text: Any) -> str:
    return html.escape(str(text), quote=False).replace('\n', '<br/>')


# ---------------------------------------------------------------------------
# Math / formula helpers
# ---------------------------------------------------------------------------

_MATH_SYMBOLS: dict[str, str] = {
    r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
    r'\epsilon': 'ε', r'\varepsilon': 'ε', r'\zeta': 'ζ', r'\eta': 'η',
    r'\theta': 'θ', r'\vartheta': 'θ', r'\iota': 'ι', r'\kappa': 'κ',
    r'\lambda': 'λ', r'\mu': 'μ', r'\nu': 'ν', r'\xi': 'ξ',
    r'\pi': 'π', r'\varpi': 'π', r'\rho': 'ρ', r'\varrho': 'ρ',
    r'\sigma': 'σ', r'\varsigma': 'ς', r'\tau': 'τ', r'\upsilon': 'υ',
    r'\phi': 'φ', r'\varphi': 'φ', r'\chi': 'χ', r'\psi': 'ψ', r'\omega': 'ω',
    r'\Gamma': 'Γ', r'\Delta': 'Δ', r'\Theta': 'Θ', r'\Lambda': 'Λ',
    r'\Xi': 'Ξ', r'\Pi': 'Π', r'\Sigma': 'Σ', r'\Upsilon': 'Υ',
    r'\Phi': 'Φ', r'\Psi': 'Ψ', r'\Omega': 'Ω',
    r'\infty': '∞', r'\partial': '∂', r'\nabla': '∇', r'\forall': '∀',
    r'\exists': '∃', r'\emptyset': '∅', r'\varnothing': '∅',
    r'\times': '×', r'\cdot': '·', r'\div': '÷', r'\pm': '±', r'\mp': '∓',
    r'\leq': '≤', r'\geq': '≥', r'\neq': '≠', r'\approx': '≈',
    r'\equiv': '≡', r'\propto': '∝', r'\sim': '∼',
    r'\in': '∈', r'\notin': '∉', r'\subset': '⊂', r'\supset': '⊃',
    r'\cup': '∪', r'\cap': '∩',
    r'\sum': 'Σ', r'\prod': 'Π', r'\int': '∫', r'\oint': '∮',
    r'\sqrt': '√', r'\to': '→', r'\rightarrow': '→', r'\leftarrow': '←',
    r'\Rightarrow': '⇒', r'\Leftarrow': '⇐', r'\Leftrightarrow': '⇔',
    r'\ldots': '…', r'\cdots': '⋯', r'\vdots': '⋮', r'\ddots': '⋱',
    r'\hbar': 'ℏ', r'\ell': 'ℓ', r'\Re': 'ℜ', r'\Im': 'ℑ',
}


def _fmt_formula(text: Any) -> str:
    """Escape text for ReportLab XML, replace LaTeX symbols with Unicode,
    and wrap $...$ spans in Courier monospace for inline formula rendering."""
    import re
    s = str(text)
    for latex, uni in _MATH_SYMBOLS.items():
        s = s.replace(latex, uni)
    s = html.escape(s, quote=False)
    # Wrap $...$ inline math in Courier
    s = re.sub(r'\$([^$\n]+?)\$', r'<font name="Courier">\1</font>', s)
    # Wrap $$...$$ display math in bold Courier
    s = re.sub(r'\$\$([^$]+?)\$\$', r'<font name="Courier-Bold">\1</font>', s)
    return s



def _as_dict(v: Any) -> dict:
    return v if isinstance(v, dict) else {}


def _as_list(v: Any) -> list:
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v:
        return [v]
    return []


# ---------------------------------------------------------------------------
# ReportLab imports (lazy)
# ---------------------------------------------------------------------------

def _rl_imports():
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable, KeepTogether, PageBreak, Paragraph, Preformatted,
        SimpleDocTemplate, Spacer, Table, TableStyle,
    )
    return {
        'colors': colors,
        'TA_CENTER': TA_CENTER, 'TA_JUSTIFY': TA_JUSTIFY, 'TA_LEFT': TA_LEFT,
        'A4': A4,
        'getSampleStyleSheet': getSampleStyleSheet,
        'ParagraphStyle': ParagraphStyle,
        'cm': cm,
        'HRFlowable': HRFlowable,
        'KeepTogether': KeepTogether,
        'PageBreak': PageBreak,
        'Paragraph': Paragraph,
        'Preformatted': Preformatted,
        'SimpleDocTemplate': SimpleDocTemplate,
        'Spacer': Spacer,
        'Table': Table,
        'TableStyle': TableStyle,
    }


def _build_styles(rl):
    ss  = rl['getSampleStyleSheet']()
    PS  = rl['ParagraphStyle']
    C   = rl['TA_CENTER']
    J   = rl['TA_JUSTIFY']
    clr = rl['colors']

    def add(name, **kw):
        if name not in ss:
            ss.add(PS(name=name, **kw))

    add('CoverTitle',
        parent=ss['Title'], fontSize=26, spaceAfter=16,
        textColor=clr.HexColor(_NAVY), alignment=C)
    add('CoverSub',
        parent=ss['Normal'], fontSize=13, spaceAfter=8,
        textColor=clr.HexColor(_BLUE), alignment=C)
    add('CoverMeta',
        parent=ss['Normal'], fontSize=10, spaceAfter=4,
        textColor=clr.HexColor(_DKGREY), alignment=C)
    add('SecHeader',
        parent=ss['Heading1'], fontSize=15, spaceBefore=18, spaceAfter=8,
        textColor=clr.HexColor(_NAVY))
    add('SubHeader',
        parent=ss['Heading2'], fontSize=12, spaceBefore=10, spaceAfter=5,
        textColor=clr.HexColor(_BLUE))
    add('SubHeader2',
        parent=ss['Heading3'], fontSize=10, spaceBefore=6, spaceAfter=3,
        textColor=clr.HexColor(_TEAL), fontName='Helvetica-Bold')
    add('SubHeader3',
        parent=ss['Heading3'], fontSize=9, spaceBefore=4, spaceAfter=2,
        textColor=clr.HexColor(_GREEN), fontName='Helvetica-Bold')
    add('Body',
        parent=ss['Normal'], fontSize=10, spaceAfter=6, leading=14, alignment=J)
    add('BodyLeft',
        parent=ss['Normal'], fontSize=10, spaceAfter=5, leading=14)
    add('Bullet',
        parent=ss['Normal'], fontSize=9, spaceAfter=3, leading=13,
        leftIndent=16, firstLineIndent=-10)
    add('Small',
        parent=ss['Normal'], fontSize=8.5, spaceAfter=3,
        textColor=clr.HexColor(_DKGREY))
    add('SmallLeft',
        parent=ss['Normal'], fontSize=8.5, spaceAfter=3, leading=12)
    add('Label',
        parent=ss['Normal'], fontSize=9, spaceAfter=3,
        textColor=clr.HexColor(_NAVY), fontName='Helvetica-Bold')
    add('Code',
        parent=ss['Code'], fontSize=7.5, spaceAfter=3, leading=10,
        fontName='Courier', backColor=clr.HexColor('#f4f4f4'),
        leftIndent=6, rightIndent=6)
    add('CodeSmall',
        parent=ss['Code'], fontSize=6.5, spaceAfter=3, leading=9,
        fontName='Courier', backColor=clr.HexColor('#f4f4f4'),
        leftIndent=6, rightIndent=6)
    return ss


def _tbl_style(rl, hdr=_NAVY, alt=_LTBLUE):
    clr = rl['colors']
    return rl['TableStyle']([
        ('BACKGROUND',    (0, 0), (-1, 0),  clr.HexColor(hdr)),
        ('TEXTCOLOR',     (0, 0), (-1, 0),  clr.white),
        ('FONTNAME',      (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, 0),  9),
        ('TOPPADDING',    (0, 0), (-1, 0),  6),
        ('BOTTOMPADDING', (0, 0), (-1, 0),  6),
        ('ALIGN',         (0, 0), (-1, 0),  'CENTER'),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [clr.white, clr.HexColor(alt)]),
        ('FONTNAME',      (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',      (0, 1), (-1, -1), 8.5),
        ('TOPPADDING',    (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
        ('ALIGN',         (1, 1), (-1, -1), 'CENTER'),
        ('ALIGN',         (0, 1), (0, -1),  'LEFT'),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID',          (0, 0), (-1, -1), 0.4, clr.HexColor('#cccccc')),
        ('WORDWRAP',      (0, 0), (-1, -1), True),
    ])


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

def _read_jsonl(path: str | None) -> list[dict]:
    if not path or not os.path.exists(path):
        return []
    records: list[dict] = []
    try:
        with open(path, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError:
        pass
    return records


def _read_perf_csv(path: str | None) -> list[dict]:
    if not path or not os.path.exists(path):
        return []
    rows: list[dict] = []
    try:
        with open(path, newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                rows.append(dict(row))
    except OSError:
        pass
    return rows


def _read_poc_results(poc_dir: str | None) -> list[dict]:
    if not poc_dir or not os.path.isdir(poc_dir):
        return []
    results: list[dict] = []
    for name in os.listdir(poc_dir):
        sub = os.path.join(poc_dir, name)
        report_file = os.path.join(sub, 'report.json')
        if os.path.isfile(report_file):
            try:
                with open(report_file, encoding='utf-8') as fh:
                    data = json.load(fh)
                    # Also read the code file if present
                    code_file = os.path.join(sub, 'poc_code.py')
                    if os.path.isfile(code_file):
                        try:
                            with open(code_file, encoding='utf-8') as cf:
                                data['_code_text'] = cf.read()
                        except OSError:
                            pass
                    results.append(data)
            except (OSError, json.JSONDecodeError):
                pass
    results.sort(key=lambda r: (r.get('rank', 999), -float(r.get('hypothesis_score', 0.0))))
    return results


# ---------------------------------------------------------------------------
# Data extraction from actions.jsonl
# ---------------------------------------------------------------------------

def _extract_review_rounds_from_actions(actions: list[dict]) -> dict[str, list[dict]]:
    """hyp_id → sorted list of {round, team, avg_score, threshold, n_reviewers}"""
    rounds: dict[str, list[dict]] = defaultdict(list)
    for rec in actions:
        if rec.get('action') != 'review_round':
            continue
        inp = _as_dict(rec.get('input'))
        out = _as_dict(rec.get('output'))
        hyp_id = str(inp.get('hyp_id', ''))
        if not hyp_id:
            continue
        rounds[hyp_id].append({
            'round':       int(inp.get('round', 0)),
            'team':        str(inp.get('team', '')),
            'avg_score':   float(out.get('avg_score', 0.0)),
            'threshold':   float(out.get('threshold', 0.0)),
            'n_reviewers': int(out.get('reviewers', 0)),
        })
    for v in rounds.values():
        v.sort(key=lambda r: r['round'])
    return dict(rounds)


def _extract_action_stats_from_actions(actions: list[dict]) -> dict:
    """Aggregate action counts, agent types, etc. from actions.jsonl."""
    agent_types: set[str] = set()
    action_counts: dict[str, int] = defaultdict(int)
    for rec in actions:
        logger_name = rec.get('logger', '')
        act = rec.get('action', '')
        if logger_name:
            agent_types.add(logger_name)
        if act:
            action_counts[act] += 1
    return {
        'agent_types':   sorted(agent_types),
        'action_counts': dict(action_counts),
    }


# ---------------------------------------------------------------------------
# Data extraction from llm_calls.jsonl
# ---------------------------------------------------------------------------

def _parse_llm_text(rec: dict) -> dict:
    """Try to extract parsed JSON from a llm_calls.jsonl record."""
    # New format: has parsed_response
    parsed = rec.get('parsed_response')
    if isinstance(parsed, dict) and parsed:
        return parsed
    # Old format: has 'text' as raw JSON string
    text = rec.get('text') or rec.get('raw_response_text') or ''
    if not text:
        return {}
    s = text.strip()
    if s.startswith('```'):
        lines = s.splitlines()
        s = '\n'.join(lines[1:-1]).strip()
    try:
        result = json.loads(s)
        return result if isinstance(result, dict) else {}
    except Exception:
        # Try to find JSON object boundaries
        start = s.find('{')
        end = s.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                pass
    return {}


def _extract_full_reviews(llm_calls: list[dict]) -> dict[str, list[dict]]:
    """
    Parse review_hypothesis calls from llm_calls.jsonl.

    Returns: hyp_id → list of reviewer critique dicts, each containing:
      reviewer, team, ts, novelty, rigor, feasibility, impact, clarity,
      score, confidence, reasoning, strengths, weaknesses, risks, recommendation, notes
    """
    reviews: dict[str, list[dict]] = defaultdict(list)
    for rec in llm_calls:
        ctx = rec.get('context') or {}
        if not isinstance(ctx, dict):
            continue
        if ctx.get('call_type') != 'review_hypothesis':
            continue
        hyp_id = str(ctx.get('hyp_id', ''))
        if not hyp_id:
            continue
        parsed = _parse_llm_text(rec)
        reviews[hyp_id].append({
            'reviewer':       str(ctx.get('reviewer', ctx.get('agent', '?'))),
            'team':           str(ctx.get('team', '')),
            'ts':             str(rec.get('ts', '')),
            'model':          str(rec.get('model_used') or rec.get('model', '')),
            'novelty':        float(parsed.get('novelty', 5)),
            'rigor':          float(parsed.get('rigor', 5)),
            'feasibility':    float(parsed.get('feasibility', 5)),
            'impact':         float(parsed.get('impact', 5)),
            'clarity':        float(parsed.get('clarity', 5)),
            'score':          float(parsed.get('score', 0)),
            'confidence':     float(parsed.get('confidence', 0)),
            'reasoning':      str(parsed.get('reasoning', '')),
            'strengths':      _as_list(parsed.get('strengths', [])),
            'weaknesses':     _as_list(parsed.get('weaknesses', [])),
            'risks':          _as_list(parsed.get('risks', parsed.get('major_risks', []))),
            'recommendation': str(parsed.get('recommendation', '')),
            'notes':          str(parsed.get('summary', parsed.get('notes', ''))),
            'usage':          _as_dict(rec.get('usage')),
        })
    return dict(reviews)


def _extract_full_refinements(llm_calls: list[dict]) -> dict[str, list[dict]]:
    """
    Parse refine_hypothesis calls from llm_calls.jsonl.

    Returns: hyp_id → list of refinement dicts, each containing:
      round, team, ts, reasoning, title, description, rationale, revision_notes
    """
    # We need to correlate with action records; we group by hyp_id and order by ts
    refinements: dict[str, list[dict]] = defaultdict(list)
    for rec in llm_calls:
        ctx = rec.get('context') or {}
        if not isinstance(ctx, dict):
            continue
        if ctx.get('call_type') != 'refine_hypothesis':
            continue
        hyp_id = str(ctx.get('hyp_id', ''))
        if not hyp_id:
            continue
        parsed = _parse_llm_text(rec)
        refinements[hyp_id].append({
            'team':           str(ctx.get('team', '')),
            'ts':             str(rec.get('ts', '')),
            'reasoning':      str(parsed.get('reasoning', '')),
            'title':          str(parsed.get('title', '')),
            'description':    str(parsed.get('description', '')),
            'rationale':      str(parsed.get('rationale', '')),
            'revision_notes': str(parsed.get('revision_notes', '')),
        })
    # Sort each list by timestamp; assign round numbers
    for hyp_id, refs in refinements.items():
        refs.sort(key=lambda r: r['ts'])
        for i, ref in enumerate(refs, 1):
            ref['round'] = i
    return dict(refinements)


def _extract_llm_stats(llm_calls: list[dict]) -> dict:
    """Aggregate model usage and token counts from llm_calls.jsonl."""
    models_used: dict[str, int] = defaultdict(int)
    total_tokens = 0
    call_types: dict[str, int] = defaultdict(int)
    for rec in llm_calls:
        model = str(rec.get('model_used') or rec.get('model', 'unknown'))
        models_used[model] += 1
        usage = _as_dict(rec.get('usage'))
        try:
            total_tokens += int(usage.get('total_tokens', 0))
        except (TypeError, ValueError):
            pass
        ctx = rec.get('context') or {}
        if isinstance(ctx, dict) and ctx.get('call_type'):
            call_types[ctx['call_type']] += 1
    return {
        'models_used': dict(models_used),
        'total_tokens': total_tokens,
        'total_llm_calls': len(llm_calls),
        'call_types': dict(call_types),
    }


# ---------------------------------------------------------------------------
# FlowCept buffer reader (optional, for performance stats only)
# ---------------------------------------------------------------------------

def _read_provenance_buffer(path: str | None) -> list[dict]:
    return _read_jsonl(path)


def _summarise_perf_csv(rows: list[dict]) -> dict[str, dict]:
    if not rows:
        return {}
    if 'n' in rows[0] and 'total_ms' in rows[0]:
        out: dict[str, dict] = {}
        for row in rows:
            cat = row.get('category', '')
            if cat:
                out[cat] = {
                    'n':        int(float(row.get('n', 0))),
                    'total_ms': float(row.get('total_ms', 0)),
                    'mean_us':  float(row.get('mean_us', 0)),
                    'min_us':   float(row.get('min_us', 0)),
                    'max_us':   float(row.get('max_us', 0)),
                }
        return out
    buckets: dict[str, list[float]] = {}
    for row in rows:
        cat = row.get('category', '')
        try:
            buckets.setdefault(cat, []).append(float(row.get('elapsed_us', 0)))
        except (TypeError, ValueError):
            pass
    agg: dict[str, dict] = {}
    for cat, vals in sorted(buckets.items()):
        n = len(vals)
        agg[cat] = {
            'n':        n,
            'total_ms': round(sum(vals) / 1e3, 4),
            'mean_us':  round(sum(vals) / n, 1) if n else 0.0,
            'min_us':   round(min(vals), 1),
            'max_us':   round(max(vals), 1),
        }
    return agg


# ---------------------------------------------------------------------------
# Claude narrative helpers (async)
# ---------------------------------------------------------------------------

async def _call_claude(system: str, user: str, max_tokens: int = 1500) -> str:
    """Safe wrapper for Claude narrative calls. Returns '' on failure."""
    try:
        from academy_coscientist.utils.utils_llm import call_claude_llm
        from academy_coscientist.utils.config import get_model, get_temperature
        return await call_claude_llm(
            system=system, user=user,
            model=get_model('report', default='claude-sonnet-4-6'),
            max_tokens=max_tokens,
            temperature=get_temperature('report', 0.3) or 0.3,
            ctx={'call_type': 'report_narrative'},
        )
    except Exception as e:
        _log.warning('Claude narrative call failed: %s', e)
        return ''


async def _llm_exec_summary(
    topic: str,
    leaderboard: list[tuple[str, float, dict]],
    poc_results: list[dict],
    llm_stats: dict,
    action_stats: dict,
) -> str:
    """Ask Claude to write the executive summary."""
    n_hyps   = len(leaderboard)
    top3     = leaderboard[:3]
    top3_txt = '\n'.join(
        f'  {i+1}. "{row[2].get("title", row[0])}" — score {row[1]:.3f}'
        for i, row in enumerate(top3)
    )
    supported = sum(1 for r in poc_results if 'SUPPORTED' in str(r.get('verdict', '')).upper())
    refuted   = sum(1 for r in poc_results if str(r.get('verdict', '')).upper() == 'REFUTED')

    user = (
        f'Topic: {topic}\n\n'
        f'Number of hypotheses evaluated: {n_hyps}\n'
        f'Top hypotheses:\n{top3_txt}\n\n'
        f'PoC experiments: {len(poc_results)} total, {supported} supported, {refuted} refuted\n'
        f'Total LLM calls: {llm_stats.get("total_llm_calls", 0)}, '
        f'tokens: {llm_stats.get("total_tokens", 0):,}\n'
        f'Agent types active: {", ".join(action_stats.get("agent_types", []))}\n\n'
        'Write a concise executive summary (3-4 paragraphs) covering: what the run '
        'investigated, the key hypotheses found, PoC outcomes, and the overall research '
        'value. Be specific and data-driven. Use plain text (no markdown).'
    )
    system = (
        'You are a scientific research report writer for an autonomous AI co-scientist system. '
        'Write clear, precise, informative executive summaries from structured data.'
    )
    result = await _call_claude(system, user, max_tokens=800)
    return result


async def _llm_hyp_narrative(
    hyp_id: str,
    hyp_payload: dict,
    reviews: list[dict],
    refinements: list[dict],
    review_rounds: list[dict],
    poc: dict | None,
) -> str:
    """Ask Claude to write a narrative summary for one hypothesis."""
    title = hyp_payload.get('title', hyp_id)
    desc  = hyp_payload.get('description', '')

    # Build review summary — structured numerical data only.
    # Reviewer reasoning text is provenance and must be rendered verbatim in the
    # PDF, NOT passed through an LLM (which could paraphrase or alter it).
    if reviews:
        dims = ['novelty', 'rigor', 'feasibility', 'impact', 'clarity']
        avg_dims = {d: sum(r.get(d, 5) for r in reviews) / len(reviews) for d in dims}
        avg_score = sum(r.get('score', 0) for r in reviews) / len(reviews)
        recommendations = [r.get('recommendation', '') for r in reviews if r.get('recommendation')]
        majority_rec = max(set(recommendations), key=recommendations.count) if recommendations else 'n/a'
        reviews_txt = (
            f'{len(reviews)} reviewer(s), avg score {avg_score:.3f}, '
            f'majority recommendation: {majority_rec}\n'
            + ', '.join(f'{d}={avg_dims[d]:.1f}' for d in dims) + '\n'
        )
    else:
        reviews_txt = 'No detailed reviews available.'

    # Refinement summary — counts and score trajectory only.
    # revision_notes are provenance and rendered verbatim elsewhere.
    refinements_txt = ''
    if refinements:
        scores = [r.get('score_before') for r in refinements if r.get('score_before') is not None]
        trajectory = ' → '.join(f'{s:.3f}' for s in scores) if scores else 'n/a'
        refinements_txt = (
            f'{len(refinements)} refinement round(s), score trajectory: {trajectory}\n'
        )

    # PoC summary — verdict and confidence only.
    # PoC interpretation text is provenance and rendered verbatim elsewhere.
    poc_txt = ''
    if poc:
        poc_txt = (
            f'PoC verdict: {poc.get("verdict")} '
            f'(confidence {poc.get("confidence", 0):.2f})\n'
        )

    user = (
        f'Hypothesis: "{title}"\n\n'
        f'Description: {desc[:400]}\n\n'
        f'Review summary (scores only):\n{reviews_txt}\n'
        f'Refinement history:\n{refinements_txt if refinements_txt else "(no refinements)"}\n'
        f'PoC result:\n{poc_txt if poc_txt else "(no PoC run)"}\n\n'
        'Write a 2-paragraph scientific narrative analysis of this hypothesis: '
        '(1) evaluate its scientific merit based on the dimension scores and verdict; '
        '(2) assess whether the quantitative evidence supports pursuing it further. '
        'Base your analysis solely on the scores and verdicts above — do not invent '
        'reasoning not present in the data. Plain text only.'
    )
    system = (
        'You are a scientific reviewer writing a narrative analysis of a hypothesis '
        'for an academic research report. Base your analysis on the provided scores '
        'and verdicts only. Be concise, critical, and evidence-based.'
    )
    result = await _call_claude(system, user, max_tokens=500)
    return result


# ---------------------------------------------------------------------------
# PDF renderer
# ---------------------------------------------------------------------------

def _render_pdf(
    output_path: str,
    topic: str,
    leaderboard: list[tuple[str, float, dict]],
    exec_summary: str,
    poc_results: list[dict],
    action_stats: dict,
    llm_stats: dict,
    perf_agg: dict[str, dict],
    review_rounds: dict[str, list[dict]],
    full_reviews: dict[str, list[dict]],
    full_refinements: dict[str, list[dict]],
    hyp_narratives: dict[str, str],
    top_n: int = 5,
) -> None:
    rl          = _rl_imports()
    ss          = _build_styles(rl)
    A4          = rl['A4']
    cm          = rl['cm']
    Paragraph   = rl['Paragraph']
    Preformatted = rl['Preformatted']
    Spacer      = rl['Spacer']
    PageBreak   = rl['PageBreak']
    Table       = rl['Table']
    TableStyle  = rl['TableStyle']
    HR          = rl['HRFlowable']
    KT          = rl['KeepTogether']
    colors      = rl['colors']

    W, H   = A4
    margin = 2.2 * cm
    cw     = W - 2 * margin

    story: list = []

    def P(text: Any, style: str = 'Body') -> Any:
        return Paragraph(str(text), ss[style])

    def PE(text: Any, style: str = 'Body') -> Any:
        return Paragraph(_esc(text), ss[style])

    def PNL(text: Any, style: str = 'Body') -> Any:
        return Paragraph(_nl(text), ss[style])

    def PF(text: Any, style: str = 'Body') -> Any:
        """Paragraph with LaTeX→Unicode substitution, HTML escaping, and $…$ in Courier."""
        return Paragraph(_fmt_formula(text).replace('\n', '<br/>'), ss[style])

    def _code_box(text: Any, small: bool = False) -> Any:
        """Preformatted code block in a bordered, shaded box."""
        fs = 6.5 if small else 7.5
        pre_style = rl['ParagraphStyle'](
            'PreCode' if not small else 'PreCodeSmall',
            fontName='Courier', fontSize=fs, leading=fs * 1.35,
            leftIndent=0, rightIndent=0,
        )
        pre = Preformatted(str(text), pre_style)
        box = Table([[pre]], colWidths=[cw - 0.3 * cm])
        box.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), colors.HexColor('#f4f4f4')),
            ('BOX',           (0, 0), (-1, -1), 0.5, colors.HexColor('#bbbbbb')),
            ('TOPPADDING',    (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING',   (0, 0), (-1, -1), 8),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 8),
        ]))
        return box

    def hr(thin: bool = False):
        return HR(
            width='100%', thickness=0.5 if thin else 1,
            color=colors.HexColor(_BLUE if not thin else '#cccccc'),
            spaceAfter=5, spaceBefore=5,
        )

    def section(title: str):
        story.append(PageBreak())
        story.append(P(_esc(title), 'SecHeader'))
        story.append(hr())

    def bullet(text: Any) -> Any:
        return P(f'• {_esc(str(text))}', 'Bullet')

    short_topic = topic if len(topic) <= 90 else topic[:87] + '…'

    # ========================= 1. COVER PAGE ========================
    story.append(Spacer(1, 3 * cm))
    story.append(P('Co-Scientist Research Report', 'CoverTitle'))
    story.append(Spacer(1, 0.4 * cm))
    story.append(PE(short_topic, 'CoverSub'))
    story.append(Spacer(1, 1.5 * cm))
    story.append(hr())
    story.append(Spacer(1, 0.6 * cm))
    story.append(P(datetime.now().strftime('Generated: %Y-%m-%d %H:%M'), 'CoverMeta'))
    story.append(P(f'Total hypotheses evaluated: {len(leaderboard)}', 'CoverMeta'))
    story.append(P(f'PoC experiments run: {len(poc_results)}', 'CoverMeta'))
    story.append(P(
        f"LLM calls: {llm_stats.get('total_llm_calls', 0)} · "
        f"tokens: {llm_stats.get('total_tokens', 0):,}",
        'CoverMeta',
    ))
    story.append(Spacer(1, 0.6 * cm))
    story.append(hr())
    story.append(Spacer(1, 1.0 * cm))
    story.append(P(
        'Generated by the Academy Co-Scientist autonomous multi-agent pipeline.',
        'CoverMeta',
    ))

    # ========================= 2. EXECUTIVE SUMMARY =================
    story.append(PageBreak())
    story.append(P('Executive Summary', 'SecHeader'))
    story.append(hr())
    if exec_summary:
        for para in exec_summary.strip().split('\n\n'):
            para = para.strip()
            if para:
                story.append(PE(para, 'Body'))
                story.append(Spacer(1, 0.25 * cm))
    else:
        # Fallback data-driven summary
        n_hyps = len(leaderboard)
        scores = [s for _, s, _ in leaderboard]
        supported = sum(1 for r in poc_results if 'SUPPORTED' in str(r.get('verdict', '')).upper())
        refuted   = sum(1 for r in poc_results if str(r.get('verdict', '')).upper() == 'REFUTED')
        story.append(PE(
            f'This report presents results of an autonomous co-scientist run on: {topic}. '
            f'{n_hyps} hypotheses evaluated (scores {min(scores):.3f}–{max(scores):.3f}). '
            f'{len(poc_results)} PoC experiments: {supported} supported, {refuted} refuted.',
            'Body',
        ))

    # ========================= 3. HYPOTHESIS OVERVIEW ===============
    section('Hypothesis Overview')
    story.append(PE(
        f'All {len(leaderboard)} hypotheses ranked by tournament score (descending).',
        'BodyLeft',
    ))
    story.append(Spacer(1, 0.3 * cm))

    if leaderboard:
        col_w = [0.7*cm, cw*0.45, cw*0.10, cw*0.10, cw*0.10, cw*0.19]
        tdata = [['#', 'Title', 'Score', 'Conf.', 'Rev. Rounds', 'Team']]
        for rank, (hid, score, payload) in enumerate(leaderboard, 1):
            meta  = payload.get('meta') or {}
            title = _esc(str(payload.get('title') or hid)[:55])
            conf  = float(payload.get('confidence') or 0.0)
            team  = _esc(str(
                meta.get('validated_by_team') or
                payload.get('validated_by_team', '')
            )[:18])
            n_rounds = len(review_rounds.get(hid, []))
            tdata.append([
                str(rank),
                Paragraph(title, ss['Small']),
                f'{score:.3f}',
                f'{conf:.2f}' if conf else '—',
                str(n_rounds) if n_rounds else '—',
                Paragraph(team, ss['Small']),
            ])
        tbl = Table(tdata, colWidths=col_w, repeatRows=1)
        tbl.setStyle(_tbl_style(rl))
        story.append(tbl)
    else:
        story.append(P('No hypothesis data available.', 'Body'))

    # ========================= 4. HYPOTHESIS DETAILS ================
    top_rows = leaderboard[:top_n]
    section(f'Hypothesis Details (Top {len(top_rows)})')

    # Build poc lookup by hyp_id
    poc_by_hyp: dict[str, dict] = {}
    for poc in poc_results:
        hid_poc = str(poc.get('hypothesis_id', ''))
        if hid_poc:
            poc_by_hyp[hid_poc] = poc

    for i, (hid, score, payload) in enumerate(top_rows, 1):
        meta  = payload.get('meta') or {}
        title = str(payload.get('title') or hid)
        desc  = str(payload.get('description') or '')
        conf  = float(payload.get('confidence') or 0.0)
        team  = str(
            meta.get('validated_by_team') or
            payload.get('validated_by_team', 'unknown')
        )
        rationale = str(meta.get('rationale') or payload.get('rationale') or '')

        block: list = []

        # --- Hypothesis heading ---
        block.append(P(
            f'<b>#{i}  {_esc(title)}</b>', 'SubHeader',
        ))
        block.append(P(
            f'Score: <b>{score:.3f}</b>  |  '
            f'Confidence: <b>{conf:.2f}</b>  |  '
            f'Validated by: <b>{_esc(team)}</b>  |  '
            f'ID: <font color="{_MIDGREY}">{_esc(hid[:16])}</font>',
            'Small',
        ))
        story.append(Spacer(1, 0.1 * cm))

        # --- Description ---
        if desc:
            block.append(P('<b>Description</b>', 'SubHeader2'))
            block.append(PF(desc, 'Body'))

        # --- Rationale ---
        if rationale:
            block.append(P('<b>Rationale</b>', 'SubHeader2'))
            block.append(PF(rationale, 'Body'))

        # --- Review Rounds Summary ---
        rounds = review_rounds.get(hid, [])
        if rounds:
            block.append(P('<b>Review Rounds Summary</b>', 'SubHeader2'))
            r_col_w = [cw*0.12, cw*0.28, cw*0.20, cw*0.20, cw*0.20]
            r_data  = [['Round', 'Team', 'Avg Score', 'Threshold', 'Reviewers']]
            for rd in rounds:
                status = '✓ pass' if rd['avg_score'] >= rd['threshold'] else '✗ fail'
                r_data.append([
                    str(rd['round']),
                    _esc(rd['team']),
                    f"{rd['avg_score']:.3f}  {status}",
                    f"{rd['threshold']:.2f}",
                    str(rd['n_reviewers']),
                ])
            rtbl = Table(r_data, colWidths=r_col_w, repeatRows=1)
            rtbl.setStyle(_tbl_style(rl, hdr=_TEAL, alt=_LTGRN))
            block.append(rtbl)
            block.append(Spacer(1, 0.2 * cm))

        # --- Full Reviewer Critiques ---
        hyp_reviews = full_reviews.get(hid, [])
        if hyp_reviews:
            block.append(P(
                f'<b>Reviewer Critiques</b>  '
                f'<font color="{_MIDGREY}">({len(hyp_reviews)} review(s))</font>',
                'SubHeader2',
            ))
            for rv_i, rv in enumerate(hyp_reviews, 1):
                block.append(P(
                    f'<b>Reviewer {rv_i}: {_esc(rv["reviewer"])}</b>  '
                    f'(team: {_esc(rv["team"])}, model: {_esc(rv["model"])})',
                    'SubHeader3',
                ))
                # Dimensions table
                dim_cols = [cw*0.25, cw*0.15, cw*0.15, cw*0.15, cw*0.15, cw*0.15]
                dim_data = [
                    ['Dimension', 'Novelty', 'Rigor', 'Feasibility', 'Impact', 'Clarity'],
                    [
                        'Score (1–10)',
                        f"{rv['novelty']:.1f}",
                        f"{rv['rigor']:.1f}",
                        f"{rv['feasibility']:.1f}",
                        f"{rv['impact']:.1f}",
                        f"{rv['clarity']:.1f}",
                    ],
                ]
                dim_tbl = Table(dim_data, colWidths=dim_cols)
                dim_tbl.setStyle(_tbl_style(rl, hdr=_BLUE, alt=_LTBLUE))
                block.append(dim_tbl)
                block.append(P(
                    f'Composite score: <b>{rv["score"]:.3f}</b>  |  '
                    f'Confidence: <b>{rv["confidence"]:.2f}</b>  |  '
                    f'Recommendation: <b>{_esc(rv["recommendation"])}</b>',
                    'Small',
                ))
                # Reasoning
                if rv['reasoning']:
                    block.append(P('<i>Reasoning:</i>', 'SmallLeft'))
                    block.append(PF(rv["reasoning"], 'Body'))
                # Notes / summary
                if rv['notes']:
                    block.append(P('<i>Summary:</i>', 'SmallLeft'))
                    block.append(PF(rv["notes"], 'Body'))
                # Strengths
                if rv['strengths']:
                    block.append(P('<i>Strengths:</i>', 'SmallLeft'))
                    for s in rv['strengths'][:5]:
                        block.append(bullet(s))
                # Weaknesses
                if rv['weaknesses']:
                    block.append(P('<i>Weaknesses:</i>', 'SmallLeft'))
                    for w in rv['weaknesses'][:5]:
                        block.append(bullet(w))
                # Risks
                if rv['risks']:
                    block.append(P('<i>Risks:</i>', 'SmallLeft'))
                    for r in rv['risks'][:4]:
                        block.append(bullet(r))
                block.append(Spacer(1, 0.1 * cm))
                if rv_i < len(hyp_reviews):
                    block.append(hr(thin=True))

        # --- Refinement History ---
        hyp_refinements = full_refinements.get(hid, [])
        if hyp_refinements:
            block.append(P(
                f'<b>Refinement History</b>  '
                f'<font color="{_MIDGREY}">({len(hyp_refinements)} refinement(s))</font>',
                'SubHeader2',
            ))
            for ref in hyp_refinements:
                rnd  = ref.get('round', '?')
                team = ref.get('team', '')
                block.append(P(
                    f'<b>Refinement Round {_esc(str(rnd))}</b>'
                    + (f'  (team: {_esc(team)})' if team else ''),
                    'SubHeader3',
                ))
                if ref.get('title'):
                    block.append(P(
                        f'<b>Refined title:</b> {_esc(ref["title"])}', 'BodyLeft',
                    ))
                if ref.get('description'):
                    block.append(P('<i>Refined description:</i>', 'SmallLeft'))
                    block.append(PF(ref["description"], 'Body'))
                if ref.get('rationale'):
                    block.append(P('<i>Refined rationale:</i>', 'SmallLeft'))
                    block.append(PF(ref["rationale"], 'Body'))
                if ref.get('revision_notes'):
                    block.append(P('<i>What changed (revision notes):</i>', 'SmallLeft'))
                    block.append(PF(ref["revision_notes"], 'Body'))
                if ref.get('reasoning'):
                    block.append(P('<i>Refiner reasoning:</i>', 'SmallLeft'))
                    block.append(PF(ref["reasoning"], 'Body'))
                block.append(Spacer(1, 0.15 * cm))

        # --- PoC result linked to this hypothesis ---
        poc_item = poc_by_hyp.get(hid)
        if poc_item:
            verdict   = str(poc_item.get('verdict', '?')).upper()
            v_color = (
                '#1a7a1a' if verdict == 'SUPPORTED'
                else '#b35900' if 'PARTIAL' in verdict
                else '#8b0000'
            )
            block.append(P(
                f'<b>Proof-of-Concept:</b>  '
                f'<font color="{v_color}"><b>{_esc(verdict)}</b></font>  '
                f'(confidence {float(poc_item.get("confidence", 0)):.2f})',
                'SubHeader2',
            ))
            interp = poc_item.get('interpretation', '')
            if interp:
                block.append(PNL(interp, 'Body'))

        # --- Claude narrative ---
        narrative = hyp_narratives.get(hid, '')
        if narrative:
            block.append(P('<b>Analysis</b>', 'SubHeader2'))
            for para in narrative.strip().split('\n\n'):
                para = para.strip()
                if para:
                    block.append(PE(para, 'Body'))

        block.append(Spacer(1, 0.5 * cm))
        story.append(KT(block))
        if i < len(top_rows):
            story.append(hr())

    # ========================= 5. POC RESULTS =======================
    section('Proof-of-Concept Experiment Results')
    if poc_results:
        story.append(P(
            f'{len(poc_results)} proof-of-concept experiment(s) executed.',
            'BodyLeft',
        ))
        story.append(Spacer(1, 0.3 * cm))

        for poc in poc_results:
            hyp_title  = str(poc.get('hypothesis_title', '?'))
            verdict    = str(poc.get('verdict', '?')).upper()
            conf       = float(poc.get('confidence') or 0.0)
            interp     = str(poc.get('interpretation', ''))
            next_steps = str(poc.get('next_steps', ''))
            score_poc  = float(poc.get('hypothesis_score') or 0.0)
            stdout     = str(poc.get('stdout', ''))
            stderr     = str(poc.get('stderr', ''))
            returncode = poc.get('returncode', '?')
            team_poc   = str(poc.get('validated_by_team') or '')
            rank_poc   = poc.get('rank', '?')
            code_text  = str(poc.get('_code_text') or poc.get('code', ''))
            hyp_desc   = str(poc.get('hypothesis_description', ''))
            hyp_id_poc = str(poc.get('hypothesis_id', ''))

            v_color = (
                '#1a7a1a' if verdict == 'SUPPORTED'
                else '#b35900' if 'PARTIAL' in verdict
                else '#8b0000'
            )

            block: list = []
            block.append(P(
                f'<font color="{v_color}"><b>Rank #{rank_poc} — {_esc(verdict)}</b></font>',
                'SubHeader',
            ))
            block.append(P(
                f'<b>{_esc(hyp_title[:80])}</b>',
                'BodyLeft',
            ))
            block.append(P(
                f'Hypothesis score: <b>{score_poc:.3f}</b>  |  '
                f'Confidence: <b>{conf:.2f}</b>  |  '
                f'Return code: <b>{_esc(str(returncode))}</b>  |  '
                f'Team: <b>{_esc(team_poc)}</b>  |  '
                f'ID: <font color="{_MIDGREY}">{_esc(hyp_id_poc[:16])}</font>',
                'Small',
            ))

            if hyp_desc:
                block.append(P('<b>Hypothesis Description</b>', 'SubHeader2'))
                block.append(PNL(hyp_desc, 'Body'))

            if interp:
                block.append(P('<b>Interpretation</b>', 'SubHeader2'))
                block.append(PNL(interp, 'Body'))

            if next_steps:
                block.append(P('<b>Recommended Next Steps</b>', 'SubHeader2'))
                block.append(PNL(next_steps, 'BodyLeft'))

            if stdout:
                block.append(P('<b>Standard Output</b>', 'SubHeader2'))
                block.append(_code_box(stdout))

            if stderr:
                block.append(P('<b>Standard Error</b>', 'SubHeader2'))
                block.append(_code_box(stderr))

            if code_text:
                block.append(P('<b>PoC Code</b>', 'SubHeader2'))
                block.append(_code_box(code_text, small=True))

            block.append(Spacer(1, 0.4 * cm))
            story.append(KT(block))
            story.append(hr())
    else:
        story.append(P(
            'No PoC results found (poc_results directory absent or empty).',
            'Body',
        ))

    # ========================= 7. PROVENANCE STATISTICS =============
    section('Provenance & LLM Statistics')

    # LLM stats
    story.append(P('LLM Usage', 'SubHeader'))
    for label, val in [
        ('Total LLM calls', str(llm_stats.get('total_llm_calls', 0))),
        ('Total tokens consumed', f"{llm_stats.get('total_tokens', 0):,}"),
    ]:
        story.append(P(f'<b>{_esc(label)}:</b>  {_esc(val)}', 'BodyLeft'))

    call_types = llm_stats.get('call_types', {})
    if call_types:
        col_w2 = [cw * 0.55, cw * 0.45]
        tdata  = [['Call Type', 'Count']]
        for ct, cnt in sorted(call_types.items(), key=lambda x: -x[1]):
            tdata.append([_esc(ct), str(cnt)])
        tbl = Table(tdata, colWidths=col_w2)
        tbl.setStyle(_tbl_style(rl, hdr=_TEAL, alt=_LTGRN))
        story.append(tbl)
        story.append(Spacer(1, 0.3 * cm))

    models = llm_stats.get('models_used', {})
    if models:
        story.append(P('Model Usage', 'SubHeader'))
        col_w3 = [cw * 0.6, cw * 0.4]
        tdata  = [['Model', 'Call Count']]
        for model, cnt in sorted(models.items(), key=lambda x: -x[1]):
            tdata.append([_esc(model), str(cnt)])
        tbl = Table(tdata, colWidths=col_w3)
        tbl.setStyle(_tbl_style(rl, hdr=_BLUE))
        story.append(tbl)
        story.append(Spacer(1, 0.4 * cm))

    # Action stats
    story.append(P('Agent Actions', 'SubHeader'))
    story.append(P(
        f'<b>Agent types observed:</b>  '
        + _esc(', '.join(action_stats.get('agent_types', [])) or 'n/a'),
        'BodyLeft',
    ))
    action_counts = action_stats.get('action_counts', {})
    if action_counts:
        top_acts = sorted(action_counts.items(), key=lambda x: -x[1])[:25]
        col_w4 = [cw * 0.65, cw * 0.35]
        tdata  = [['Action', 'Count']]
        for act, cnt in top_acts:
            tdata.append([_esc(act), str(cnt)])
        tbl = Table(tdata, colWidths=col_w4)
        tbl.setStyle(_tbl_style(rl, hdr=_BLUE))
        story.append(tbl)

    # ========================= 8. PROVENANCE PERFORMANCE ============
    if perf_agg:
        section('Provenance Capture Overhead')
        story.append(P(
            'Wall-clock overhead of each provenance-capture category.',
            'BodyLeft',
        ))
        story.append(Spacer(1, 0.3 * cm))
        col_w5 = [cw*0.28, cw*0.12, cw*0.18, cw*0.14, cw*0.14, cw*0.14]
        tdata  = [['Category', 'N', 'Total (ms)', 'Mean (µs)', 'Min (µs)', 'Max (µs)']]
        for cat, v in sorted(perf_agg.items()):
            tdata.append([
                _esc(cat), str(v['n']),
                f"{v['total_ms']:.3f}",
                f"{v['mean_us']:.1f}",
                f"{v['min_us']:.1f}",
                f"{v['max_us']:.1f}",
            ])
        tbl = Table(tdata, colWidths=col_w5, repeatRows=1)
        tbl.setStyle(_tbl_style(rl))
        story.append(tbl)
        story.append(Spacer(1, 0.4 * cm))
        total_ms = sum(v['total_ms'] for v in perf_agg.values())
        story.append(P(
            f'<b>Total provenance overhead:</b>  {total_ms:.2f} ms.',
            'BodyLeft',
        ))

    # ========================= BUILD DOC ============================
    def _on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 7)
        canvas.setFillColor(colors.HexColor(_DKGREY))
        canvas.drawString(margin, 0.8 * cm,
                          f'Co-Scientist Report — {_esc(short_topic[:60])}')
        canvas.drawRightString(W - margin, 0.8 * cm, f'Page {doc.page}')
        canvas.setStrokeColor(colors.HexColor(_BLUE))
        canvas.setLineWidth(0.5)
        canvas.line(margin, H - 1.2 * cm, W - margin, H - 1.2 * cm)
        canvas.restoreState()

    doc = rl['SimpleDocTemplate'](
        output_path,
        pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=1.8 * cm, bottomMargin=1.6 * cm,
        title='Co-Scientist Research Report',
        author='Academy Co-Scientist',
        subject=topic,
    )
    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ProvenancePDFReportAgent(Agent):
    """
    Academy Agent that generates a comprehensive PDF report at simulation end.
    Uses Claude to synthesise narrative sections from collected provenance data.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logger = make_struct_logger('ProvenancePDFReportAgent')
        self._tournament               = None
        self._poc_dir: str | None      = None
        self._provenance_buffer_path: str | None = None
        self._provenance_perf_csv: str | None    = None
        self._run_dir: str | None      = None
        self._top_n: int = 5

    # ------------------------------------------------------------------
    # Wiring actions
    # ------------------------------------------------------------------

    @action
    async def set_tournament(self, tournament) -> None:
        self._tournament = tournament
        log_action(self.logger, 'set_tournament',
                   {'tournament': str(tournament)}, {'ok': True})

    @action
    async def set_poc_dir(self, poc_dir: str) -> None:
        self._poc_dir = poc_dir
        log_action(self.logger, 'set_poc_dir', {'poc_dir': poc_dir}, {'ok': True})

    @action
    async def set_provenance_buffer_path(self, path: str) -> None:
        self._provenance_buffer_path = path
        log_action(self.logger, 'set_provenance_buffer_path', {'path': path}, {'ok': True})

    @action
    async def set_provenance_perf_csv(self, path: str) -> None:
        self._provenance_perf_csv = path
        log_action(self.logger, 'set_provenance_perf_csv', {'path': path}, {'ok': True})

    @action
    async def set_run_dir(self, run_dir: str) -> None:
        """Set the run log directory (contains actions.jsonl and llm_calls.jsonl)."""
        self._run_dir = run_dir
        log_action(self.logger, 'set_run_dir', {'run_dir': run_dir}, {'ok': True})

    @action
    async def set_top_n(self, n: int) -> None:
        self._top_n = int(n)

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    async def _fetch_leaderboard(self) -> list[tuple[str, float, dict]]:
        if not self._tournament:
            return []
        try:
            board = await self._tournament.get_leaderboard()
            return [
                (row[0], float(row[1]), row[2])
                for row in (board or [])
                if isinstance(row, (list, tuple)) and len(row) == 3
            ]
        except Exception as exc:
            _log.warning('Could not fetch leaderboard: %s', exc)
            return []

    def _resolve_run_dir(self) -> str | None:
        """Try to find the run dir from various sources."""
        if self._run_dir and os.path.isdir(self._run_dir):
            return self._run_dir
        # Try to derive from provenance buffer path
        if self._provenance_buffer_path:
            parent = os.path.dirname(os.path.abspath(self._provenance_buffer_path))
            if os.path.isfile(os.path.join(parent, 'actions.jsonl')):
                return parent
        # Try the global logging context
        try:
            from academy_coscientist.utils.utils_logging import get_run_dir
            rd = get_run_dir()
            if os.path.isdir(rd):
                return rd
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @action
    async def generate_pdf_report(
        self,
        topic: str,
        output_path: str = 'coscientist_report.pdf',
    ) -> str:
        """Collect all data, call Claude for narratives, render PDF."""
        print('[ProvenancePDFReportAgent] Collecting data…', flush=True)

        # --- Gather all data sources ---
        leaderboard  = await self._fetch_leaderboard()
        poc_results  = _read_poc_results(self._poc_dir)

        run_dir = self._resolve_run_dir()
        actions_path   = os.path.join(run_dir, 'actions.jsonl')   if run_dir else None
        llm_calls_path = os.path.join(run_dir, 'llm_calls.jsonl') if run_dir else None

        actions   = _read_jsonl(actions_path)
        llm_calls = _read_jsonl(llm_calls_path)

        action_stats    = _extract_action_stats_from_actions(actions)
        review_rounds   = _extract_review_rounds_from_actions(actions)
        full_reviews    = _extract_full_reviews(llm_calls)
        full_refinements= _extract_full_refinements(llm_calls)
        llm_stats       = _extract_llm_stats(llm_calls)

        perf_rows = _read_perf_csv(self._provenance_perf_csv)
        perf_agg  = _summarise_perf_csv(perf_rows)

        # --- Build poc lookup ---
        poc_by_hyp: dict[str, dict] = {
            str(p.get('hypothesis_id', '')): p
            for p in poc_results
            if p.get('hypothesis_id')
        }

        print('[ProvenancePDFReportAgent] Calling Claude for narrative sections…', flush=True)

        # --- Parallel Claude narrative calls ---
        top_rows = leaderboard[:self._top_n]

        async def _safe_hyp_narrative(hid: str, payload: dict) -> tuple[str, str]:
            rrs  = review_rounds.get(hid, [])
            revs = full_reviews.get(hid, [])
            refs = full_refinements.get(hid, [])
            poc  = poc_by_hyp.get(hid)
            text = await _llm_hyp_narrative(hid, payload, revs, refs, rrs, poc)
            return hid, text

        exec_task     = asyncio.create_task(
            _llm_exec_summary(topic, leaderboard, poc_results, llm_stats, action_stats)
        )
        narrative_tasks = [
            asyncio.create_task(_safe_hyp_narrative(hid, payload))
            for hid, _, payload in top_rows
        ]

        exec_summary = await exec_task
        narrative_results = await asyncio.gather(*narrative_tasks, return_exceptions=True)

        hyp_narratives: dict[str, str] = {}
        for result in narrative_results:
            if isinstance(result, Exception):
                _log.warning('Hypothesis narrative failed: %s', result)
            elif isinstance(result, tuple):
                hid, text = result
                hyp_narratives[hid] = text

        abs_path = os.path.abspath(output_path)
        print(f'[ProvenancePDFReportAgent] Rendering PDF → {abs_path}', flush=True)

        try:
            _render_pdf(
                output_path=abs_path,
                topic=topic,
                leaderboard=leaderboard,
                exec_summary=exec_summary,
                poc_results=poc_results,
                action_stats=action_stats,
                llm_stats=llm_stats,
                perf_agg=perf_agg,
                review_rounds=review_rounds,
                full_reviews=full_reviews,
                full_refinements=full_refinements,
                hyp_narratives=hyp_narratives,
                top_n=self._top_n,
            )
        except ImportError:
            msg = (
                '[ProvenancePDFReportAgent] reportlab is not installed — '
                'run: pip install reportlab'
            )
            print(msg, flush=True)
            _log.error(msg)
            log_action(self.logger, 'generate_pdf_report',
                       {'topic': topic}, {'error': 'reportlab_not_installed'})
            return ''

        print(f'[ProvenancePDFReportAgent] Report saved → {abs_path}', flush=True)
        log_action(
            self.logger, 'generate_pdf_report',
            {'topic': topic, 'output_path': output_path},
            {
                'pdf_path': abs_path,
                'n_hypotheses': len(leaderboard),
                'n_poc': len(poc_results),
                'n_reviews_detailed': sum(len(v) for v in full_reviews.values()),
                'n_refinements': sum(len(v) for v in full_refinements.values()),
            },
        )
        return abs_path
