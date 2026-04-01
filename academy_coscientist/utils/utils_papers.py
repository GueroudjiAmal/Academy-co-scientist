# academy_coscientist/utils/utils_papers.py
"""
Lightweight paper search utility for agents that need live literature context
without the full harvesting pipeline.

Sources tried in order:
  1. Semantic Scholar  — most-cited papers; reads SEMANTIC_SCHOLAR_API_KEY env var
  2. OpenAlex          — fully open; reads OPENALEX_EMAIL env var for polite pool
  3. arXiv             — newest preprints; no key needed

Returns normalised paper dicts so callers don't need to know which source fired:

    {
        "title":         str,
        "abstract":      str,
        "year":          int | None,
        "authors":       [{"name": str}, ...],
        "citationCount": int,
        "openAccessPdf": {"url": str} | {},
        "externalIds":   {"DOI": str | None, "ArXiv": str | None},
        "_source":       "semantic_scholar" | "openalex" | "arxiv",
    }
"""

from __future__ import annotations

import asyncio
import os
import re
import xml.etree.ElementTree as ET
from typing import Any

import aiohttp

# ---------------------------------------------------------------------------
# Constants (shared with paper_harvester_agent but kept independent)
# ---------------------------------------------------------------------------

_S2_URL          = "https://api.semanticscholar.org/graph/v1/paper/search"
_OPENALEX_URL    = "https://api.openalex.org/works"
_ARXIV_URL       = "https://export.arxiv.org/api/query"

_S2_FIELDS       = "title,abstract,citationCount,openAccessPdf,year,authors,externalIds"
_OPENALEX_SELECT = "title,abstract_inverted_index,cited_by_count,doi,open_access,publication_year,authorships,ids"
_ARXIV_NS        = {"atom": "http://www.w3.org/2005/Atom"}

_MIN_ABSTRACT    = 80   # characters — shorter than harvester to keep more results
_TIMEOUT         = aiohttp.ClientTimeout(total=25)

# ---------------------------------------------------------------------------
# Section extraction constants
# ---------------------------------------------------------------------------

# Sections that carry no scientific content worth indexing
_SKIP_SECTION_NAMES: frozenset[str] = frozenset({
    "acknowledgement", "acknowledgements", "acknowledgment", "acknowledgments",
    "references", "bibliography", "works cited",
    "author contributions", "authors contributions",
    "funding", "competing interests", "conflict of interest",
    "conflicts of interest", "data availability", "data availability statement",
    "supplementary material", "supplementary materials",
    "appendix", "declaration of interests", "ethical approval",
    "ethics statement", "supporting information",
})

# Canonical section names — exact lower-cased match is a reliable header signal
_KNOWN_SECTION_NAMES: frozenset[str] = frozenset({
    "abstract", "introduction", "related work", "related works",
    "background", "motivation", "problem statement", "preliminaries",
    "methodology", "method", "methods", "approach", "framework",
    "model", "architecture", "system design", "system overview",
    "experiments", "experimental setup", "experimental results",
    "experimental evaluation", "results", "evaluation", "analysis",
    "discussion", "conclusion", "conclusions", "future work",
    "limitations", "summary",
})

# Matches numbered headings: "1. Introduction", "2 Methods", "I. Background"
_NUMBERED_HEADER_RE = re.compile(
    r"^\s*(?:\d{1,2}|[IVXivx]{1,4})[.\s]\s*([A-Z][A-Za-z &\-/:(,)]{1,60})\s*$"
)


# ---------------------------------------------------------------------------
# PDF-as-topic: extract text from a PDF and use it as the research topic
# ---------------------------------------------------------------------------

def extract_pdf_text(path: str, max_chars: int = 16000) -> str:
    """Extract and return plain text from a PDF file.

    Reads all pages in order and concatenates their text.  Truncates to
    *max_chars* so the result fits comfortably in an LLM prompt.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("pypdf is required to use a PDF as a topic. Install it with: pip install pypdf") from exc

    reader = PdfReader(path)
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text.strip())

    full = "\n\n".join(p for p in parts if p)
    if len(full) > max_chars:
        full = full[:max_chars] + "\n\n[... document truncated ...]"
    return full


def resolve_topic(topic: str) -> str:
    """Return the research topic string to use in LLM prompts.

    If *topic* is a path to an existing PDF file the function extracts the
    document text and returns it (prefixed with the filename so agents know
    what they are reading).  Otherwise *topic* is returned unchanged.
    """
    import os
    if topic.lower().endswith(".pdf") and os.path.isfile(topic):
        filename = os.path.basename(topic)
        print(f"[topic] Reading topic from PDF: {filename}", flush=True)
        text = extract_pdf_text(topic)
        return f"[Source document: {filename}]\n\n{text}"
    return topic


# ---------------------------------------------------------------------------
# PDF section extraction
# ---------------------------------------------------------------------------

def _is_section_header(line: str) -> bool:
    """Heuristic: return True if *line* looks like a paper section header."""
    s = line.strip()
    if not s or len(s) > 80:
        return False
    # All-caps line (e.g. "INTRODUCTION", "RELATED WORK")
    if s.isupper() and 3 <= len(s) <= 60:
        return True
    # Numbered heading: "1. Introduction", "2 Methods", "I. Background"
    if _NUMBERED_HEADER_RE.match(s):
        return True
    # Exact match of a known section name (any case)
    if s.lower().rstrip(".") in _KNOWN_SECTION_NAMES:
        return True
    return False


def _should_skip_section(title: str) -> bool:
    """Return True for sections that carry no scientific content to index."""
    # Strip leading numbering: "5. References" → "references"
    clean = re.sub(r"^[\d\s.IVXivx]+", "", title).strip().lower().rstrip(".")
    return clean in _SKIP_SECTION_NAMES or any(
        skip in clean for skip in _SKIP_SECTION_NAMES
    )


def section_title_slug(title: str) -> str:
    """Return a safe filesystem slug for a section title (strips numbering)."""
    clean = re.sub(r"^[\d\s.IVXivx]+", "", title).strip()
    slug = re.sub(r"[^\w\s-]", "", clean)
    slug = re.sub(r"\s+", "_", slug).strip("_-")
    return slug[:60] or "Section"


def extract_pdf_sections(
    source: "str | bytes",
    *,
    max_chars_per_section: int = 4000,
    min_chars: int = 100,
) -> list[dict[str, str]]:
    """
    Parse a research-paper PDF into its body sections.

    Each element of the returned list is ``{"title": str, "text": str}``.

    Content before the abstract (title page, authors, affiliations) and the
    bibliography/references section are automatically discarded.  Sections
    listed in ``_SKIP_SECTION_NAMES`` (acknowledgements, funding, etc.) are
    also dropped.

    Parameters
    ----------
    source:
        Absolute file path (``str``) or raw PDF bytes.
    max_chars_per_section:
        Truncate each section body to this many characters.
    min_chars:
        Drop sections shorter than this (noise / empty pages).
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for section extraction. "
            "Install with: pip install pypdf"
        ) from exc

    import io as _io

    if isinstance(source, bytes):
        reader = PdfReader(_io.BytesIO(source))
    else:
        reader = PdfReader(str(source))

    # Concatenate all pages
    pages: list[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        pages.append(t)
    full_text = "\n".join(pages)

    # Strip front matter — everything before the abstract
    lower = full_text.lower()
    body_start = 0
    for token in ["\nabstract\n", "\nabstract:", "abstract\n", "abstract:"]:
        pos = lower.find(token)
        if pos != -1:
            body_start = pos
            break
    body = full_text[body_start:]
    lower_body = body.lower()

    # Strip bibliography — cut at the LAST references/bibliography header
    body_end = len(body)
    for token in [
        "\nreferences\n", "\nreferences:", "\nreferences ",
        "\nbibliography\n", "\nbibliography:",
        "\nworks cited\n", "\nworks cited:",
    ]:
        pos = lower_body.rfind(token)
        if 0 < pos < body_end:
            body_end = pos
    body = body[:body_end]

    # Split into sections
    lines = body.split("\n")
    sections: list[tuple[str, list[str]]] = []
    current_title = "Abstract"
    current_lines: list[str] = []

    for line in lines:
        if _is_section_header(line):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_lines))

    # Filter, truncate, and return
    result: list[dict[str, str]] = []
    for title, text_lines in sections:
        if _should_skip_section(title):
            continue
        text = "\n".join(text_lines).strip()
        if len(text) < min_chars:
            continue
        if len(text) > max_chars_per_section:
            text = text[:max_chars_per_section] + "\n[… section truncated …]"
        result.append({"title": title, "text": text})

    return result


# ---------------------------------------------------------------------------
# Internal: abstract reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_abstract_openalex(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    pos_word: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            pos_word.append((pos, word))
    pos_word.sort()
    return " ".join(w for _, w in pos_word)


# ---------------------------------------------------------------------------
# Source 1: Semantic Scholar
# ---------------------------------------------------------------------------

async def _search_s2(
    session: aiohttp.ClientSession,
    topic: str,
    limit: int,
    api_key: str | None,
) -> list[dict[str, Any]]:
    headers: dict[str, str] = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    params = {
        "query":  topic,
        "limit":  min(limit * 4, 80),
        "fields": _S2_FIELDS,
    }
    try:
        async with session.get(_S2_URL, params=params, headers=headers, timeout=_TIMEOUT) as resp:
            if resp.status in (429, 503):
                await asyncio.sleep(5)
                return []
            if resp.status != 200:
                return []
            data = await resp.json()
    except Exception:
        return []

    papers = [
        p for p in (data.get("data") or [])
        if len(p.get("abstract") or "") >= _MIN_ABSTRACT
    ]
    papers.sort(key=lambda p: p.get("citationCount") or 0, reverse=True)
    for p in papers:
        p["_source"] = "semantic_scholar"
    return papers[:limit]


# ---------------------------------------------------------------------------
# Source 2: OpenAlex
# ---------------------------------------------------------------------------

async def _search_openalex(
    session: aiohttp.ClientSession,
    topic: str,
    limit: int,
    email: str | None,
) -> list[dict[str, Any]]:
    ua = f"academy-coscientist/1.0 (mailto:{email or 'academy@users.noreply'})"
    headers = {"Accept": "application/json", "User-Agent": ua}
    params  = {
        "search":   topic,
        "filter":   "open_access.is_oa:true",
        "sort":     "cited_by_count:desc",
        "per-page": min(limit * 3, 50),
        "select":   _OPENALEX_SELECT,
    }
    try:
        async with session.get(_OPENALEX_URL, params=params, headers=headers, timeout=_TIMEOUT) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
    except Exception:
        return []

    papers: list[dict] = []
    for r in data.get("results") or []:
        abstract = _reconstruct_abstract_openalex(r.get("abstract_inverted_index"))
        if len(abstract) < _MIN_ABSTRACT:
            continue

        ids      = r.get("ids") or {}
        doi      = (r.get("doi") or "").replace("https://doi.org/", "") or None
        arxiv_id = (ids.get("arxiv") or "").replace("https://arxiv.org/abs/", "") or None
        oa_url   = (r.get("open_access") or {}).get("oa_url")
        authors  = [
            {"name": (a.get("author") or {}).get("display_name", "")}
            for a in (r.get("authorships") or [])[:5]
            if (a.get("author") or {}).get("display_name")
        ]

        papers.append({
            "title":         r.get("title") or "",
            "abstract":      abstract,
            "citationCount": r.get("cited_by_count") or 0,
            "year":          r.get("publication_year"),
            "authors":       authors,
            "openAccessPdf": {"url": oa_url} if oa_url else {},
            "externalIds":   {"DOI": doi, "ArXiv": arxiv_id},
            "_source":       "openalex",
        })

    papers.sort(key=lambda p: p["citationCount"], reverse=True)
    return papers[:limit]


# ---------------------------------------------------------------------------
# Source 3: arXiv
# ---------------------------------------------------------------------------

async def _search_arxiv(
    session: aiohttp.ClientSession,
    topic: str,
    limit: int,
) -> list[dict[str, Any]]:
    params = {
        "search_query": f"all:{topic}",
        "start":        0,
        "max_results":  min(limit * 2, 50),
        "sortBy":       "submittedDate",
        "sortOrder":    "descending",
    }
    try:
        async with session.get(_ARXIV_URL, params=params, timeout=_TIMEOUT) as resp:
            if resp.status != 200:
                return []
            xml_text = await resp.text()
    except Exception:
        return []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    papers: list[dict] = []
    for entry in root.findall("atom:entry", _ARXIV_NS):
        title_el   = entry.find("atom:title",   _ARXIV_NS)
        summary_el = entry.find("atom:summary", _ARXIV_NS)
        title    = (title_el.text or "").strip()   if title_el   is not None else ""
        abstract = (summary_el.text or "").strip() if summary_el is not None else ""
        if len(abstract) < _MIN_ABSTRACT:
            continue

        id_el    = entry.find("atom:id", _ARXIV_NS)
        arxiv_url = (id_el.text or "").strip() if id_el is not None else ""
        arxiv_id  = (
            arxiv_url
            .replace("https://arxiv.org/abs/", "")
            .replace("http://arxiv.org/abs/", "")
        )

        pdf_url: str | None = None
        for link in entry.findall("atom:link", _ARXIV_NS):
            if link.get("type") == "application/pdf" or link.get("title") == "pdf":
                pdf_url = link.get("href")
                break
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        year: int | None = None
        pub_el = entry.find("atom:published", _ARXIV_NS)
        if pub_el is not None and pub_el.text:
            try:
                year = int(pub_el.text[:4])
            except ValueError:
                pass

        authors = []
        for auth_el in entry.findall("atom:author", _ARXIV_NS)[:5]:
            name_el = auth_el.find("atom:name", _ARXIV_NS)
            if name_el is not None and name_el.text:
                authors.append({"name": name_el.text.strip()})

        papers.append({
            "title":         title,
            "abstract":      abstract,
            "citationCount": 0,
            "year":          year,
            "authors":       authors,
            "openAccessPdf": {"url": pdf_url} if pdf_url else {},
            "externalIds":   {"ArXiv": arxiv_id},
            "_source":       "arxiv",
        })

    return papers[:limit]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def search_papers(
    topic: str,
    limit: int = 5,
    min_citation_count: int = 0,
) -> list[dict[str, Any]]:
    """
    Search for papers relevant to *topic* using S2 → OpenAlex → arXiv fallback.

    Parameters
    ----------
    topic:
        Free-text research topic or query string.
    limit:
        Maximum number of papers to return.
    min_citation_count:
        Filter out papers below this citation threshold (ignored for arXiv).

    Returns
    -------
    List of normalised paper dicts (see module docstring).
    """
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    email   = os.environ.get("OPENALEX_EMAIL")

    connector = aiohttp.TCPConnector(limit=4)
    async with aiohttp.ClientSession(connector=connector) as session:
        # 1. Semantic Scholar
        papers = await _search_s2(session, topic, limit, api_key)
        if papers:
            if min_citation_count:
                papers = [p for p in papers if p.get("citationCount", 0) >= min_citation_count]
            if papers:
                return papers

        await asyncio.sleep(0.5)

        # 2. OpenAlex
        papers = await _search_openalex(session, topic, limit, email)
        if papers:
            if min_citation_count:
                papers = [p for p in papers if p.get("citationCount", 0) >= min_citation_count]
            if papers:
                return papers

        await asyncio.sleep(0.5)

        # 3. arXiv (citation filter not applied — arXiv has no counts)
        return await _search_arxiv(session, topic, limit)


def papers_to_context_text(papers: list[dict[str, Any]], max_per_paper: int = 600) -> str:
    """
    Format a list of paper dicts into a compact literature-context string
    suitable for injection into an LLM prompt.
    """
    if not papers:
        return "No relevant literature found."
    blocks: list[str] = []
    for i, p in enumerate(papers, 1):
        authors = ", ".join(a.get("name", "") for a in (p.get("authors") or [])[:3])
        year    = p.get("year") or "n/a"
        source  = p.get("_source", "?")
        abstract = (p.get("abstract") or "")[:max_per_paper]
        if len(p.get("abstract") or "") > max_per_paper:
            abstract += " …"
        blocks.append(
            f"[{i}] {p.get('title','(no title)')} ({authors}, {year}) [{source}]\n"
            f"    {abstract}"
        )
    return "\n\n".join(blocks)
