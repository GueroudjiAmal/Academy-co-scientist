# academy_coscientist/agents/paper_harvester_agent.py
"""
PaperHarvesterAgent — discovers and downloads the most-cited open-access papers
for each configured research topic, then triggers a vector-index rebuild so
fresh knowledge is available to all agents.

Sources (tried in order):
  1. Semantic Scholar  — most-cited papers, requires optional API key
  2. OpenAlex          — fully open, no key, good citation counts
  3. arXiv             — newest preprints, free, no key required

Loop cadence: configurable (default weekly).  Also exposed as an @action so the
user or supervisor can trigger a harvest on-demand.

API docs:
  Semantic Scholar : https://api.semanticscholar.org/api-docs/
  OpenAlex         : https://docs.openalex.org/
  arXiv            : https://info.arxiv.org/help/api/
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import io

import aiohttp

from academy.agent import Agent, action, loop
from academy_coscientist.utils.config import get_path
from academy_coscientist.utils.utils_logging import log_action, make_struct_logger
from academy_coscientist.utils.utils_papers import extract_pdf_sections, section_title_slug
from academy_coscientist.utils.utils_llm import extract_search_keywords

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_S2_SEARCH_URL      = "https://api.semanticscholar.org/graph/v1/paper/search"
_OPENALEX_URL       = "https://api.openalex.org/works"
_ARXIV_URL          = "https://export.arxiv.org/api/query"

_S2_FIELDS = (
    "title,abstract,citationCount,openAccessPdf,year,"
    "authors,externalIds,publicationTypes"
)
_OPENALEX_SELECT = (
    "title,abstract_inverted_index,cited_by_count,doi,"
    "open_access,publication_year,authorships,ids"
)

_DEFAULT_PAPERS_PER_TOPIC = 20
_DEFAULT_LOOP_INTERVAL    = 7 * 24 * 3600.0   # one week in seconds
_REQUEST_DELAY            = 1.2                # seconds between API requests
_DOWNLOAD_TIMEOUT         = 60                 # seconds per PDF download
_MIN_ABSTRACT_LEN         = 50                 # skip papers with very short/missing abstracts

_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_filename(text: str, suffix: str = "") -> str:
    """Turn arbitrary text into a safe filesystem filename."""
    slug = re.sub(r"[^\w\s-]", "", text)
    slug = re.sub(r"[\s]+", "_", slug).strip("_-")
    slug = slug[:120]
    return f"{slug}{suffix}" if suffix else slug


def _extract_abstract_from_pdf_bytes(data: bytes) -> str:
    """
    Quick abstract extraction from PDF bytes — used only for S2 metadata recovery.

    Delegates to extract_pdf_sections and returns the Abstract section text.
    Falls back to first section or first 1500 chars if no abstract is found.
    """
    try:
        sections = extract_pdf_sections(data)
    except Exception:
        return ""
    for s in sections:
        if "abstract" in s["title"].lower():
            return s["text"][:2000]
    if sections:
        return sections[0]["text"][:2000]
    return ""


def _paper_id_from_meta(meta: dict) -> str:
    """Return a stable paper identifier for deduplication."""
    ext = meta.get("externalIds") or {}
    return (
        ext.get("DOI")
        or ext.get("ArXiv")
        or ext.get("CorpusId")
        or meta.get("paperId", "")
        or hashlib.md5(meta.get("title", "").encode()).hexdigest()[:12]
    )


def _reconstruct_abstract_openalex(inverted_index: dict | None) -> str:
    """
    Reconstruct an abstract string from OpenAlex's inverted-index format.

    OpenAlex stores abstracts as {word: [position, ...]} to save bandwidth.
    """
    if not inverted_index:
        return ""
    pos_word: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            pos_word.append((pos, word))
    pos_word.sort()
    return " ".join(w for _, w in pos_word)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class PaperHarvesterAgent(Agent):
    """
    Periodically discovers the most-cited open-access papers for a list of
    research topics and stores them locally so the vector DB stays current.

    Paper sources are tried in order: Semantic Scholar → OpenAlex → arXiv.
    If one source fails or returns no results, the next is tried automatically.

    Wiring
    ------
    Call ``set_vectordb(handle)`` after launch so the agent can trigger
    an index rebuild after each harvest run.

    Autonomous mode
    ---------------
    Pass ``loop_enabled=True`` (and ``loop_interval`` in seconds) to activate
    the weekly background harvest.  The first harvest starts after
    ``loop_start_delay`` seconds.
    """

    def __init__(
        self,
        topics: list[str] | None = None,
        papers_per_topic: int = _DEFAULT_PAPERS_PER_TOPIC,
        research_papers_dir: str | None = None,
        download_pdfs: bool = True,
        min_citation_count: int = 0,
        loop_enabled: bool = False,
        loop_interval: float = _DEFAULT_LOOP_INTERVAL,
        loop_start_delay: float = 0.0,
        api_key: str | None = None,
    ) -> None:
        super().__init__()
        self.logger = make_struct_logger("PaperHarvesterAgent")

        self._topics: list[str] = list(topics or [])
        self._papers_per_topic = max(1, papers_per_topic)
        self._download_pdfs = download_pdfs
        self._min_citation_count = min_citation_count

        # Resolve paper directory
        cfg_dir = get_path("docs_dir", "research_papers")
        raw_dir = research_papers_dir or cfg_dir or "research_papers"
        self._papers_dir = self._resolve_path(raw_dir)
        self._papers_dir.mkdir(parents=True, exist_ok=True)

        # Resolve abstracts cache directory (for FAISS embeddings input)
        cfg_abstracts = get_path("abstracts_cache_dir", "embeddings/abstracts")
        self._abstracts_dir = self._resolve_path(cfg_abstracts or "embeddings/abstracts")
        self._abstracts_dir.mkdir(parents=True, exist_ok=True)

        # Manifest: tracks already-downloaded paper IDs to avoid re-fetching
        self._manifest_path = self._papers_dir / ".harvest_manifest.json"
        self._downloaded: set[str] = self._load_manifest()

        # Handle to the vector DB agent (set via set_vectordb)
        self._vectordb = None

        # Loop config
        self._loop_enabled = loop_enabled
        self._loop_interval = loop_interval
        self._loop_start_delay = loop_start_delay

        # Semantic Scholar API key (optional, raises rate limit)
        self._api_key: str | None = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

        # Cache: topic text → extracted keyword query (avoids repeated LLM calls)
        self._keyword_cache: dict[str, str] = {}

        log_action(
            self.logger,
            "init",
            {
                "topics": self._topics,
                "papers_per_topic": self._papers_per_topic,
                "papers_dir": str(self._papers_dir),
                "abstracts_dir": str(self._abstracts_dir),
                "loop_enabled": self._loop_enabled,
                "loop_interval_h": self._loop_interval / 3600,
            },
            {"ok": True},
        )

    # ------------------------------------------------------------------
    # Path helper
    # ------------------------------------------------------------------

    def _resolve_path(self, path_like: str) -> Path:
        p = Path(path_like)
        if not p.is_absolute():
            root = Path(__file__).resolve().parents[2]
            p = root / p
        return p

    # ------------------------------------------------------------------
    # Manifest (deduplication)
    # ------------------------------------------------------------------

    def _load_manifest(self) -> set[str]:
        if self._manifest_path.exists():
            try:
                data = json.loads(self._manifest_path.read_text())
                return set(data.get("downloaded", []))
            except Exception:
                pass
        return set()

    def _save_manifest(self) -> None:
        self._manifest_path.write_text(
            json.dumps({"downloaded": sorted(self._downloaded)}, indent=2)
        )

    # ------------------------------------------------------------------
    # Configuration actions
    # ------------------------------------------------------------------

    @action
    async def set_topics(self, topics: list[str]) -> None:
        """Replace the topic list at runtime."""
        self._topics = list(topics)
        log_action(self.logger, "set_topics", {"topics": self._topics}, {"ok": True})

    @action
    async def add_topic(self, topic: str) -> None:
        """Append a topic without replacing existing ones."""
        if topic not in self._topics:
            self._topics.append(topic)
        log_action(self.logger, "add_topic", {"topic": topic}, {"ok": True})

    @action
    async def set_vectordb(self, vectordb) -> None:
        """Wire to a ResearchVectorDBAgent so index is rebuilt after each harvest."""
        self._vectordb = vectordb
        log_action(self.logger, "set_vectordb", {"vectordb": str(vectordb)}, {"ok": True})

    # ------------------------------------------------------------------
    # Keyword extraction (pre-step before API search)
    # ------------------------------------------------------------------

    async def _get_search_query(self, topic: str) -> str:
        """
        Return a short keyword query suitable for paper-database APIs.

        If the topic is already short (≤ 120 chars), it is used as-is.
        Otherwise an LLM call extracts 3-5 focused keywords and the result
        is cached so subsequent calls for the same topic are free.
        """
        if topic in self._keyword_cache:
            return self._keyword_cache[topic]

        if len(topic) <= 120:
            # Short topic — use directly, no LLM needed
            self._keyword_cache[topic] = topic
            return topic

        print(
            f"[Harvester] Extracting search keywords from topic "
            f"({len(topic)} chars) …",
            flush=True,
        )
        try:
            keywords = await extract_search_keywords(topic)
        except Exception as e:
            self.logger.warning(
                "keyword_extraction_failed",
                extra={"error": repr(e)},
            )
            keywords = topic[:120]

        self._keyword_cache[topic] = keywords
        print(f"[Harvester] Search query: {keywords!r}", flush=True)
        log_action(
            self.logger,
            "keyword_extraction",
            {"topic_len": len(topic)},
            {"keywords": keywords},
        )
        return keywords

    # ------------------------------------------------------------------
    # Source 1: Semantic Scholar
    # ------------------------------------------------------------------

    def _s2_headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        return headers

    async def _search_topic_s2(
        self,
        session: aiohttp.ClientSession,
        topic: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Query Semantic Scholar; returns normalised paper dicts or [] on failure.

        Retries up to 3 times with exponential backoff on 429 (rate limited).
        Without an API key S2 enforces a strict per-IP rate limit; retrying
        after a short wait is usually sufficient.
        """
        params = {
            "query": topic,
            "limit": min(limit * 5, 100),
            "fields": _S2_FIELDS,
        }
        max_retries = 3
        backoff = 15  # seconds; doubles each retry
        for attempt in range(1, max_retries + 1):
            try:
                async with session.get(
                    _S2_SEARCH_URL,
                    params=params,
                    headers=self._s2_headers(),
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 429:
                        wait = backoff * (2 ** (attempt - 1))
                        self.logger.warning(
                            "s2_rate_limited",
                            extra={"topic": topic, "attempt": attempt, "retry_in_s": wait},
                        )
                        print(
                            f"[Harvester] S2 rate-limited (attempt {attempt}/{max_retries}), "
                            f"retrying in {wait}s …",
                            flush=True,
                        )
                        await asyncio.sleep(wait)
                        continue
                    if resp.status != 200:
                        self.logger.warning(
                            "s2_search_failed",
                            extra={"topic": topic, "status": resp.status},
                        )
                        return []
                    data = await resp.json()
                    break  # success
            except Exception as e:
                self.logger.error("s2_search_error", extra={"topic": topic, "error": repr(e)})
                return []
        else:
            # All retries exhausted
            self.logger.warning("s2_rate_limited_gave_up", extra={"topic": topic, "attempts": max_retries})
            print(f"[Harvester] S2 rate-limited after {max_retries} attempts — giving up.", flush=True)
            return []

        papers = data.get("data", [])
        raw_count = len(papers)
        has_abstract   = sum(1 for p in papers if len(p.get("abstract") or "") >= _MIN_ABSTRACT_LEN)
        has_pdf_url    = sum(1 for p in papers if isinstance(p.get("openAccessPdf"), dict) and p["openAccessPdf"].get("url"))
        print(
            f"[Harvester] S2 raw={raw_count}  with_abstract={has_abstract}  "
            f"with_pdf_url={has_pdf_url}  topic={topic!r:.60}",
            flush=True,
        )
        papers = [
            p for p in papers
            if len(p.get("abstract") or "") >= _MIN_ABSTRACT_LEN
        ]
        if self._min_citation_count > 0:
            papers = [
                p for p in papers
                if (p.get("citationCount") or 0) >= self._min_citation_count
            ]
        if not papers and raw_count > 0:
            # Some papers came back but lacked abstracts — try to extract from PDF
            recoverable = [
                p for p in data.get("data", [])
                if len(p.get("abstract") or "") < _MIN_ABSTRACT_LEN
                and isinstance(p.get("openAccessPdf"), dict)
                and p["openAccessPdf"].get("url")
            ]
            if recoverable:
                print(
                    f"[Harvester] S2 returned {raw_count} papers for {topic!r} with no "
                    f"abstracts — downloading PDFs for {len(recoverable)} open-access "
                    f"papers to extract abstracts …",
                    flush=True,
                )
                for p in recoverable:
                    pdf_url = p["openAccessPdf"]["url"]
                    try:
                        async with session.get(
                            pdf_url,
                            timeout=aiohttp.ClientTimeout(total=_DOWNLOAD_TIMEOUT),
                            allow_redirects=True,
                        ) as resp:
                            if resp.status == 200:
                                pdf_bytes = await resp.read()
                                extracted = _extract_abstract_from_pdf_bytes(pdf_bytes)
                                if extracted:
                                    p["abstract"] = extracted
                                    p["_abstract_source"] = "pdf_extracted"
                                    p["_pdf_bytes"] = pdf_bytes  # cache to avoid re-download
                    except Exception as e:
                        self.logger.warning(
                            "s2_pdf_extract_failed",
                            extra={"url": pdf_url, "error": repr(e)},
                        )
                # Re-apply filter with newly populated abstracts
                papers = [
                    p for p in recoverable
                    if len(p.get("abstract") or "") >= _MIN_ABSTRACT_LEN
                ]
                if papers:
                    print(
                        f"[Harvester] Recovered {len(papers)} S2 papers via PDF extraction.",
                        flush=True,
                    )
            if not papers:
                no_abstract = sum(1 for p in data.get("data", []) if not p.get("abstract"))
                print(
                    f"[Harvester] S2 returned {raw_count} papers for {topic!r} but "
                    f"{no_abstract} had no abstract and PDF extraction yielded nothing "
                    f"— falling back to OpenAlex.",
                    flush=True,
                )
                self.logger.info(
                    "s2_all_filtered",
                    extra={"topic": topic, "raw": raw_count, "no_abstract": no_abstract,
                           "min_abstract": _MIN_ABSTRACT_LEN,
                           "min_citations": self._min_citation_count},
                )
        papers.sort(key=lambda p: p.get("citationCount") or 0, reverse=True)
        for p in papers:
            p["_source"] = "semantic_scholar"
        return papers[:limit]

    # ------------------------------------------------------------------
    # Source 2: OpenAlex
    # ------------------------------------------------------------------

    async def _search_topic_openalex(
        self,
        session: aiohttp.ClientSession,
        topic: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        Query OpenAlex (fully open, no API key needed).

        Results are normalised to the same dict shape as Semantic Scholar.
        Citation filter is applied; results sorted by citation count.
        """
        params = {
            "search": topic,
            "filter": "open_access.is_oa:true",
            "sort": "cited_by_count:desc",
            "per-page": min(limit * 3, 50),
            "select": _OPENALEX_SELECT,
        }
        # OpenAlex asks for a mailto contact in the User-Agent for the polite pool
        email = os.environ.get("OPENALEX_EMAIL", "academy-coscientist@users.noreply")
        headers = {
            "Accept": "application/json",
            "User-Agent": f"academy-coscientist/1.0 (mailto:{email})",
        }
        try:
            async with session.get(
                _OPENALEX_URL,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    self.logger.warning(
                        "openalex_search_failed",
                        extra={"topic": topic, "status": resp.status},
                    )
                    return []
                data = await resp.json()
        except Exception as e:
            self.logger.error("openalex_search_error", extra={"topic": topic, "error": repr(e)})
            return []

        papers: list[dict] = []
        for r in data.get("results") or []:
            abstract = _reconstruct_abstract_openalex(r.get("abstract_inverted_index"))
            if len(abstract) < _MIN_ABSTRACT_LEN:
                continue

            ids       = r.get("ids") or {}
            doi_raw   = r.get("doi") or ""
            doi       = doi_raw.replace("https://doi.org/", "") or None
            arxiv_raw = ids.get("arxiv") or ""
            arxiv_id  = arxiv_raw.replace("https://arxiv.org/abs/", "") or None

            oa      = r.get("open_access") or {}
            pdf_url = oa.get("oa_url")

            authors = [
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
                "openAccessPdf": {"url": pdf_url} if pdf_url else {},
                "externalIds":   {"DOI": doi, "ArXiv": arxiv_id},
                "_source":       "openalex",
            })

        if self._min_citation_count > 0:
            papers = [p for p in papers if p["citationCount"] >= self._min_citation_count]
        papers.sort(key=lambda p: p["citationCount"], reverse=True)
        return papers[:limit]

    # ------------------------------------------------------------------
    # Source 3: arXiv
    # ------------------------------------------------------------------

    async def _search_topic_arxiv(
        self,
        session: aiohttp.ClientSession,
        topic: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        Query the arXiv Atom API for the newest matching preprints.

        arXiv doesn't provide citation counts, so papers are returned
        in submission-date order (newest first).  Useful for cutting-edge
        research not yet indexed elsewhere.
        """
        params = {
            "search_query": f"all:{topic}",
            "start":        0,
            "max_results":  min(limit * 2, 50),
            "sortBy":       "submittedDate",
            "sortOrder":    "descending",
        }
        try:
            async with session.get(
                _ARXIV_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    self.logger.warning(
                        "arxiv_search_failed",
                        extra={"topic": topic, "status": resp.status},
                    )
                    return []
                xml_text = await resp.text()
        except Exception as e:
            self.logger.error("arxiv_search_error", extra={"topic": topic, "error": repr(e)})
            return []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            self.logger.error("arxiv_xml_parse_error", extra={"error": repr(e)})
            return []

        papers: list[dict] = []
        for entry in root.findall("atom:entry", _ARXIV_NS):
            title_el   = entry.find("atom:title",   _ARXIV_NS)
            summary_el = entry.find("atom:summary", _ARXIV_NS)
            title      = (title_el.text or "").strip()   if title_el   is not None else ""
            abstract   = (summary_el.text or "").strip() if summary_el is not None else ""

            if len(abstract) < _MIN_ABSTRACT_LEN:
                continue

            id_el     = entry.find("atom:id", _ARXIV_NS)
            arxiv_url = (id_el.text or "").strip() if id_el is not None else ""
            arxiv_id  = (
                arxiv_url
                .replace("https://arxiv.org/abs/", "")
                .replace("http://arxiv.org/abs/", "")
            )

            # Prefer the explicit PDF link; fall back to constructing it
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
                "citationCount": 0,  # arXiv doesn't provide citation counts
                "year":          year,
                "authors":       authors,
                "openAccessPdf": {"url": pdf_url} if pdf_url else {},
                "externalIds":   {"ArXiv": arxiv_id},
                "_source":       "arxiv",
            })

        return papers[:limit]

    # ------------------------------------------------------------------
    # Unified search with automatic fallback
    # ------------------------------------------------------------------

    async def _search_topic(
        self,
        session: aiohttp.ClientSession,
        topic: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        Try each paper source in order until one succeeds.

        A keyword extraction pre-step distils long or PDF-derived topics into
        a concise query before hitting any API.

        Order: Semantic Scholar → OpenAlex → arXiv
        """
        query = await self._get_search_query(topic)

        # 1. Semantic Scholar
        papers = await self._search_topic_s2(session, query, limit)
        if papers:
            return papers

        # 2. OpenAlex
        self.logger.warning(
            "harvest_fallback",
            extra={"query": query, "from": "semantic_scholar", "to": "openalex"},
        )
        print(f"[Harvester] S2 → 0 usable papers for {query!r}, trying OpenAlex …", flush=True)
        await asyncio.sleep(_REQUEST_DELAY)
        papers = await self._search_topic_openalex(session, query, limit)
        if papers:
            return papers

        # 3. arXiv
        self.logger.warning(
            "harvest_fallback",
            extra={"query": query, "from": "openalex", "to": "arxiv"},
        )
        print(f"[Harvester] OpenAlex → 0 usable papers for {query!r}, trying arXiv …", flush=True)
        await asyncio.sleep(_REQUEST_DELAY)
        return await self._search_topic_arxiv(session, query, limit)

    # ------------------------------------------------------------------
    # PDF / abstract helpers
    # ------------------------------------------------------------------

    async def _download_pdf(
        self,
        session: aiohttp.ClientSession,
        url: str,
        dest: Path,
    ) -> bool:
        """Download a PDF from *url* to *dest*. Returns True on success."""
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=_DOWNLOAD_TIMEOUT),
                allow_redirects=True,
            ) as resp:
                if resp.status != 200:
                    return False
                content_type = resp.headers.get("Content-Type", "")
                if "pdf" not in content_type and not url.lower().endswith(".pdf"):
                    content = await resp.read()
                    if not content.startswith(b"%PDF"):
                        return False
                    dest.write_bytes(content)
                    return True
                dest.write_bytes(await resp.read())
                return True
        except Exception as e:
            self.logger.warning("pdf_download_error", extra={"url": url, "error": repr(e)})
            return False

    def _save_abstract_txt(self, paper: dict, dest: Path) -> None:
        """Persist title + abstract as a plain-text file for the vector DB."""
        title   = paper.get("title") or "Unknown Title"
        abstract = paper.get("abstract") or ""
        year    = paper.get("year") or ""
        authors = ", ".join(
            a.get("name", "") for a in (paper.get("authors") or [])[:5]
        )
        source  = paper.get("_source", "unknown")
        text = (
            f"Title: {title}\nYear: {year}\nAuthors: {authors}\n"
            f"Source: {source}\n\nAbstract:\n{abstract}\n"
        )
        dest.write_text(text, encoding="utf-8")

    # ------------------------------------------------------------------
    # Section-level saving helpers
    # ------------------------------------------------------------------

    def _save_sections_from_bytes(
        self,
        paper: dict,
        base_name: str,
        raw_bytes: bytes,
        force: bool,
    ) -> int:
        """
        Extract body sections from PDF bytes and save one .txt file per section.

        Files are written to ``self._abstracts_dir`` as::

            {base_name}___{section_slug}.txt

        Returns the number of section files written (0 if extraction failed).
        """
        try:
            sections = extract_pdf_sections(raw_bytes)
        except Exception as e:
            self.logger.warning(
                "section_extract_failed",
                extra={"base": base_name, "error": repr(e)},
            )
            return 0

        paper_title = paper.get("title") or base_name
        year        = paper.get("year") or ""
        saved = 0
        for sec in sections:
            slug = section_title_slug(sec["title"])
            dest = self._abstracts_dir / f"{base_name}___{slug}.txt"
            if not dest.exists() or force:
                header = (
                    f"Paper: {paper_title}\nYear: {year}\n"
                    f"Section: {sec['title']}\n\n"
                )
                dest.write_text(header + sec["text"], encoding="utf-8")
                saved += 1
        return saved

    def _save_abstract_as_section(self, paper: dict, base_name: str) -> None:
        """
        Fallback: save the paper's API-provided abstract as ``{base_name}___Abstract.txt``.

        Used when no PDF is available for full section extraction.
        """
        dest = self._abstracts_dir / f"{base_name}___Abstract.txt"
        title    = paper.get("title") or base_name
        year     = paper.get("year") or ""
        abstract = paper.get("abstract") or ""
        text = (
            f"Paper: {title}\nYear: {year}\nSection: Abstract\n\n{abstract}"
        )
        dest.write_text(text, encoding="utf-8")

    # ------------------------------------------------------------------
    # Main harvest action
    # ------------------------------------------------------------------

    @action
    async def harvest_papers(
        self,
        topics: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Download the most-cited open-access papers for every topic.

        Sources are tried in order: Semantic Scholar → OpenAlex → arXiv.

        Parameters
        ----------
        topics:
            Override the agent's topic list for this run only.
        force:
            If True, re-download papers already in the manifest.

        Returns
        -------
        A summary dict: {topic: {"fetched": N, "downloaded_pdf": M, "abstract_only": K, "source": S}}
        """
        active_topics = topics if topics is not None else list(self._topics)
        if not active_topics:
            self.logger.warning("harvest_no_topics", extra={})
            return {}

        summary: dict[str, Any] = {}
        t0 = time.monotonic()

        connector = aiohttp.TCPConnector(limit=4)
        async with aiohttp.ClientSession(connector=connector) as session:
            for topic in active_topics:
                self.logger.info("harvest_topic_start", extra={"topic": topic})
                papers = await self._search_topic(session, topic, self._papers_per_topic)
                await asyncio.sleep(_REQUEST_DELAY)

                source      = papers[0].get("_source", "unknown") if papers else "none"
                fetched     = len(papers)
                pdfs_saved  = 0
                abstracts_saved = 0

                for paper in papers:
                    pid = _paper_id_from_meta(paper)
                    if pid in self._downloaded and not force:
                        continue

                    title      = paper.get("title") or f"paper_{pid}"
                    safe_title = _safe_filename(title)
                    year       = paper.get("year") or "0000"
                    base_name  = f"{year}_{safe_title}"

                    pdf_dest  = self._papers_dir / f"{base_name}.pdf"
                    txt_dest  = self._papers_dir / f"{base_name}.txt"
                    # Canonical existence check: Abstract section file
                    abstract_section_dest = self._abstracts_dir / f"{base_name}___Abstract.txt"

                    # Skip if already fully processed (exists in abstracts cache)
                    if not force and abstract_section_dest.exists():
                        self._downloaded.add(pid)
                        continue

                    # --- Try PDF download first ---
                    pdf_saved  = False
                    pdf_bytes_cached: bytes | None = paper.get("_pdf_bytes")
                    if self._download_pdfs:
                        pdf_info = paper.get("openAccessPdf") or {}
                        pdf_url  = pdf_info.get("url") if isinstance(pdf_info, dict) else None
                        if pdf_url:
                            if not pdf_dest.exists() or force:
                                if pdf_bytes_cached is not None:
                                    # Reuse bytes already fetched during S2 recovery
                                    pdf_dest.write_bytes(pdf_bytes_cached)
                                    pdf_saved = True
                                else:
                                    pdf_saved = await self._download_pdf(session, pdf_url, pdf_dest)
                                if pdf_saved:
                                    pdfs_saved += 1
                                    self.logger.info(
                                        "pdf_downloaded",
                                        extra={"title": title, "year": year, "source": source},
                                    )
                            else:
                                pdf_saved = True  # already on disk

                    # --- Extract sections from PDF and save one file per section ---
                    if not abstract_section_dest.exists() or force:
                        raw_bytes: bytes | None = None
                        if pdf_bytes_cached is not None:
                            raw_bytes = pdf_bytes_cached
                        elif pdf_saved and pdf_dest.exists():
                            try:
                                raw_bytes = pdf_dest.read_bytes()
                            except Exception:
                                raw_bytes = None

                        if raw_bytes is not None:
                            n_sections = self._save_sections_from_bytes(
                                paper, base_name, raw_bytes, force
                            )
                            if n_sections == 0:
                                # PDF extraction yielded nothing — fall back to abstract
                                self._save_abstract_as_section(paper, base_name)
                        else:
                            # No PDF available — save API abstract as single section
                            self._save_abstract_as_section(paper, base_name)
                        abstracts_saved += 1

                    # --- Also save human-readable .txt in papers dir (for browsing) ---
                    if not txt_dest.exists() or force:
                        self._save_abstract_txt(paper, txt_dest)

                    self._downloaded.add(pid)
                    await asyncio.sleep(0.05)

                self._save_manifest()
                summary[topic] = {
                    "fetched":       fetched,
                    "downloaded_pdf": pdfs_saved,
                    "abstract_only": abstracts_saved,
                    "source":        source,
                }
                log_action(
                    self.logger,
                    "harvest_topic_done",
                    {"topic": topic},
                    summary[topic],
                )
                print(
                    f"[Harvester] {topic!r}: "
                    f"{fetched} papers via {source}, "
                    f"{pdfs_saved} PDFs, "
                    f"{abstracts_saved} abstracts saved.",
                    flush=True,
                )
                await asyncio.sleep(_REQUEST_DELAY)

        elapsed = time.monotonic() - t0

        # Rebuild vector index with fresh content
        if self._vectordb is not None:
            try:
                await self._vectordb.rebuild_index()
                log_action(
                    self.logger, "vectordb_rebuilt", {"elapsed_s": elapsed}, {"ok": True}
                )
            except Exception as e:
                self.logger.error("vectordb_rebuild_failed", extra={"error": repr(e)})

        log_action(
            self.logger,
            "harvest_done",
            {"topics": active_topics, "elapsed_s": round(elapsed, 1)},
            summary,
        )
        return summary

    @action
    async def get_manifest(self) -> dict[str, Any]:
        """Return the set of already-downloaded paper IDs."""
        return {"downloaded_count": len(self._downloaded), "ids": sorted(self._downloaded)}

    @action
    async def clear_manifest(self) -> None:
        """Reset the deduplication manifest so all papers are re-fetched next run."""
        self._downloaded.clear()
        self._save_manifest()
        log_action(self.logger, "manifest_cleared", {}, {"ok": True})

    # ------------------------------------------------------------------
    # Autonomous loop (weekly harvest)
    # ------------------------------------------------------------------

    @loop
    async def harvest_loop(self, shutdown: asyncio.Event) -> None:
        """
        Background loop: harvest papers on the configured schedule (default: weekly).

        The first run happens after ``loop_start_delay`` seconds, then repeats
        every ``loop_interval`` seconds.
        """
        if not self._loop_enabled:
            return

        if self._loop_start_delay > 0:
            self.logger.info(
                "harvest_loop_waiting",
                extra={"delay_s": self._loop_start_delay},
            )
            await asyncio.sleep(self._loop_start_delay)

        while not shutdown.is_set():
            self.logger.info("harvest_loop_cycle_start", extra={"topics": self._topics})
            print(f"\n[Harvester] Starting harvest for topics: {self._topics}", flush=True)
            try:
                summary = await self.harvest_papers()
                total_pdfs = sum(v["downloaded_pdf"] for v in summary.values())
                total_abs  = sum(v["abstract_only"]  for v in summary.values())
                sources    = {v["source"] for v in summary.values()}
                print(
                    f"[Harvester] Harvest complete: "
                    f"{total_pdfs} new PDFs, {total_abs} new abstracts "
                    f"(sources: {', '.join(sorted(sources))}).",
                    flush=True,
                )
            except Exception as e:
                self.logger.error("harvest_loop_error", extra={"error": repr(e)})

            # Sleep in short chunks so shutdown is responsive
            remaining = self._loop_interval
            chunk = 60.0
            while remaining > 0 and not shutdown.is_set():
                await asyncio.sleep(min(chunk, remaining))
                remaining -= chunk
