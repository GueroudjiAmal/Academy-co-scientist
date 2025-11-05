from __future__ import annotations

import re
import os
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

from pypdf import PdfReader

# Robust abstract detectors: handles "Abstract", "ABSTRACT", "Summary", "Résumé", etc.
_ABSTRACT_HEADINGS = [
    r"\babstract\b",
    r"\bsummary\b",
    r"\brésumé\b",            # FR
    r"\bresumen\b",           # ES
    r"\bzusammenfassung\b",   # DE
]
# Common section headings to stop at (case-insensitive, ignores numbering).
_STOP_HEADINGS = [
    r"\bintroduction\b",
    r"\bbackground\b",
    r"\bmethods?\b",
    r"\bmaterials\b",
    r"\bresults?\b",
    r"\b1\.\s+introduction\b",
    r"^\s*\d+\s+\w+",  # numbered headings line
]

HEADING_RE = re.compile("|".join(_ABSTRACT_HEADINGS), re.IGNORECASE)
STOP_RE = re.compile("|".join(_STOP_HEADINGS), re.IGNORECASE)

@dataclass
class AbstractRecord:
    file_path: str
    file_name: str
    title_guess: str
    abstract_text: str
    abstract_sha256: str
    pages_scanned: int

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def _clean(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _guess_title(pdf: PdfReader) -> str:
    # Try metadata title first
    meta_title = ""
    try:
        meta_title = (pdf.metadata.title or "").strip()
    except Exception:
        meta_title = ""
    if meta_title:
        return meta_title

    # Fall back: first non-empty line of first page
    try:
        first_text = pdf.pages[0].extract_text() or ""
        lines = [ln.strip() for ln in first_text.splitlines() if ln.strip()]
        if lines:
            # Heuristic: skip 'Abstract' if it's accidentally first
            if re.match(HEADING_RE, lines[0], flags=0):
                return lines[1] if len(lines) > 1 else ""
            return lines[0][:200]
    except Exception:
        pass
    return ""

def _find_abstract_in_text(full_text: str) -> Optional[str]:
    """
    Extract only the Abstract block:
    - Find first heading that looks like 'Abstract'
    - Then slice until the next major heading (Introduction, Background, etc.)
    - If not found, None
    """
    if not full_text:
        return None

    text = _clean(full_text)

    # Find abstract start (by heading line)
    # Accept cases like 'ABSTRACT' alone in a line, or 'Abstract—' with dash, or 'Abstract:' etc.
    start_match = None
    for m in re.finditer(r"(?mi)^(?:\s*)(" + "|".join(_ABSTRACT_HEADINGS) + r")\s*[:—\-]?\s*$", text):
        start_match = m
        break
    if not start_match:
        # sometimes it's "Abstract:" followed by text on same line
        for m in re.finditer(r"(?mi)^(?:\s*)(" + "|".join(_ABSTRACT_HEADINGS) + r")\s*[:—\-]\s*(.+)$", text):
            # Return whatever follows on that line plus subsequent lines until STOP
            start = m.start(2)
            # find stop after that
            stop = None
            for sm in re.finditer(r"(?mi)^(?:\s*)(?:" + "|".join(_STOP_HEADINGS) + r")\s*[:—\-]?\s*$", text[m.end():]):
                stop = m.end() + sm.start()
                break
            candidate = text[start:stop].strip() if stop else text[start:].strip()
            return candidate if candidate else None

    if not start_match:
        return None

    start = start_match.end()
    # Find the next stop heading after start
    stop = None
    for m in re.finditer(r"(?mi)^(?:\s*)(?:" + "|".join(_STOP_HEADINGS) + r")\s*[:—\-]?\s*$", text[start:]):
        stop = start + m.start()
        break

    abstract = text[start:stop].strip() if stop else text[start:].strip()
    # Defensive: trim if it ran long (e.g., missed a stop); keep first ~3000 chars
    if len(abstract) > 3000:
        abstract = abstract[:3000].rsplit("\n", 1)[0].strip()
    return abstract or None

def _extract_abstract_from_pdf(path: str) -> Tuple[Optional[str], int]:
    """
    Read a PDF and try to locate the Abstract section.
    Strategy:
      - Scan first ~3 pages text concatenated (covers most layouts)
      - Fallback: scan first 5 if not found
    Returns (abstract_text_or_none, pages_scanned)
    """
    reader = PdfReader(path)
    pages_to_scan = min(5, len(reader.pages))
    text_chunks: List[str] = []

    # Start with first 2-3 pages (many venues put abstract on page 1, sometimes spills to page 2)
    primary_scan = min(3, pages_to_scan)
    for i in range(primary_scan):
        try:
            t = reader.pages[i].extract_text() or ""
        except Exception:
            t = ""
        if t:
            text_chunks.append(t)

    abstract = _find_abstract_in_text("\n".join(text_chunks))
    if abstract:
        return abstract, primary_scan

    # fallback: scan up to 5 pages in total
    if pages_to_scan > primary_scan:
        for i in range(primary_scan, pages_to_scan):
            try:
                t = reader.pages[i].extract_text() or ""
            except Exception:
                t = ""
            if t:
                text_chunks.append(t)
        abstract = _find_abstract_in_text("\n".join(text_chunks))
        if abstract:
            return abstract, pages_to_scan

    return None, pages_to_scan

def extract_abstract_records(docs_dir: str) -> List[AbstractRecord]:
    """
    Walk DOCs and return AbstractRecord list (no embeddings here).
    """
    recs: List[AbstractRecord] = []
    for root, _, files in os.walk(docs_dir):
        for fn in files:
            if not fn.lower().endswith(".pdf"):
                continue
            path = os.path.join(root, fn)
            try:
                abstract, scanned = _extract_abstract_from_pdf(path)
                if not abstract:
                    # No abstract found; skip but record empty entry if desired
                    continue
                abstract = _clean(abstract)
                title = ""
                try:
                    title = _guess_title(PdfReader(path))
                except Exception:
                    title = ""
                recs.append(
                    AbstractRecord(
                        file_path=path,
                        file_name=fn,
                        title_guess=title,
                        abstract_text=abstract,
                        abstract_sha256=_sha256(abstract),
                        pages_scanned=scanned,
                    )
                )
            except Exception:
                # Skip unreadable PDFs
                continue
    return recs

def save_abstract_json(record: AbstractRecord, out_dir: str) -> str:
    """
    Save the exact abstract we extracted, so embeddings can be verified later.
    Returns output JSON path.
    """
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(record.file_name))[0]
    out_path = os.path.join(out_dir, f"{base}.abstract.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(record), f, ensure_ascii=False, indent=2)
    return out_path
