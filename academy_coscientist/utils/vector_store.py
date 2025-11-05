from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any

from academy_coscientist.utils import utils_llm
from academy_coscientist.utils.pdf_ingest import AbstractRecord
from academy_coscientist.utils.pdf_ingest import extract_abstract_records
from academy_coscientist.utils.pdf_ingest import save_abstract_json

# A super simple JSONL vector "store" with local caching in embeddings/
# Each entry: {"id": str, "file": str, "title": str, "sha": str, "embedding": [floats], "text": "..."}
# We keep `text` (the abstract) for auditability and to ensure 1-1 correspondence.


@dataclass
class VectorEntry:
    id: str
    file: str
    title: str
    sha: str
    embedding: list[float]
    text: str


def _id_for(file_path: str, sha: str) -> str:
    h = hashlib.sha256((file_path + '::' + sha).encode('utf-8', errors='ignore')).hexdigest()
    return h[:16]


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: list[dict[str, Any]] = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _append_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def _index_path(emb_dir: str) -> str:
    return os.path.join(emb_dir, 'index.jsonl')


def _already_indexed(index: list[dict[str, Any]], file_path: str, sha: str) -> bool:
    for row in index:
        if row.get('file') == file_path and row.get('sha') == sha:
            return True
    return False


async def ensure_embeddings_for_docs(
    docs_dir: str,
    embeddings_dir: str,
    abstracts_dir: str | None = None,
    audit_context: dict[str, Any] | None = None,
) -> tuple[list[VectorEntry], list[VectorEntry]]:
    """Idempotent ingest:
    - Extract abstracts from DOCs/
    - For each abstract, if (file, sha) not in embeddings, create embedding
    - Persist BOTH the abstract JSON and the vector entry
    - Returns (new_entries, all_entries)
    """
    os.makedirs(embeddings_dir, exist_ok=True)
    abstracts_dir = abstracts_dir or os.path.join(embeddings_dir, 'abstracts')
    os.makedirs(abstracts_dir, exist_ok=True)

    # Load current index
    index_file = _index_path(embeddings_dir)
    index_rows = _load_jsonl(index_file)

    # Extract abstracts (pure text + sha)
    records: list[AbstractRecord] = extract_abstract_records(docs_dir)

    # Determine which ones are new
    new_records: list[AbstractRecord] = []
    for rec in records:
        if not _already_indexed(index_rows, rec.file_path, rec.abstract_sha256):
            new_records.append(rec)
            # Save the exact abstract now (for auditing)
            save_abstract_json(rec, abstracts_dir)

    # Embed new ones
    new_entries: list[VectorEntry] = []
    if new_records:
        texts = [r.abstract_text for r in new_records]
        vectors = await utils_llm.embed_texts(
            texts,
            context={
                **(audit_context or {}),
                'call_type': 'embed_abstracts',
                'count': len(texts),
            },
        )
        for rec, vec in zip(new_records, vectors, strict=False):
            vid = _id_for(rec.file_path, rec.abstract_sha256)
            row = {
                'id': vid,
                'file': rec.file_path,
                'title': rec.title_guess,
                'sha': rec.abstract_sha256,
                'embedding': vec,
                'text': rec.abstract_text,  # store exactly what we embedded
            }
            _append_jsonl(index_file, [row])
            new_entries.append(
                VectorEntry(
                    id=vid,
                    file=rec.file_path,
                    title=rec.title_guess,
                    sha=rec.abstract_sha256,
                    embedding=vec,
                    text=rec.abstract_text,
                ),
            )

    # Load final index for return
    final_rows = _load_jsonl(index_file)
    all_entries = [
        VectorEntry(
            id=row['id'],
            file=row['file'],
            title=row.get('title', ''),
            sha=row['sha'],
            embedding=row['embedding'],
            text=row.get('text', ''),
        )
        for row in final_rows
    ]
    return new_entries, all_entries


def cosine_similarity(a: list[float], b: list[float]) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


async def query_similar(
    query_text: str,
    embeddings_dir: str,
    top_k: int = 5,
    audit_context: dict[str, Any] | None = None,
) -> list[tuple[VectorEntry, float]]:
    """Embed the query, then search cosine similarity over stored abstracts."""
    index_file = _index_path(embeddings_dir)
    rows = _load_jsonl(index_file)
    if not rows:
        return []

    qvec = (
        await utils_llm.embed_texts(
            [query_text],
            context={**(audit_context or {}), 'call_type': 'embed_query'},
        )
    )[0]

    entries = [
        VectorEntry(
            id=row['id'],
            file=row['file'],
            title=row.get('title', ''),
            sha=row['sha'],
            embedding=row['embedding'],
            text=row.get('text', ''),
        )
        for row in rows
    ]

    scored = [(ve, cosine_similarity(qvec, ve.embedding)) for ve in entries]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
