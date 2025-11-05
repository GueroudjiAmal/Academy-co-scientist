from __future__ import annotations

import json
import os
from typing import Any

from academy_coscientist.utils.utils_llm import embed_texts
from academy_coscientist.utils.utils_logging import make_struct_logger

EMBED_DIR = 'embeddings'


def _ensure_embed_dir() -> None:
    os.makedirs(EMBED_DIR, exist_ok=True)


def _cache_path_for_pdf(pdf_name: str) -> str:
    safe_name = pdf_name.replace('/', '_')
    return os.path.join(EMBED_DIR, f'{safe_name}.json')


def _load_pdf_cache(pdf_name: str) -> list[dict[str, Any]]:
    path = _cache_path_for_pdf(pdf_name)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def _save_pdf_cache(pdf_name: str, rows: list[dict[str, Any]]) -> None:
    path = _cache_path_for_pdf(pdf_name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


async def build_embeddings_with_cache(
    texts: list[str],
    metas: list[dict[str, Any]],
) -> list[list[float]]:
    """Given parallel lists:
        texts[i] = chunk text
        metas[i] = {
            "source_file": str,
            "chunk_index": int,
            "file_size": int,
            "file_mtime": int,
            ...
        }

    We will:
      - Load per-PDF cache from embeddings/<pdfname>.json
      - For each chunk, see if we already have an embedding for
        (chunk_index, file_size, file_mtime).
      - For missing ones, call embed_texts() and then update caches.
      - Return embeddings list aligned with `texts`.
    """
    logger = make_struct_logger('EmbeddingCache')
    _ensure_embed_dir()

    # Load all caches, group rows per PDF name, and map keys.
    # cache_maps[pdf]["<chunk_index>:<size>:<mtime>"] = embedding
    cache_rows_by_pdf: dict[str, list[dict[str, Any]]] = {}
    cache_maps: dict[str, dict[str, dict[str, Any]]] = {}

    unique_pdfs = {m['source_file'] for m in metas if 'source_file' in m}
    for pdf_name in unique_pdfs:
        rows = _load_pdf_cache(pdf_name)
        cache_rows_by_pdf[pdf_name] = rows
        mp: dict[str, dict[str, Any]] = {}
        for r in rows:
            ck = f'{r.get("chunk_index")}:{r.get("file_size")}:{r.get("file_mtime")}'
            mp[ck] = r
        cache_maps[pdf_name] = mp

    logger.debug(
        'loaded existing caches',
        extra={
            'num_pdfs': len(unique_pdfs),
            'pdfs': list(unique_pdfs)[:10],
        },
    )

    # We'll gather which items we still need to embed.
    embeddings_out: list[list[float]] = [None] * len(texts)  # type: ignore
    missing_batch: list[str] = []
    missing_meta_index: list[int] = []

    for i, (t, m) in enumerate(zip(texts, metas, strict=False)):
        pdf_name = m['source_file']
        cidx = m['chunk_index']
        fsize = m['file_size']
        fmtime = m['file_mtime']

        ck = f'{cidx}:{fsize}:{fmtime}'
        cached_row = cache_maps[pdf_name].get(ck)

        if cached_row and 'embedding' in cached_row:
            embeddings_out[i] = cached_row['embedding']
        else:
            missing_batch.append(t)
            missing_meta_index.append(i)

    logger.debug(
        'cache lookup done',
        extra={
            'total_chunks': len(texts),
            'missing': len(missing_batch),
        },
    )

    # Embed the missing ones in one or more batches.
    # We'll just do one batch call here, you could batch-split if needed.
    if missing_batch:
        new_embs = await embed_texts(
            missing_batch,
            context={'stage': 'embedding_cache_build'},
        )
        # assign back
        for j, emb in enumerate(new_embs):
            global_i = missing_meta_index[j]
            embeddings_out[global_i] = emb

    # Now we must update / write cache_rows_by_pdf for each new item
    for global_i in missing_meta_index:
        m = metas[global_i]
        pdf_name = m['source_file']
        cidx = m['chunk_index']
        fsize = m['file_size']
        fmtime = m['file_mtime']
        text_val = texts[global_i]
        emb_val = embeddings_out[global_i]

        # augment pdf cache rows
        cache_rows = cache_rows_by_pdf[pdf_name]
        # check if row already existed (might not, but if so we overwrite)
        replaced = False
        for row in cache_rows:
            if (
                row.get('chunk_index') == cidx
                and row.get('file_size') == fsize
                and row.get('file_mtime') == fmtime
            ):
                row['text'] = text_val
                row['embedding'] = emb_val
                replaced = True
                break
        if not replaced:
            cache_rows.append(
                {
                    'chunk_index': cidx,
                    'file_size': fsize,
                    'file_mtime': fmtime,
                    'text': text_val,
                    'embedding': emb_val,
                },
            )
        cache_rows_by_pdf[pdf_name] = cache_rows

    # Write updated cache files back to disk
    for pdf_name, rows in cache_rows_by_pdf.items():
        _save_pdf_cache(pdf_name, rows)

    logger.debug(
        'cache save complete',
        extra={
            'updated_pdfs': list(cache_rows_by_pdf.keys())[:10],
        },
    )

    # embeddings_out should now be fully populated
    # but in pathological error cases we still ensure no None
    for i, emb in enumerate(embeddings_out):
        if emb is None:
            # Fallback: zero vector if something truly failed
            # We'll just pick a small default dim = 256
            embeddings_out[i] = [0.0] * 256

    return embeddings_out  # List[List[float]]
