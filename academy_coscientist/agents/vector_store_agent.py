from __future__ import annotations

from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from academy.agent import Agent, action
from academy_coscientist.utils.utils_llm import embed_texts
from academy_coscientist.utils.utils_logging import (
    make_struct_logger,
    get_llm_audit_path,
    log_action,
)


class VectorStoreAgent(Agent):
    """
    Simple in-memory FAISS index for semantic retrieval.
    We treat embeddings as 'LLM calls' because they hit the embedding model.
    """

    def __init__(
        self,
        preload_texts: Optional[List[str]] = None,
        preload_metas: Optional[List[Dict[str, Any]]] = None,
        preload_embs: Optional[List[List[float]]] = None,
    ) -> None:
        super().__init__()

        self.logger = make_struct_logger("VectorStoreAgent")

        self._texts: List[str] = preload_texts or []
        self._metas: List[Dict[str, Any]] = preload_metas or []
        self._embs: Optional[List[List[float]]] = preload_embs

        self._index = None  # faiss.IndexFlatIP | None
        self._dim: Optional[int] = None

        self.logger.debug("VectorStoreAgent init", extra={})

    async def _ensure_index(self) -> None:
        """
        Build FAISS index if not already built.
        """
        if self._index is not None:
            return

        if not self._texts:
            # nothing to index
            self._index = faiss.IndexFlatIP(1)  # dummy
            self._dim = 1
            return

        # use cached embeddings if provided, else embed now
        if self._embs and len(self._embs) == len(self._texts):
            embs = self._embs
        else:
            embs = await embed_texts(
                self._texts,
                context={"audit_path": get_llm_audit_path(), "stage": "vector_build"},
            )

        mat = np.asarray(embs, dtype="float32")
        faiss.normalize_L2(mat)
        dim = mat.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(mat)

        self._index = index
        self._dim = dim

        self.logger.debug(
            "faiss_index_built",
            extra={"ntotal": index.ntotal, "dim": dim},
        )

    @action
    async def build_from_texts(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        self._texts = texts
        self._metas = metadatas
        self._embs = embeddings
        self._index = None  # force rebuild on next query
        await self._ensure_index()
        log_action(
            self.logger,
            "build_from_texts",
            {
                "num_chunks": len(texts),
                "example_meta": metadatas[0] if metadatas else None,
                "embeddings_provided": embeddings is not None,
            },
            {"status": "ok", "ntotal": int(self._index.ntotal) if self._index else 0},
        )

    @action
    async def query(
        self,
        query_text: str,
        k: int = 5,
    ) -> Dict[str, Any]:
        await self._ensure_index()

        if self._index is None or self._index.ntotal == 0:
            result = {"matches": []}
            log_action(
                self.logger,
                "query",
                {"query_text": query_text, "k": k},
                result,
            )
            return result

        q_emb = await embed_texts(
            [query_text],
            context={"audit_path": get_llm_audit_path(), "stage": "vector_query"},
        )

        if not q_emb or not q_emb[0]:
            out = {"matches": []}
            log_action(
                self.logger,
                "query",
                {"query_text": query_text, "k": k},
                out,
            )
            return out

        qvec = np.asarray(q_emb, dtype="float32")
        faiss.normalize_L2(qvec)

        D, I = self._index.search(qvec, k)
        sims = D[0]
        idxs = I[0]

        results: List[Dict[str, Any]] = []
        for score, idx in zip(sims, idxs):
            if idx < 0:
                continue
            chunk_text = self._texts[idx]
            chunk_meta = self._metas[idx] if idx < len(self._metas) else {}
            results.append(
                {
                    "text": chunk_text,
                    "metadata": chunk_meta,
                    "score": float(score),
                }
            )

        out = {"matches": results}
        log_action(
            self.logger,
            "query",
            {"query_text": query_text, "k": k},
            out,
        )
        return out
