# academy_coscientist/agents/research_vector_agent.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss
import numpy as np

from academy.agent import action, Agent
from academy_coscientist.utils.utils_logging import make_struct_logger, log_action
from academy_coscientist.utils.utils_llm import embed_texts
from academy_coscientist.utils.config import get_model, get_path
from academy_coscientist.utils.utils_papers import extract_pdf_sections, section_title_slug


class ResearchVectorDBAgent(Agent):
    """
    Handles vector embeddings of research abstracts and FAISS-based retrieval.

    Behavior (by default, overridable via config.paths):
      - PDFs:        <project_root>/research_papers
      - Abstracts:   <project_root>/embeddings/abstracts   (cached .txt)
      - Embeddings:  <project_root>/embeddings/abstracts   (index + metadata + .npy)

    Config example:

    paths:
      docs_dir: "research_papers"
      embeddings_dir: "embeddings"          # not used directly here
      abstracts_cache_dir: "embeddings/abstracts"

    models:
      embedding: "local-all-MiniLM-L6-v2"   # or OpenAI embedding model
    """

    def __init__(
        self,
        pdf_dir: Optional[str] = None,
        abstracts_dir: Optional[str] = None,
        embeddings_dir: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.logger = make_struct_logger("ResearchVectorDBAgent")

        root = self._project_root()

        # Resolve paths from config, but allow constructor overrides
        cfg_pdfs = get_path("docs_dir", "research_papers") or "research_papers"
        cfg_abstracts = get_path("abstracts_cache_dir", "embeddings/abstracts") or "embeddings/abstracts"

        self.pdf_dir = self._resolve_under_root(pdf_dir or cfg_pdfs, root)
        # For simplicity, we keep abstracts and embedding artifacts in the same dir
        self.abstracts_dir = self._resolve_under_root(abstracts_dir or cfg_abstracts, root)
        self.embeddings_dir = self._resolve_under_root(embeddings_dir or cfg_abstracts, root)

        self.embedding_model = embedding_model or get_model("embedding")

        # Runtime state
        self.index: faiss.Index | None = None
        self.metadata: List[Dict[str, Any]] = []
        self.embeddings_array: np.ndarray | None = None

        log_action(
            self.logger,
            "init",
            {
                "pdf_dir": str(self.pdf_dir),
                "abstracts_dir": str(self.abstracts_dir),
                "embeddings_dir": str(self.embeddings_dir),
                "embedding_model": self.embedding_model,
            },
            {"ok": True},
        )

    # ----------------------------------------------------------------------
    # Path helpers
    # ----------------------------------------------------------------------

    def _project_root(self) -> Path:
        here = Path(__file__).resolve()
        # .../academy-coscientist/academy_coscientist/agents/research_vector_agent.py
        return here.parents[2]

    def _resolve_under_root(self, path_like: str | Path, root: Path) -> Path:
        p = Path(path_like)
        if not p.is_absolute():
            p = root / p
        return p

    # ----------------------------------------------------------------------
    # PDF → Section extraction
    # ----------------------------------------------------------------------

    def _prepare_abstracts_from_pdfs(self) -> None:
        """
        Extract body sections from all PDFs in ``pdf_dir`` and write one
        ``.txt`` file per section into ``abstracts_dir``.

        File naming: ``{pdf_stem}___{section_slug}.txt``

        Sections before the abstract (authors, affiliations) and the
        bibliography/references section are automatically skipped.
        """
        self.abstracts_dir.mkdir(parents=True, exist_ok=True)

        if not self.pdf_dir.exists():
            self.logger.warning("pdf_dir_missing", extra={"pdf_dir": str(self.pdf_dir)})
            log_action(
                self.logger,
                "pdf_dir_missing",
                {"pdf_dir": str(self.pdf_dir)},
                {"ok": False},
            )
            return

        pdf_files = sorted(self.pdf_dir.glob("*.pdf"))
        print(f"[VectorDB] Found {len(pdf_files)} PDFs in {self.pdf_dir}")

        total_sections = 0
        skipped = 0

        for pdf_path in pdf_files:
            try:
                sections = extract_pdf_sections(str(pdf_path))
            except Exception as e:
                self.logger.warning(
                    "pdf_section_extract_failed",
                    extra={"file": str(pdf_path), "error": str(e)},
                )
                skipped += 1
                continue

            if not sections:
                print(f"[VectorDB] No sections extracted from {pdf_path.name}")
                skipped += 1
                continue

            for sec in sections:
                slug = section_title_slug(sec["title"])
                out_path = self.abstracts_dir / f"{pdf_path.stem}___{slug}.txt"
                header = f"Paper: {pdf_path.stem}\nSection: {sec['title']}\n\n"
                out_path.write_text(header + sec["text"], encoding="utf-8")
                total_sections += 1

            print(
                f"[VectorDB] {pdf_path.name}: {len(sections)} sections extracted"
            )

        log_action(
            self.logger,
            "sections_extracted",
            {"pdf_dir": str(self.pdf_dir)},
            {"total_sections": total_sections, "skipped_pdfs": skipped},
        )

    # ----------------------------------------------------------------------
    # Index management
    # ----------------------------------------------------------------------

    @action
    async def ensure_vector_index(
        self,
        docs_dir: Optional[str] = None,
        embeddings_dir: Optional[str] = None,
    ) -> None:
        """
        Ensure FAISS index & metadata exist.

        Backwards compatible:
          - If docs_dir / embeddings_dir are provided, they override config paths.
          - Otherwise, use configured / default paths.
        """
        # Optional overrides (e.g., older code paths)
        root = self._project_root()
        if docs_dir is not None:
            self.abstracts_dir = self._resolve_under_root(docs_dir, root)
        if embeddings_dir is not None:
            self.embeddings_dir = self._resolve_under_root(embeddings_dir, root)

        index_path = self.embeddings_dir / "index.faiss"
        meta_path = self.embeddings_dir / "metadata.json"
        npy_path = self.embeddings_dir / "embeddings.npy"

        # In-memory already
        if self.index is not None and self.metadata:
            log_action(
                self.logger,
                "ensure_vector_index",
                {"mode": "in_memory"},
                {"docs": len(self.metadata)},
            )
            print(f"✅ FAISS index already in memory ({len(self.metadata)} docs)")
            return

        # Try load from disk
        if index_path.exists() and meta_path.exists() and npy_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                self.embeddings_array = np.load(npy_path)
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

                log_action(
                    self.logger,
                    "ensure_vector_index",
                    {
                        "mode": "loaded_from_disk",
                        "index_path": str(index_path),
                        "meta_path": str(meta_path),
                        "npy_path": str(npy_path),
                    },
                    {"docs": len(self.metadata)},
                )
                print(f"✅ Loaded FAISS index ({len(self.metadata)} docs) from {self.embeddings_dir}")
                return
            except Exception as e:
                self.logger.error(
                    "index_load_failed",
                    extra={
                        "index_path": str(index_path),
                        "error": str(e),
                    },
                )
                log_action(
                    self.logger,
                    "ensure_vector_index",
                    {
                        "mode": "load_failed",
                        "index_path": str(index_path),
                        "error": str(e),
                    },
                    {"docs": 0},
                )

        # Need to build from scratch
        print("🔧 No existing index found — extracting abstracts and rebuilding.")
        log_action(
            self.logger,
            "ensure_vector_index",
            {"mode": "rebuild_triggered", "abstracts_dir": str(self.abstracts_dir)},
            {},
        )

        self._prepare_abstracts_from_pdfs()
        await self.rebuild_index()

    @action
    async def rebuild_index(self) -> None:
        """
        Rebuild FAISS index and save embeddings + metadata.
        Uses self.abstracts_dir as source of .txt files.
        """
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        abstracts: List[str] = []
        metas: List[Dict[str, Any]] = []

        for txt_path in sorted(self.abstracts_dir.glob("*.txt")):
            content = txt_path.read_text(encoding="utf-8").strip()
            if not content:
                continue
            abstracts.append(content)
            # Parse section and paper name from filename convention
            # New format: {paper_base}___{section_slug}.txt
            # Old format: {paper_base}.txt
            fname = txt_path.name
            if "___" in fname:
                paper_part, sec_part = fname.rsplit("___", 1)
                section = sec_part.removesuffix(".txt").replace("_", " ")
            else:
                paper_part = fname.removesuffix(".txt")
                section = "Abstract"
            metas.append({
                "file":    fname,
                "paper":   paper_part,
                "section": section,
            })

        if not abstracts:
            self.logger.warning(
                "no_abstract_txt_files",
                extra={"dir": str(self.abstracts_dir)},
            )
            log_action(
                self.logger,
                "rebuild_index_no_docs",
                {"abstracts_dir": str(self.abstracts_dir)},
                {"docs": 0},
            )
            print(f"⚠️ No abstract .txt files found in {self.abstracts_dir}")
            return

        print(f"[VectorDB] Embedding {len(abstracts)} sections using {self.embedding_model}...")
        log_action(
            self.logger,
            "rebuild_index_start",
            {"abstracts_dir": str(self.abstracts_dir), "docs": len(abstracts)},
            {},
        )

        # embed_texts uses config.get_model("embedding") internally
        vectors = await embed_texts(abstracts)
        vec_np = np.array(vectors, dtype=np.float32)

        if vec_np.ndim != 2 or vec_np.shape[0] == 0:
            self.logger.error("embedding_shape_invalid", extra={"shape": vec_np.shape})
            log_action(
                self.logger,
                "embedding_shape_invalid",
                {"abstracts_dir": str(self.abstracts_dir)},
                {"shape": tuple(vec_np.shape)},
            )
            print("❌ Embeddings have invalid shape; aborting index build.")
            return

        dim = vec_np.shape[1]

        # Save embeddings + metadata
        npy_path = self.embeddings_dir / "embeddings.npy"
        meta_path = self.embeddings_dir / "metadata.json"
        index_path = self.embeddings_dir / "index.faiss"

        np.save(npy_path, vec_np)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metas, f, indent=2)

        # Build index
        index = faiss.IndexFlatL2(dim)
        index.add(vec_np)
        faiss.write_index(index, str(index_path))

        self.index = index
        self.metadata = metas
        self.embeddings_array = vec_np

        log_action(
            self.logger,
            "rebuild_index_done",
            {
                "abstracts_dir": str(self.abstracts_dir),
                "embeddings_dir": str(self.embeddings_dir),
                "dim": dim,
            },
            {"docs": len(metas), "ok": True},
        )

        print(f"[VectorDB] FAISS index built with {len(metas)} sections, dim={dim}")
        print(f"[VectorDB] Saved index, metadata, and embeddings to {self.embeddings_dir}")

    # ----------------------------------------------------------------------
    # Query
    # ----------------------------------------------------------------------

    @action
    async def query_texts(self, query_text: str, k: int = 5) -> List[str]:
        """
        Like query() but returns the actual abstract text content instead of
        file-name metadata.  Useful for agents that need to read the abstracts
        directly for RAG context.
        """
        results = await self.query(query_text, k=k)
        texts: List[str] = []
        for meta in results:
            fname = meta.get("file", "")
            if not fname:
                continue
            txt_path = self.abstracts_dir / fname
            try:
                if txt_path.exists():
                    content = txt_path.read_text(encoding="utf-8").strip()
                    if content:
                        texts.append(content)
            except Exception as e:
                self.logger.warning(
                    "query_texts_read_error",
                    extra={"file": fname, "error": repr(e)},
                )
        return texts

    @action
    async def query(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most similar abstracts.

        If the index is not ready, this will call ensure_vector_index().
        """
        log_action(
            self.logger,
            "query_start",
            {"query": query_text, "k": k},
            {},
        )
        print(f"🔎 [VectorDB] Query: {query_text}")

        # Lazy init
        if self.index is None or not self.metadata:
            self.logger.warning("query_on_empty_index", extra={})
            log_action(
                self.logger,
                "query_on_empty_index",
                {"query": query_text, "k": k},
                {},
            )
            await self.ensure_vector_index()

            if self.index is None or not self.metadata:
                print("❌ Index still not available after ensure_vector_index()")
                log_action(
                    self.logger,
                    "query_failed_no_index",
                    {"query": query_text, "k": k},
                    {"results": 0},
                )
                return []

        # Embed query
        query_vecs = await embed_texts([query_text])
        query_np = np.array(query_vecs, dtype=np.float32)
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)

        # Dimension check
        if query_np.shape[1] != self.index.d:
            self.logger.error(
                "dimension_mismatch",
                extra={"expected": self.index.d, "got": query_np.shape[1]},
            )
            log_action(
                self.logger,
                "query_dimension_mismatch",
                {
                    "query": query_text,
                    "k": k,
                    "expected_dim": self.index.d,
                    "got_dim": query_np.shape[1],
                },
                {"results": 0},
            )
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.index.d}, got {query_np.shape[1]}"
            )

        # Search
        D, I = self.index.search(query_np, k)
        results: List[Dict[str, Any]] = []
        for idx in I[0]:
            if 0 <= idx < len(self.metadata):
                results.append(self.metadata[idx])

        log_action(
            self.logger,
            "query_success",
            {"query": query_text, "k": k},
            {"results": len(results)},
        )
        print(f"✅ Query returned {len(results)} results.")
        return results
