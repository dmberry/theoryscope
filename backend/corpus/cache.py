"""
On-disk cache for embedded corpora.

Embedding is expensive; re-embedding hundreds of documents on every
operation is infeasible. An ingested corpus (documents + embeddings)
is therefore keyed by a deterministic hash of (document IDs + chunking)
and written to `.theoryscope-cache/corpora/{corpus_hash}.npz` + `.json`.

The cache is scoped per repository root. It is not committed (see the
root .gitignore).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from .loader import Document
from .provenance import ChunkingSpec, CorpusSource, EmbeddingSpec


CACHE_ROOT = Path(os.environ.get("THEORYSCOPE_CACHE_DIR", ".theoryscope-cache"))
CORPORA_DIR = CACHE_ROOT / "corpora"
TRANSLATIONS_DIR = CACHE_ROOT / "translations"


def ensure_cache_dirs() -> None:
    CORPORA_DIR.mkdir(parents=True, exist_ok=True)
    TRANSLATIONS_DIR.mkdir(parents=True, exist_ok=True)


def corpus_cache_key(
    document_ids: list[str],
    chunking: ChunkingSpec,
    embedding_model_id: str,
) -> str:
    """Stable hash over the corpus definition + embedding model."""
    payload = {
        "document_ids": sorted(document_ids),
        "chunking": asdict(chunking),
        "embedding_model_id": embedding_model_id,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def _paths(key: str) -> tuple[Path, Path]:
    return CORPORA_DIR / f"{key}.npz", CORPORA_DIR / f"{key}.json"


def load(key: str) -> Optional[tuple[list[Document], np.ndarray, CorpusSource, EmbeddingSpec, ChunkingSpec]]:
    """Try to load a previously cached corpus. Returns None on miss."""
    npz_path, json_path = _paths(key)
    if not npz_path.exists() or not json_path.exists():
        return None
    try:
        with np.load(npz_path) as npz:
            embeddings = np.asarray(npz["embeddings"], dtype=np.float32)
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        documents = [Document(**d) for d in meta["documents"]]
        source = CorpusSource(**meta["corpus_source"])
        embedding = EmbeddingSpec(**meta["embedding"])
        chunking = ChunkingSpec(**meta["chunking"])
        return documents, embeddings, source, embedding, chunking
    except Exception:
        # Corrupt cache entry — silently miss and re-embed.
        return None


def save(
    key: str,
    documents: list[Document],
    embeddings: np.ndarray,
    corpus_source: CorpusSource,
    embedding_spec: EmbeddingSpec,
    chunking: ChunkingSpec,
) -> None:
    ensure_cache_dirs()
    npz_path, json_path = _paths(key)
    np.savez_compressed(npz_path, embeddings=embeddings.astype(np.float32))
    meta = {
        "documents": [asdict(d) for d in documents],
        "corpus_source": asdict(corpus_source),
        "embedding": asdict(embedding_spec),
        "chunking": asdict(chunking),
    }
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
