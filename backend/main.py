"""
Theoryscope backend — FastAPI server for corpus-level theory-space inspection.

Phase 0 + 1 + 1.5 exposes:
    GET  /status                 liveness + phase
    POST /corpus-map             ingest + embed + PCA-3D
    POST /eigendirections        ingest + embed + top-N PCA with loadings
    POST /zotero/collections     list collections in a Zotero library
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from dataclasses import asdict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from corpus.pipeline import CorpusSpec, ZoteroSourceSpec
from operations.corpus_map import compute_corpus_map
from operations.eigendirections import compute_eigendirections
from operations.flow import (
    compute_coarse_graining_trajectory,
    compute_fixed_points,
    compute_universality_classes,
)

app = FastAPI(title="Theoryscope Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- request models ---------------------------------------------------------

class CorpusSourcePayload(BaseModel):
    """Carries the corpus spec in the request body.

    Exactly one of `hardcoded_name` or `zotero` is expected to be set.
    """

    hardcoded_name: Optional[str] = Field(default=None)
    zotero: Optional["ZoteroSourcePayload"] = Field(default=None)

    def to_spec(self) -> CorpusSpec:
        if self.zotero is not None:
            return CorpusSpec(
                zotero=ZoteroSourceSpec(
                    library_id=self.zotero.library_id,
                    library_type=self.zotero.library_type,
                    api_key=self.zotero.api_key,
                    collection_key=self.zotero.collection_key,
                    collection_name=self.zotero.collection_name or "",
                )
            )
        if self.hardcoded_name:
            return CorpusSpec(hardcoded_name=self.hardcoded_name)
        # Default to Phase 0 corpus if nothing specified (convenience for tests).
        return CorpusSpec(hardcoded_name="philosophy-of-technology-v1")


class ZoteroSourcePayload(BaseModel):
    library_id: str
    library_type: str                   # "user" | "group"
    api_key: str
    collection_key: str
    collection_name: Optional[str] = ""


# Rebuild forward refs now that ZoteroSourcePayload is defined
CorpusSourcePayload.model_rebuild()


class CorpusMapRequest(BaseModel):
    corpus: CorpusSourcePayload = Field(default_factory=CorpusSourcePayload)
    # Back-compat: Phase 0 clients may still send `corpus_name`
    corpus_name: Optional[str] = None


class EigendirectionsRequest(BaseModel):
    corpus: CorpusSourcePayload = Field(default_factory=CorpusSourcePayload)
    corpus_name: Optional[str] = None    # back-compat
    n_components: int = 6
    n_loadings: int = 5


class FlowRequest(BaseModel):
    """Shared request shape for the three Flow operations."""
    corpus: CorpusSourcePayload = Field(default_factory=CorpusSourcePayload)
    n_steps: int = 6
    seed: int = 0


class ZoteroCollectionsRequest(BaseModel):
    library_id: str
    library_type: str
    api_key: str


# --- helpers ---------------------------------------------------------------

def _resolve_spec(
    corpus: CorpusSourcePayload,
    legacy_name: Optional[str],
) -> CorpusSpec:
    """Honour the legacy `corpus_name` field when the new `corpus` is empty."""
    if legacy_name and not corpus.hardcoded_name and corpus.zotero is None:
        return CorpusSpec(hardcoded_name=legacy_name)
    return corpus.to_spec()


# --- routes ----------------------------------------------------------------

@app.get("/status")
async def status() -> Dict[str, Any]:
    return {
        "status": "ok",
        "tool": "theoryscope",
        "version": "0.2.0",
        "phase": "2",
        "corpora_available": ["philosophy-of-technology-v1", "zotero"],
        "operations_available": [
            "corpus_map",
            "eigendirections",
            "coarse_graining_trajectory",
            "fixed_points",
            "universality_classes",
        ],
    }


@app.post("/corpus-map")
async def corpus_map(req: CorpusMapRequest) -> Dict[str, Any]:
    spec = _resolve_spec(req.corpus, req.corpus_name)
    try:
        return await asyncio.to_thread(compute_corpus_map, spec)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:  # noqa: BLE001 — surface real error to the UI
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/eigendirections")
async def eigendirections(req: EigendirectionsRequest) -> Dict[str, Any]:
    spec = _resolve_spec(req.corpus, req.corpus_name)
    try:
        return await asyncio.to_thread(
            compute_eigendirections,
            spec,
            req.n_components,
            req.n_loadings,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/coarse-graining-trajectory")
async def coarse_graining_trajectory(req: FlowRequest) -> Dict[str, Any]:
    spec = req.corpus.to_spec()
    try:
        return await asyncio.to_thread(
            compute_coarse_graining_trajectory,
            spec,
            req.n_steps,
            req.seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fixed-points")
async def fixed_points(req: FlowRequest) -> Dict[str, Any]:
    spec = req.corpus.to_spec()
    try:
        return await asyncio.to_thread(
            compute_fixed_points,
            spec,
            req.n_steps,
            req.seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/universality-classes")
async def universality_classes(req: FlowRequest) -> Dict[str, Any]:
    spec = req.corpus.to_spec()
    try:
        return await asyncio.to_thread(
            compute_universality_classes,
            spec,
            req.n_steps,
            req.seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/zotero/collections")
async def zotero_collections(req: ZoteroCollectionsRequest) -> Dict[str, Any]:
    from corpus.zotero import list_collections

    try:
        items = await asyncio.to_thread(
            list_collections,
            req.library_id,
            req.library_type,
            req.api_key,
        )
        return {"collections": [asdict(c) for c in items]}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Zotero API error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
