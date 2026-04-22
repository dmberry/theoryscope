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
from operations.author_constellation import compute_author_constellation
from operations.concept_locator import compute_concept_locator
from operations.corpus_map import compute_corpus_map
from operations.debated_vs_computed import (
    DebatePayload,
    compute_debated_vs_computed,
)
from operations.eigendirections import compute_eigendirections
from operations.embedding_probe import (
    compute_embedding_probe,
    list_available_models,
)
from operations.flow import (
    compute_coarse_graining_trajectory,
    compute_fixed_points,
    compute_universality_classes,
)
from operations.forgetting import compute_forgetting_curve
from operations.operator_spectrum import (
    ConceptProbe,
    compute_operator_spectrum,
)
from operations.perturbation import compute_perturbation_test
from operations.phase_diagram import compute_phase_diagram
from operations.symmetry_breaking import compute_symmetry_breaking
from operations.temporal_flow import compute_temporal_flow
from operations.translation_probe import (
    compute_translation_probe,
    list_available_languages,
)

app = FastAPI(title="Theoryscope Backend", version="0.6.0")

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


class EmbeddingProbeRequest(BaseModel):
    corpus: CorpusSourcePayload = Field(default_factory=CorpusSourcePayload)
    probe_model_id: str
    n_components: int = 5
    n_loadings: int = 3


class PerturbationRequest(BaseModel):
    corpus: CorpusSourcePayload = Field(default_factory=CorpusSourcePayload)
    perturbation_text: str
    perturbation_label: str = "perturbation"
    n_components: int = 5
    n_loadings: int = 3


class ForgettingRequest(BaseModel):
    corpus: CorpusSourcePayload = Field(default_factory=CorpusSourcePayload)
    n_components: int = 5
    drop_fraction: float = 0.2
    n_iterations: int = 20
    seed: int = 0


class ConceptLocatorRequest(BaseModel):
    corpus: CorpusSourcePayload = Field(default_factory=CorpusSourcePayload)
    query_text: str
    query_label: str = ""
    n_nearest_docs: int = 8
    n_nearest_authors: int = 5
    n_components: int = 6


class AuthorConstellationRequest(BaseModel):
    corpus: CorpusSourcePayload = Field(default_factory=CorpusSourcePayload)
    min_documents: int = 1


class DebatePairPayload(BaseModel):
    label: str
    pole_a_text: str
    pole_b_text: str
    pole_a_label: str = ""
    pole_b_label: str = ""


class DebatedVsComputedRequest(BaseModel):
    corpus: CorpusSourcePayload = Field(default_factory=CorpusSourcePayload)
    debates: List[DebatePairPayload]
    n_components: int = 6


class ConceptProbePayload(BaseModel):
    label: str
    text: str


class OperatorSpectrumRequest(BaseModel):
    corpus: CorpusSourcePayload = Field(default_factory=CorpusSourcePayload)
    concepts: List[ConceptProbePayload]
    n_steps: int = 6
    seed: int = 0


class TemporalFlowRequest(BaseModel):
    corpus: CorpusSourcePayload = Field(default_factory=CorpusSourcePayload)
    n_steps: int = 6


class SymmetryBreakingRequest(BaseModel):
    corpus: CorpusSourcePayload = Field(default_factory=CorpusSourcePayload)
    splitter: str = "year_decade"
    threshold: Optional[int] = None
    n_components: int = 5


class PhaseDiagramRequest(BaseModel):
    corpus: CorpusSourcePayload = Field(default_factory=CorpusSourcePayload)
    n_steps: int = 6
    seed: int = 0


class TranslationProbeRequest(BaseModel):
    corpus: CorpusSourcePayload = Field(default_factory=CorpusSourcePayload)
    target_lang: str
    n_components: int = 5
    n_samples: int = 3


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
        "version": "0.6.0",
        "phase": "Critique complete (backends)",
        "corpora_available": ["philosophy-of-technology-v1", "zotero"],
        "operations_available": [
            "corpus_map",
            "eigendirections",
            "concept_locator",
            "author_constellation",
            "debated_vs_computed",
            "coarse_graining_trajectory",
            "fixed_points",
            "universality_classes",
            "operator_spectrum",
            "temporal_flow",
            "embedding_probe",
            "perturbation_test",
            "forgetting_curve",
            "symmetry_breaking",
            "phase_diagram",
            "translation_probe",
        ],
    }


@app.get("/translation-probe/languages")
async def translation_probe_languages() -> Dict[str, Any]:
    return {"languages": list_available_languages()}


@app.get("/embedding-probe/models")
async def embedding_probe_models() -> Dict[str, Any]:
    return {"models": list_available_models()}


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


@app.post("/concept-locator")
async def concept_locator(req: ConceptLocatorRequest) -> Dict[str, Any]:
    spec = req.corpus.to_spec()
    try:
        return await asyncio.to_thread(
            compute_concept_locator,
            spec,
            req.query_text,
            req.query_label,
            req.n_nearest_docs,
            req.n_nearest_authors,
            req.n_components,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/author-constellation")
async def author_constellation(req: AuthorConstellationRequest) -> Dict[str, Any]:
    spec = req.corpus.to_spec()
    try:
        return await asyncio.to_thread(
            compute_author_constellation,
            spec,
            req.min_documents,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debated-vs-computed")
async def debated_vs_computed(req: DebatedVsComputedRequest) -> Dict[str, Any]:
    spec = req.corpus.to_spec()
    try:
        debates = [
            DebatePayload(
                label=d.label,
                pole_a_text=d.pole_a_text,
                pole_b_text=d.pole_b_text,
                pole_a_label=d.pole_a_label,
                pole_b_label=d.pole_b_label,
            )
            for d in req.debates
        ]
        return await asyncio.to_thread(
            compute_debated_vs_computed,
            spec,
            debates,
            req.n_components,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/operator-spectrum")
async def operator_spectrum(req: OperatorSpectrumRequest) -> Dict[str, Any]:
    spec = req.corpus.to_spec()
    try:
        concepts = [
            ConceptProbe(label=c.label, text=c.text) for c in req.concepts
        ]
        return await asyncio.to_thread(
            compute_operator_spectrum,
            spec,
            concepts,
            req.n_steps,
            req.seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/temporal-flow")
async def temporal_flow(req: TemporalFlowRequest) -> Dict[str, Any]:
    spec = req.corpus.to_spec()
    try:
        return await asyncio.to_thread(
            compute_temporal_flow,
            spec,
            req.n_steps,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
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


@app.post("/embedding-probe")
async def embedding_probe(req: EmbeddingProbeRequest) -> Dict[str, Any]:
    spec = req.corpus.to_spec()
    try:
        return await asyncio.to_thread(
            compute_embedding_probe,
            spec,
            req.probe_model_id,
            req.n_components,
            req.n_loadings,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/perturbation-test")
async def perturbation_test(req: PerturbationRequest) -> Dict[str, Any]:
    spec = req.corpus.to_spec()
    try:
        return await asyncio.to_thread(
            compute_perturbation_test,
            spec,
            req.perturbation_text,
            req.perturbation_label,
            req.n_components,
            req.n_loadings,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/forgetting-curve")
async def forgetting_curve(req: ForgettingRequest) -> Dict[str, Any]:
    spec = req.corpus.to_spec()
    try:
        return await asyncio.to_thread(
            compute_forgetting_curve,
            spec,
            req.n_components,
            req.drop_fraction,
            req.n_iterations,
            req.seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/symmetry-breaking")
async def symmetry_breaking(req: SymmetryBreakingRequest) -> Dict[str, Any]:
    spec = req.corpus.to_spec()
    try:
        return await asyncio.to_thread(
            compute_symmetry_breaking,
            spec,
            req.splitter,
            req.threshold,
            req.n_components,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/phase-diagram")
async def phase_diagram(req: PhaseDiagramRequest) -> Dict[str, Any]:
    spec = req.corpus.to_spec()
    try:
        return await asyncio.to_thread(
            compute_phase_diagram,
            spec,
            req.n_steps,
            req.seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translation-probe")
async def translation_probe(req: TranslationProbeRequest) -> Dict[str, Any]:
    spec = req.corpus.to_spec()
    try:
        return await asyncio.to_thread(
            compute_translation_probe,
            spec,
            req.target_lang,
            req.n_components,
            req.n_samples,
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
