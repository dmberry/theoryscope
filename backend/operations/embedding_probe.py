"""
Embedding Dependence Probe.

Recompute eigendirections under a second open-weight embedding model
and report the agreement with the baseline. This is the methodological
keystone of the Critique tab: every Theoryscope finding is a finding
about the corpus-as-measured-by-a-particular-embedding-model.
Findings that survive re-embedding are stronger claims about the field
than findings that do not.

The probe compares bases via document projections rather than via the
component vectors directly, because different embedding models may
produce embeddings of different dimensions.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np
from sklearn.decomposition import PCA

from corpus.pipeline import CorpusSpec, build_provenance, ingest_and_embed
from geometry.eigen_align import align_via_doc_projection


# Shortlist of comparably-sized open-weight embedding models. Each entry
# is a (hf_repo_id, human_label, dimension) triple. Kept short because
# every model triggers a ~50MB+ download on first use.
AVAILABLE_MODELS: List[dict] = [
    {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "label": "MiniLM L6 v2 · 384d",
        "dimension": 384,
    },
    {
        "model_id": "sentence-transformers/all-mpnet-base-v2",
        "label": "MPNet base v2 · 768d",
        "dimension": 768,
    },
    {
        "model_id": "BAAI/bge-small-en-v1.5",
        "label": "BGE small en 1.5 · 384d",
        "dimension": 384,
    },
]


def _format_loadings(
    docs: list, coords: np.ndarray, n_loadings: int
) -> List[List[Dict[str, Any]]]:
    """Return per-component ranked [+pos, -neg] loadings for the top components."""
    n_components = coords.shape[1]
    n_loadings = max(1, min(n_loadings, coords.shape[0]))
    out: List[List[Dict[str, Any]]] = []
    for pc in range(n_components):
        scores = coords[:, pc]
        pos = np.argsort(-scores)[:n_loadings]
        neg = np.argsort(scores)[:n_loadings]
        out.append(
            [
                {
                    "id": docs[i].id,
                    "author": docs[i].author,
                    "year": docs[i].year,
                    "title": docs[i].title,
                    "score": float(scores[i]),
                    "pole": "positive",
                }
                for i in pos
            ]
            + [
                {
                    "id": docs[i].id,
                    "author": docs[i].author,
                    "year": docs[i].year,
                    "title": docs[i].title,
                    "score": float(scores[i]),
                    "pole": "negative",
                }
                for i in neg
            ]
        )
    return out


def compute_embedding_probe(
    spec: CorpusSpec,
    probe_model_id: str,
    n_components: int = 5,
    n_loadings: int = 3,
) -> Dict[str, Any]:
    """
    Run PCA on the corpus under two different embedding models and
    compare the resulting eigenbases.

    Parameters
    ----------
    spec : CorpusSpec
        Corpus spec; ``spec.model_id`` is used as the baseline model.
    probe_model_id : str
        HuggingFace repo id for the second embedding model.
    n_components : int
        Number of components to compare per basis.
    n_loadings : int
        How many documents to show per pole per component.
    """
    baseline_spec = spec
    probe_spec = CorpusSpec(
        hardcoded_name=spec.hardcoded_name,
        zotero=spec.zotero,
        model_id=probe_model_id,
    )

    baseline = ingest_and_embed(baseline_spec)
    probe = ingest_and_embed(probe_spec)

    if baseline.embeddings.shape[0] != probe.embeddings.shape[0]:
        raise ValueError(
            "Baseline and probe embeddings have different row counts; "
            "something corrupted the corpus cache. Try re-embedding."
        )

    n_docs = baseline.embeddings.shape[0]
    n_components = max(2, min(int(n_components), min(n_docs, 20)))

    pca_baseline = PCA(n_components=n_components)
    coords_baseline = pca_baseline.fit_transform(baseline.embeddings).astype(
        np.float32
    )

    pca_probe = PCA(n_components=n_components)
    coords_probe = pca_probe.fit_transform(probe.embeddings).astype(np.float32)

    alignment = align_via_doc_projection(coords_baseline, coords_probe)

    baseline_loadings = _format_loadings(
        baseline.documents, coords_baseline, n_loadings
    )
    probe_loadings = _format_loadings(probe.documents, coords_probe, n_loadings)

    provenance = build_provenance(
        bundle=baseline,
        operation="embedding_probe",
        operator_name="pca_eigenbasis_alignment",
        operator_params={
            "baseline_model_id": baseline.embedding_spec.model_id,
            "probe_model_id": probe.embedding_spec.model_id,
            "n_components": n_components,
        },
    )

    return {
        "documents": [
            {
                "id": d.id,
                "author": d.author,
                "year": d.year,
                "title": d.title,
                "tags": list(d.tags),
            }
            for d in baseline.documents
        ],
        "baseline": {
            "model_id": baseline.embedding_spec.model_id,
            "dimension": baseline.embedding_spec.dimension,
            "variance_explained": [
                float(v) for v in pca_baseline.explained_variance_ratio_
            ],
            "loadings": baseline_loadings,
        },
        "probe": {
            "model_id": probe.embedding_spec.model_id,
            "dimension": probe.embedding_spec.dimension,
            "variance_explained": [
                float(v) for v in pca_probe.explained_variance_ratio_
            ],
            "loadings": probe_loadings,
        },
        "alignment": {
            "matches": [asdict(m) for m in alignment.matches],
            "per_component": alignment.per_component,
            "stability": alignment.stability,
        },
        "provenance": provenance.to_dict(),
    }


def list_available_models() -> List[dict]:
    return list(AVAILABLE_MODELS)
