"""
Perturbation Test.

Add one (or a few) out-of-field text(s) to the corpus, recompute the
eigenbasis, and report how far each principal component rotated.

The critical use of this operation is twofold:
  1. A diagnostic for the fragility of a specific eigendirection: if
     adding one paper moves PC1 substantially, PC1 was not really a
     structural feature of the field.
  2. A way to see which axis was most sensitive to the new text, which
     can be read as "what axis of variation did this out-of-field
     perturbation activate?"

The perturbation is provided by the caller as free text and is
embedded using the same baseline embedding model.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np
from sklearn.decomposition import PCA

from corpus.embed import embed_documents as _embed_documents
from corpus.loader import Document
from corpus.pipeline import CorpusSpec, build_provenance, ingest_and_embed
from geometry.eigen_align import align_eigenbases


def compute_perturbation_test(
    spec: CorpusSpec,
    perturbation_text: str,
    perturbation_label: str = "perturbation",
    n_components: int = 5,
    n_loadings: int = 3,
) -> Dict[str, Any]:
    """
    Baseline: ingest_and_embed(spec), PCA.
    Perturbed: append the perturbation to the corpus, re-embed that one
    document, stack onto baseline embeddings, re-run PCA on the full
    set. Align component-wise via the shared ``align_eigenbases`` (both
    bases live in the same embedding dim).
    """
    if not perturbation_text.strip():
        raise ValueError("perturbation_text is empty.")

    baseline = ingest_and_embed(spec)
    n_docs = baseline.embeddings.shape[0]
    n_components = max(2, min(int(n_components), min(n_docs, 20)))

    # Construct a one-item document for the perturbation and embed it
    # with the same model.
    probe_doc = Document(
        id=f"__perturbation__:{perturbation_label[:48]}",
        author="(perturbation)",
        year=0,
        title=perturbation_label or "perturbation",
        text=perturbation_text,
        tags=["__perturbation__"],
    )
    perturb_embedding, _ = _embed_documents(
        [probe_doc], model_id=baseline.embedding_spec.model_id
    )

    # Baseline PCA.
    pca_baseline = PCA(n_components=n_components)
    coords_baseline = pca_baseline.fit_transform(baseline.embeddings).astype(
        np.float32
    )

    # Perturbed PCA (on baseline rows + the one new row).
    combined = np.vstack([baseline.embeddings, perturb_embedding]).astype(
        np.float32
    )
    pca_perturbed = PCA(n_components=n_components)
    coords_perturbed_all = pca_perturbed.fit_transform(combined).astype(
        np.float32
    )
    coords_perturbed_docs = coords_perturbed_all[:n_docs]
    coords_perturbed_probe = coords_perturbed_all[n_docs]

    # Align the two principal-component matrices directly — same
    # embedding dim, so a cosine-similarity alignment is meaningful.
    alignment = align_eigenbases(
        pca_baseline.components_, pca_perturbed.components_
    )

    # Per-component rotation = 1 - |cos|; ranked descending.
    per_pc_rotation = [
        1.0 - score for score in alignment.per_component
    ]
    ranked = sorted(
        range(len(per_pc_rotation)), key=lambda i: -per_pc_rotation[i]
    )

    # Where did the probe itself land on each component?
    probe_projection = [float(v) for v in coords_perturbed_probe]

    provenance = build_provenance(
        bundle=baseline,
        operation="perturbation_test",
        operator_name="pca_eigenbasis_perturb",
        operator_params={
            "perturbation_label": perturbation_label,
            "perturbation_char_length": len(perturbation_text),
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
            "variance_explained": [
                float(v) for v in pca_baseline.explained_variance_ratio_
            ],
        },
        "perturbed": {
            "variance_explained": [
                float(v) for v in pca_perturbed.explained_variance_ratio_
            ],
        },
        "alignment": {
            "matches": [asdict(m) for m in alignment.matches],
            "per_component": alignment.per_component,
            "stability": alignment.stability,
            "per_component_rotation": per_pc_rotation,
            "ranked_by_rotation": ranked,
        },
        "probe": {
            "label": perturbation_label,
            "char_length": len(perturbation_text),
            "projection_on_perturbed_basis": probe_projection,
        },
        "provenance": provenance.to_dict(),
    }
