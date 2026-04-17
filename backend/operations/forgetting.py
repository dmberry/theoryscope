"""
Forgetting Curve.

Repeatedly remove a random subset of the corpus, recompute the
eigenbasis on the survivors, and align it to the baseline eigenbasis.
The distribution of per-component alignment scores across bootstrap
iterations tells you which eigendirections are robust under corpus
resampling and which are fragile.

This is a formal bootstrap over the corpus. The finding is not "the
field really is structured along PC1" but "PC1 is or is not stable
when one fifth of the documents are removed."
"""

from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.decomposition import PCA

from corpus.pipeline import CorpusSpec, build_provenance, ingest_and_embed
from geometry.eigen_align import align_eigenbases


def _bootstrap_pca(
    baseline_embeddings: np.ndarray,
    n_components: int,
    drop_fraction: float,
    n_iterations: int,
    seed: int,
) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Run ``n_iterations`` bootstrap resamples. Each iteration removes a
    random ``drop_fraction`` of rows, fits a PCA on the survivors, and
    aligns the resulting basis to the baseline basis.

    Returns
    -------
    per_iteration_per_pc : (n_iterations, n_components) float array
        |cos| between baseline PC_i and its best match in the bootstrap
        basis, for every iteration.
    per_iteration_stability : list of floats
        Mean |cos| across components for each iteration.
    """
    rng = random.Random(seed)
    n_docs = baseline_embeddings.shape[0]
    n_keep = max(n_components + 1, int(round(n_docs * (1.0 - drop_fraction))))
    n_keep = min(n_keep, n_docs)

    pca_baseline = PCA(n_components=n_components)
    pca_baseline.fit(baseline_embeddings)

    per_iteration = np.zeros((n_iterations, n_components), dtype=np.float32)
    per_iteration_stability: List[float] = []

    for it in range(n_iterations):
        idx = list(range(n_docs))
        rng.shuffle(idx)
        keep = sorted(idx[:n_keep])
        sample = baseline_embeddings[keep]

        pca_sample = PCA(n_components=n_components)
        pca_sample.fit(sample)

        alignment = align_eigenbases(
            pca_baseline.components_, pca_sample.components_
        )
        per_iteration[it, :] = np.array(
            alignment.per_component, dtype=np.float32
        )
        per_iteration_stability.append(alignment.stability)

    return per_iteration, per_iteration_stability


def compute_forgetting_curve(
    spec: CorpusSpec,
    n_components: int = 5,
    drop_fraction: float = 0.2,
    n_iterations: int = 20,
    seed: int = 0,
) -> Dict[str, Any]:
    baseline = ingest_and_embed(spec)
    n_docs = baseline.embeddings.shape[0]
    if n_docs < 4:
        raise ValueError(
            f"Corpus has only {n_docs} documents; the forgetting curve "
            "needs at least 4."
        )
    n_components = max(2, min(int(n_components), min(n_docs - 1, 20)))
    drop_fraction = max(0.05, min(float(drop_fraction), 0.8))
    n_iterations = max(5, min(int(n_iterations), 200))

    per_iter, per_iter_stability = _bootstrap_pca(
        baseline.embeddings,
        n_components=n_components,
        drop_fraction=drop_fraction,
        n_iterations=n_iterations,
        seed=seed,
    )

    per_pc_mean = per_iter.mean(axis=0).tolist()
    per_pc_std = per_iter.std(axis=0, ddof=1).tolist() if n_iterations > 1 else [0.0] * n_components
    per_pc_p25 = np.percentile(per_iter, 25, axis=0).tolist()
    per_pc_p75 = np.percentile(per_iter, 75, axis=0).tolist()
    per_pc_min = per_iter.min(axis=0).tolist()

    provenance = build_provenance(
        bundle=baseline,
        operation="forgetting_curve",
        operator_name="pca_bootstrap",
        operator_params={
            "n_components": n_components,
            "drop_fraction": drop_fraction,
            "n_iterations": n_iterations,
            "seed": seed,
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
        "n_components": n_components,
        "n_iterations": n_iterations,
        "drop_fraction": drop_fraction,
        "per_pc_mean": [float(v) for v in per_pc_mean],
        "per_pc_std": [float(v) for v in per_pc_std],
        "per_pc_p25": [float(v) for v in per_pc_p25],
        "per_pc_p75": [float(v) for v in per_pc_p75],
        "per_pc_min": [float(v) for v in per_pc_min],
        "per_iteration": per_iter.tolist(),
        "per_iteration_stability": [float(v) for v in per_iter_stability],
        "overall_stability": float(np.mean(per_iter_stability)) if per_iter_stability else 0.0,
        "provenance": provenance.to_dict(),
    }
