"""
Symmetry Breaking Map.

The critic names a parameter that might split the field (year-based,
tag-based, or author-based), and the tool reports how far the cloud
actually separates along that parameter. The name comes from
condensed-matter physics: when a symmetric system picks out a
preferred direction, the symmetry has broken. Here the question is
whether the proposed splitter aligns with a genuine symmetry
breaking in the corpus's geometry, or whether the field is
symmetric with respect to it.

Metrics returned per splitter:
  - Silhouette score: how tightly each group is separated from the
    others (-1 to 1, higher = cleaner split).
  - F-statistic: ratio of between-group to within-group variance on
    the top-K PCs, a classical ANOVA-flavour readout.
  - Per-PC alignment: for each of the top PCs, the cosine between
    the between-group direction (centroid difference for 2 groups;
    leading discriminant direction for >2 groups) and that PC.

The UI then colours the PCA-2D scatter by group so the reader can
see the split (or its absence) alongside the numbers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from corpus.pipeline import CorpusSpec, build_provenance, ingest_and_embed


def _bucket_by_decade(year: int) -> str:
    if year <= 0:
        return "(unknown)"
    decade = (year // 10) * 10
    return f"{decade}s"


def _bucket_by_year_threshold(year: int, threshold: int) -> str:
    if year <= 0:
        return "(unknown)"
    return f"≥{threshold}" if year >= threshold else f"<{threshold}"


def _assign_labels(documents, splitter: str, threshold: Optional[int]) -> List[str]:
    if splitter == "year_decade":
        return [_bucket_by_decade(d.year) for d in documents]
    if splitter == "year_threshold":
        t = threshold if threshold is not None else 1990
        return [_bucket_by_year_threshold(d.year, t) for d in documents]
    if splitter == "first_tag":
        return [d.tags[0] if d.tags else "(untagged)" for d in documents]
    if splitter == "author":
        return [d.author for d in documents]
    raise ValueError(
        f"Unknown splitter '{splitter}'. Supported: year_decade, year_threshold, first_tag, author."
    )


def _between_group_direction(
    coords: np.ndarray, labels: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Return a unit direction that best separates groups in ``coords``,
    using the leading eigenvector of the between-group covariance matrix.
    Falls back to a simple centroid difference for two groups."""
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        raise ValueError("Need at least two groups for symmetry breaking.")
    overall_mean = coords.mean(axis=0)
    between_cov = np.zeros((coords.shape[1], coords.shape[1]), dtype=np.float64)
    for g in unique_labels:
        mask = np.array([lbl == g for lbl in labels])
        n = int(mask.sum())
        if n == 0:
            continue
        mean_g = coords[mask].mean(axis=0)
        diff = (mean_g - overall_mean).reshape(-1, 1)
        between_cov += n * (diff @ diff.T)

    if len(unique_labels) == 2:
        # Two-group centroid difference is the natural direction.
        m0 = coords[[labels[i] == unique_labels[0] for i in range(len(labels))]].mean(axis=0)
        m1 = coords[[labels[i] == unique_labels[1] for i in range(len(labels))]].mean(axis=0)
        direction = m1 - m0
    else:
        eigvals, eigvecs = np.linalg.eigh(between_cov)
        direction = eigvecs[:, -1]

    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return np.zeros_like(direction), unique_labels
    return (direction / norm).astype(np.float32), unique_labels


def compute_symmetry_breaking(
    spec: CorpusSpec,
    splitter: str,
    threshold: Optional[int] = None,
    n_components: int = 5,
) -> Dict[str, Any]:
    bundle = ingest_and_embed(spec)
    embeddings = bundle.embeddings
    n_docs = embeddings.shape[0]
    if n_docs < 4:
        raise ValueError(
            f"Corpus has only {n_docs} document(s); Symmetry Breaking needs at least 4."
        )
    n_components = max(2, min(int(n_components), min(n_docs, 20)))

    labels = _assign_labels(bundle.documents, splitter, threshold)
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        raise ValueError(
            f"Splitter '{splitter}' produced only one group — nothing to split."
        )

    # Fit PCA to get eigenbasis and 2D projection.
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(embeddings).astype(np.float32)

    # PCA-2D projection (for the scatter).
    pca2d = PCA(n_components=min(2, embeddings.shape[1]))
    coords_2d = pca2d.fit_transform(embeddings).astype(np.float32)
    if coords_2d.shape[1] < 2:
        pad = np.zeros(
            (coords_2d.shape[0], 2 - coords_2d.shape[1]), dtype=np.float32
        )
        coords_2d = np.hstack([coords_2d, pad])

    # Silhouette on the FULL embedding matrix — meaningful only when
    # groups are reasonably populated.
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    int_labels = np.array([label_to_int[l] for l in labels], dtype=np.int32)
    if len(unique_labels) >= 2 and int(int_labels.size) > len(unique_labels):
        try:
            sil = float(
                silhouette_score(embeddings, int_labels, metric="cosine")
            )
        except Exception:
            sil = float("nan")
    else:
        sil = float("nan")

    # F-statistic on the top-K PC scores: ratio of between-group to
    # within-group variance summed across PCs.
    between_var = 0.0
    within_var = 0.0
    overall_mean = projected.mean(axis=0)
    for g in unique_labels:
        mask = int_labels == label_to_int[g]
        n_g = int(mask.sum())
        if n_g == 0:
            continue
        mean_g = projected[mask].mean(axis=0)
        between_var += float(n_g * np.sum((mean_g - overall_mean) ** 2))
        within_var += float(np.sum((projected[mask] - mean_g) ** 2))
    df_between = max(1, len(unique_labels) - 1)
    df_within = max(1, n_docs - len(unique_labels))
    f_statistic = (
        (between_var / df_between) / (within_var / df_within)
        if within_var > 1e-12
        else float("inf")
    )

    # Per-PC alignment with the between-group direction.
    direction, _ = _between_group_direction(embeddings, labels)
    # Cosine between direction and each principal component.
    comp_norms = np.linalg.norm(pca.components_, axis=1) + 1e-12
    cosines = pca.components_ @ direction / comp_norms
    per_pc = [
        {
            "pc": i,
            "variance_explained": float(pca.explained_variance_ratio_[i]),
            "signed_cosine": float(cosines[i]),
            "abs_cosine": float(abs(cosines[i])),
        }
        for i in range(n_components)
    ]
    best_pc = int(np.argmax([p["abs_cosine"] for p in per_pc]))

    # Per-group centroid on the 2D basis.
    groups_payload = []
    for g in unique_labels:
        mask = int_labels == label_to_int[g]
        n_g = int(mask.sum())
        centroid_2d = coords_2d[mask].mean(axis=0) if n_g > 0 else np.zeros(2)
        groups_payload.append(
            {
                "label": g,
                "n_documents": n_g,
                "centroid_2d": [float(centroid_2d[0]), float(centroid_2d[1])],
            }
        )

    documents_payload = [
        {
            "id": d.id,
            "author": d.author,
            "year": d.year,
            "title": d.title,
            "tags": list(d.tags),
            "group": labels[i],
            "coords_2d": [float(coords_2d[i, 0]), float(coords_2d[i, 1])],
        }
        for i, d in enumerate(bundle.documents)
    ]

    pca2d_variance = [float(v) for v in pca2d.explained_variance_ratio_[:2]]
    while len(pca2d_variance) < 2:
        pca2d_variance.append(0.0)

    provenance = build_provenance(
        bundle=bundle,
        operation="symmetry_breaking",
        operator_name="pca_between_group",
        operator_params={
            "splitter": splitter,
            "threshold": threshold,
            "n_components": n_components,
        },
    )

    return {
        "splitter": splitter,
        "threshold": threshold,
        "groups": groups_payload,
        "documents": documents_payload,
        "pca2d_variance": pca2d_variance,
        "silhouette_score": sil,
        "f_statistic": float(f_statistic),
        "between_variance": float(between_var),
        "within_variance": float(within_var),
        "best_pc": best_pc,
        "per_component": per_pc,
        "provenance": provenance.to_dict(),
    }
