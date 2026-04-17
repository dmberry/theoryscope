"""
Renormalisation-group-style flow on a corpus.

The aggregative coarse-graining operator implemented here works by
progressively k-means clustering the corpus and, at each step,
replacing every document's current position with the centroid of its
assigned cluster. As the target number of clusters shrinks, fine-
grained distinctions are "integrated out" and only the long-wavelength
structure of the cloud remains.

The operator is fully in-process and its parameters are visible to the
caller: the schedule of cluster counts across steps, the current
centroid of each document at each step, and the final basin each
document falls into. No black-box clustering library decides the flow
on the critic's behalf.

Outputs are structured so that three Flow operations can share the
same computation:

  - Coarse-Graining Trajectory: the full list of step positions, for
    animation.
  - Fixed Point Finder: the terminal step's centroids and basin sizes.
  - Universality Class Finder: the set of documents that flow to the
    same terminal basin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from corpus.pipeline import CorpusSpec, build_provenance, ingest_and_embed


def default_schedule(n_docs: int, n_steps: int = 6) -> List[int]:
    """A sensible schedule of cluster counts from fine to coarse.

    The step 0 baseline is n_docs (each document is its own cluster).
    Subsequent steps geometrically reduce the cluster count down to
    a minimum of 2 so that the final step still shows at least two
    competing basins.
    """
    n_steps = max(2, int(n_steps))
    schedule = [n_docs]
    # Geometric interpolation from n_docs down to 2.
    lo = 2
    hi = max(lo + 1, n_docs)
    if hi <= lo:
        return schedule
    # Evenly spaced in log-space for n_steps - 1 subsequent steps.
    log_targets = np.linspace(np.log(hi), np.log(lo), n_steps)
    for t in log_targets[1:]:
        k = int(round(float(np.exp(t))))
        k = max(lo, min(k, n_docs - 1))
        if k != schedule[-1]:
            schedule.append(k)
    if schedule[-1] != lo:
        schedule.append(lo)
    return schedule


def _assign_kmeans(
    embeddings: np.ndarray,
    k: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (labels, centroids)."""
    k = max(1, min(k, embeddings.shape[0]))
    if k == embeddings.shape[0]:
        # Identity mapping — each document is its own centroid.
        labels = np.arange(embeddings.shape[0])
        return labels, embeddings.astype(np.float32)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(embeddings)
    return labels.astype(np.int32), km.cluster_centers_.astype(np.float32)


@dataclass
class FlowStep:
    """One coarse-graining pass."""

    step: int
    k: int                                # target cluster count at this step
    labels: np.ndarray                    # per-document cluster assignment (n_docs,)
    centroids: np.ndarray                 # (k, dim) — cluster centroids in embed space
    doc_positions: np.ndarray             # (n_docs, dim) — each doc's centroid
    doc_coords_2d: np.ndarray             # (n_docs, 2) — PCA-2D for visual animation


@dataclass
class FlowResult:
    steps: List[FlowStep]
    documents: List[dict]                 # [{id, author, year, title, tags}, ...]
    pca2d_axes: Tuple[List[float], List[float]] = field(default_factory=lambda: ([], []))
    schedule: List[int] = field(default_factory=list)


def _compute_pca_2d(embeddings: np.ndarray) -> tuple[PCA, np.ndarray]:
    n_comp = min(2, embeddings.shape[0], embeddings.shape[1])
    n_comp = max(1, n_comp)
    pca = PCA(n_components=n_comp)
    projected = pca.fit_transform(embeddings).astype(np.float32)
    if projected.shape[1] < 2:
        pad = np.zeros((projected.shape[0], 2 - projected.shape[1]), dtype=np.float32)
        projected = np.hstack([projected, pad])
    return pca, projected


def run_flow(
    spec: CorpusSpec,
    n_steps: int = 6,
    seed: int = 0,
) -> FlowResult:
    """Run the aggregative k-means coarse-graining flow."""
    bundle = ingest_and_embed(spec)
    embeddings = bundle.embeddings
    n_docs = embeddings.shape[0]

    if n_docs < 3:
        raise ValueError(
            f"Corpus has only {n_docs} document(s); coarse-graining "
            "needs at least 3 documents to produce a non-trivial flow."
        )

    schedule = default_schedule(n_docs, n_steps=n_steps)
    pca, _ = _compute_pca_2d(embeddings)

    steps: List[FlowStep] = []
    for i, k in enumerate(schedule):
        labels, centroids = _assign_kmeans(embeddings, k, seed=seed)
        # Every document is placed at its cluster centroid in the embedding space.
        doc_positions = centroids[labels]
        # Project those positions into the shared 2D basis fit on the raw cloud.
        doc_coords_2d = pca.transform(doc_positions).astype(np.float32)
        if doc_coords_2d.shape[1] < 2:
            pad = np.zeros(
                (doc_coords_2d.shape[0], 2 - doc_coords_2d.shape[1]),
                dtype=np.float32,
            )
            doc_coords_2d = np.hstack([doc_coords_2d, pad])
        steps.append(
            FlowStep(
                step=i,
                k=k,
                labels=labels,
                centroids=centroids,
                doc_positions=doc_positions,
                doc_coords_2d=doc_coords_2d,
            )
        )

    documents = [
        {
            "id": d.id,
            "author": d.author,
            "year": d.year,
            "title": d.title,
            "tags": list(d.tags),
        }
        for d in bundle.documents
    ]
    pca2d_axes = (
        [float(v) for v in pca.explained_variance_ratio_[:1]],
        [float(v) for v in pca.explained_variance_ratio_[1:2]]
        if pca.explained_variance_ratio_.shape[0] > 1
        else [0.0],
    )

    return FlowResult(
        steps=steps,
        documents=documents,
        pca2d_axes=pca2d_axes,
        schedule=schedule,
    )


# --- Operation dispatchers --------------------------------------------------

def compute_coarse_graining_trajectory(
    spec: CorpusSpec,
    n_steps: int = 6,
    seed: int = 0,
) -> Dict[str, Any]:
    """Full per-step trajectory suitable for frontend animation."""
    result = run_flow(spec, n_steps=n_steps, seed=seed)
    bundle = ingest_and_embed(spec)
    provenance = build_provenance(
        bundle=bundle,
        operation="coarse_graining_trajectory",
        operator_name="aggregative_kmeans",
        operator_params={"n_steps": n_steps, "seed": seed, "schedule": result.schedule},
    )
    return {
        "documents": result.documents,
        "schedule": result.schedule,
        "pca2d_variance": [
            result.pca2d_axes[0][0] if result.pca2d_axes[0] else 0.0,
            result.pca2d_axes[1][0] if result.pca2d_axes[1] else 0.0,
        ],
        "steps": [
            {
                "step": s.step,
                "k": s.k,
                "labels": [int(x) for x in s.labels],
                "doc_coords_2d": s.doc_coords_2d.tolist(),
            }
            for s in result.steps
        ],
        "provenance": provenance.to_dict(),
    }


def compute_fixed_points(
    spec: CorpusSpec,
    n_steps: int = 6,
    seed: int = 0,
) -> Dict[str, Any]:
    """Terminal basins and their populations."""
    result = run_flow(spec, n_steps=n_steps, seed=seed)
    bundle = ingest_and_embed(spec)
    provenance = build_provenance(
        bundle=bundle,
        operation="fixed_points",
        operator_name="aggregative_kmeans",
        operator_params={"n_steps": n_steps, "seed": seed, "schedule": result.schedule},
    )

    terminal = result.steps[-1]
    n_basins = int(terminal.centroids.shape[0])
    basins = []
    for b in range(n_basins):
        member_indices = np.where(terminal.labels == b)[0]
        members = [result.documents[int(i)] for i in member_indices]
        # Find exemplar: the member whose 2D position is closest to the basin centroid.
        centroid_2d = np.mean(terminal.doc_coords_2d[member_indices], axis=0)
        if len(member_indices) > 0:
            dists = np.linalg.norm(terminal.doc_coords_2d[member_indices] - centroid_2d, axis=1)
            exemplar = result.documents[int(member_indices[int(np.argmin(dists))])]
        else:
            exemplar = None
        basins.append(
            {
                "basin_index": b,
                "size": len(members),
                "exemplar": exemplar,
                "members": members,
                "centroid_2d": [float(v) for v in centroid_2d],
            }
        )
    basins.sort(key=lambda x: -x["size"])

    return {
        "documents": result.documents,
        "schedule": result.schedule,
        "n_basins": n_basins,
        "basins": basins,
        "terminal_coords_2d": terminal.doc_coords_2d.tolist(),
        "terminal_labels": [int(x) for x in terminal.labels],
        "provenance": provenance.to_dict(),
    }


def compute_universality_classes(
    spec: CorpusSpec,
    n_steps: int = 6,
    seed: int = 0,
) -> Dict[str, Any]:
    """Cluster documents by their terminal basin.

    The payload is structurally similar to `fixed_points` but the
    framing is different: here each basin is called a universality
    class and the caller reads it as the set of starting positions
    that flow to the same endpoint. Surprise is the key diagnostic:
    positions that look different at the surface but converge here,
    and positions that look similar at the surface but diverge.
    """
    result = run_flow(spec, n_steps=n_steps, seed=seed)
    bundle = ingest_and_embed(spec)
    provenance = build_provenance(
        bundle=bundle,
        operation="universality_classes",
        operator_name="aggregative_kmeans",
        operator_params={"n_steps": n_steps, "seed": seed, "schedule": result.schedule},
    )

    initial = result.steps[0]
    terminal = result.steps[-1]
    classes = []
    for cls_idx in range(int(terminal.centroids.shape[0])):
        member_idx = np.where(terminal.labels == cls_idx)[0]
        members = [result.documents[int(i)] for i in member_idx]
        # Surface similarity within the class (pairwise cos sim of initial
        # embeddings). Lower mean cos sim means the class assembles
        # surface-diverse positions under the flow — a universality-class
        # finding worth inspection.
        if len(member_idx) >= 2:
            emb = bundle.embeddings[member_idx]
            # Normalise rows for cosine.
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            unit = emb / norms
            sims = unit @ unit.T
            # Exclude the diagonal when averaging.
            mask = ~np.eye(sims.shape[0], dtype=bool)
            mean_cos = float(np.mean(sims[mask]))
        else:
            mean_cos = 1.0
        classes.append(
            {
                "class_index": cls_idx,
                "size": len(members),
                "members": members,
                "surface_mean_cosine": mean_cos,
            }
        )
    classes.sort(key=lambda x: -x["size"])

    return {
        "documents": result.documents,
        "schedule": result.schedule,
        "n_classes": int(terminal.centroids.shape[0]),
        "classes": classes,
        "initial_coords_2d": initial.doc_coords_2d.tolist(),
        "terminal_labels": [int(x) for x in terminal.labels],
        "provenance": provenance.to_dict(),
    }
