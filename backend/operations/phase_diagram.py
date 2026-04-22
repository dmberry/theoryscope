"""
Phase Diagram rendering.

A single readable image that combines the entire coarse-graining flow:
initial document positions, flow arrows from initial to terminal
positions, fixed points (terminal basin centroids) as markers, and
basin membership as colour.

Phase diagrams in physics are a compressed view of a flow's long-time
behaviour. Here we compress the per-step trajectory into a
"where did each document start, where did it end up, which basin
claimed it" readout, so the critic can see the flow's topology in a
single view rather than scrubbing through the animated trajectory.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.decomposition import PCA

from corpus.pipeline import CorpusSpec, build_provenance, ingest_and_embed
from operations.flow import run_flow


def compute_phase_diagram(
    spec: CorpusSpec,
    n_steps: int = 6,
    seed: int = 0,
) -> Dict[str, Any]:
    bundle = ingest_and_embed(spec)
    flow = run_flow(spec, n_steps=n_steps, seed=seed)
    if not flow.steps:
        raise ValueError("Flow produced no steps; cannot render a phase diagram.")

    initial = flow.steps[0]
    terminal = flow.steps[-1]
    n_basins = int(terminal.centroids.shape[0])

    # PCA-2D shared basis — fit on the raw embeddings (same as Corpus Map).
    n_comp = min(2, bundle.embeddings.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(bundle.embeddings)

    def project(matrix: np.ndarray) -> np.ndarray:
        proj = pca.transform(matrix).astype(np.float32)
        if proj.shape[1] < 2:
            pad = np.zeros((proj.shape[0], 2 - proj.shape[1]), dtype=np.float32)
            proj = np.hstack([proj, pad])
        return proj

    initial_2d = project(bundle.embeddings)
    terminal_2d = project(terminal.centroids[terminal.labels])
    centroids_2d = project(terminal.centroids)

    # Per-document payload: starting position, terminal position, and
    # terminal basin assignment (colour index for the frontend).
    documents_payload: List[Dict[str, Any]] = []
    for i, doc in enumerate(bundle.documents):
        documents_payload.append(
            {
                "id": doc.id,
                "author": doc.author,
                "year": doc.year,
                "title": doc.title,
                "initial_2d": [float(initial_2d[i, 0]), float(initial_2d[i, 1])],
                "terminal_2d": [float(terminal_2d[i, 0]), float(terminal_2d[i, 1])],
                "basin": int(terminal.labels[i]),
            }
        )

    # Fixed points (basin centroids) with size and convex-hull points
    # of their members for basin shading on the frontend.
    basins_payload: List[Dict[str, Any]] = []
    for b in range(n_basins):
        member_indices = np.where(terminal.labels == b)[0]
        n_members = int(member_indices.size)
        if n_members == 0:
            continue
        member_initial_2d = initial_2d[member_indices]
        hull = _convex_hull_2d(member_initial_2d)
        basins_payload.append(
            {
                "basin_index": b,
                "n_members": n_members,
                "fixed_point_2d": [
                    float(centroids_2d[b, 0]),
                    float(centroids_2d[b, 1]),
                ],
                "hull_2d": [[float(x), float(y)] for x, y in hull],
                "members": [int(i) for i in member_indices.tolist()],
            }
        )

    variance_explained = [float(v) for v in pca.explained_variance_ratio_[:2]]
    while len(variance_explained) < 2:
        variance_explained.append(0.0)

    provenance = build_provenance(
        bundle=bundle,
        operation="phase_diagram",
        operator_name="aggregative_kmeans_phase",
        operator_params={
            "n_steps": n_steps,
            "seed": seed,
            "schedule": flow.schedule,
        },
    )

    return {
        "documents": documents_payload,
        "basins": basins_payload,
        "n_basins": len(basins_payload),
        "schedule": flow.schedule,
        "pca2d_variance": variance_explained,
        "provenance": provenance.to_dict(),
    }


def _convex_hull_2d(points: np.ndarray) -> List[List[float]]:
    """Graham-scan-free, numpy-only convex hull for small point sets.

    Returns the hull vertices in counter-clockwise order. If there are
    fewer than 3 points, returns them unchanged (a segment or a point
    is its own hull).
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n <= 2:
        return pts.tolist()
    # Sort by x then y.
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    sorted_pts = pts[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Lower hull.
    lower: List[np.ndarray] = []
    for p in sorted_pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    # Upper hull.
    upper: List[np.ndarray] = []
    for p in sorted_pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return [[float(v[0]), float(v[1])] for v in hull]
