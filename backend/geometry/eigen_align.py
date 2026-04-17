"""
Eigenbasis alignment and stability metrics.

Shared primitive for the Critique operations:
  - Embedding Dependence Probe (align bases from two embedding models)
  - Perturbation Test (align baseline to perturbed basis)
  - Forgetting Curve (align baseline to bootstrap bases)

The alignment logic is kept simple and visible: for each component in
basis A, find the component in basis B with the highest absolute cosine
similarity. Sign is ignored because PCA component signs are arbitrary
and cosine magnitude is the meaningful quantity. Basis dimensions may
differ (embeddings from different models live in different spaces); in
that case an alignment is meaningless and callers must project to a
shared space first (see ``align_via_doc_projection`` below).

Stability is reported both per-component (the individual cosine
magnitudes) and as an overall score (mean of the top-K magnitudes where
K is min(len(basis_a), len(basis_b))).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class ComponentMatch:
    a_index: int
    b_index: int
    abs_cosine: float        # |cos(a_i, b_j)|
    signed_cosine: float     # cos(a_i, b_j) — preserves the sign flip


@dataclass
class AlignmentResult:
    matches: List[ComponentMatch]
    stability: float              # mean |cos| across the matched components
    per_component: List[float]    # |cos| in a-basis order (len == len(basis_a))


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Return a copy of ``matrix`` with each row L2-normalised."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return matrix / norms


def align_eigenbases(
    basis_a: np.ndarray,
    basis_b: np.ndarray,
    *,
    greedy: bool = True,
) -> AlignmentResult:
    """
    Align ``basis_a`` (k_a, dim) with ``basis_b`` (k_b, dim).

    Requires both bases to live in the same ``dim``. For alignment
    across embedding models (different dims), use
    ``align_via_doc_projection``.

    Parameters
    ----------
    basis_a, basis_b : np.ndarray
        Principal-component matrices. Rows are components.
    greedy : bool
        If True, perform greedy best-match matching with uniqueness
        enforcement (each row of ``basis_b`` may be matched at most
        once). If False, each ``a_i`` is matched independently to its
        best ``b_j`` (matches may repeat).

    Returns
    -------
    AlignmentResult
    """
    if basis_a.shape[1] != basis_b.shape[1]:
        raise ValueError(
            f"basis_a has dim {basis_a.shape[1]} but basis_b has dim "
            f"{basis_b.shape[1]}; use align_via_doc_projection instead."
        )

    a = _unit_rows(basis_a.astype(np.float32))
    b = _unit_rows(basis_b.astype(np.float32))

    # Signed cosine matrix (k_a, k_b); |.| taken below.
    sim = a @ b.T
    abs_sim = np.abs(sim)

    matches: List[ComponentMatch] = []
    per_component: List[float] = []

    if greedy:
        used_b: set[int] = set()
        for a_idx in range(a.shape[0]):
            best_j = -1
            best_val = -1.0
            for b_idx in range(b.shape[0]):
                if b_idx in used_b:
                    continue
                v = float(abs_sim[a_idx, b_idx])
                if v > best_val:
                    best_val = v
                    best_j = b_idx
            if best_j == -1:
                per_component.append(0.0)
                continue
            used_b.add(best_j)
            matches.append(
                ComponentMatch(
                    a_index=a_idx,
                    b_index=best_j,
                    abs_cosine=float(abs_sim[a_idx, best_j]),
                    signed_cosine=float(sim[a_idx, best_j]),
                )
            )
            per_component.append(float(abs_sim[a_idx, best_j]))
    else:
        for a_idx in range(a.shape[0]):
            b_idx = int(np.argmax(abs_sim[a_idx]))
            matches.append(
                ComponentMatch(
                    a_index=a_idx,
                    b_index=b_idx,
                    abs_cosine=float(abs_sim[a_idx, b_idx]),
                    signed_cosine=float(sim[a_idx, b_idx]),
                )
            )
            per_component.append(float(abs_sim[a_idx, b_idx]))

    stability = (
        float(np.mean([m.abs_cosine for m in matches])) if matches else 0.0
    )
    return AlignmentResult(
        matches=matches, stability=stability, per_component=per_component
    )


def align_via_doc_projection(
    proj_a: np.ndarray,
    proj_b: np.ndarray,
    *,
    greedy: bool = True,
) -> AlignmentResult:
    """
    Align two bases by the rank-correlation of document projections.

    When basis_a and basis_b live in different dims (e.g. embeddings
    from two different models), the components themselves cannot be
    compared directly. Instead, compare how the two bases *rank the
    documents*. For each PC of basis_a, the vector of per-document
    scores is (n_docs,); ditto for basis_b. Their absolute Pearson
    correlation is a basis-dim-independent measure of agreement.

    Parameters
    ----------
    proj_a : np.ndarray
        (n_docs, k_a) — documents projected onto basis_a.
    proj_b : np.ndarray
        (n_docs, k_b) — same documents projected onto basis_b.
    """
    if proj_a.shape[0] != proj_b.shape[0]:
        raise ValueError(
            f"proj_a has {proj_a.shape[0]} rows but proj_b has "
            f"{proj_b.shape[0]}; must match (same documents)."
        )

    a = proj_a.astype(np.float64)
    b = proj_b.astype(np.float64)

    # Column-wise standardise (mean 0, unit std) so Pearson == dot / n.
    a_std = _standardise_columns(a)
    b_std = _standardise_columns(b)

    n = a_std.shape[0]
    corr = (a_std.T @ b_std) / max(n - 1, 1)  # (k_a, k_b)
    abs_corr = np.abs(corr)

    matches: List[ComponentMatch] = []
    per_component: List[float] = []

    if greedy:
        used_b: set[int] = set()
        for a_idx in range(a_std.shape[1]):
            best_j = -1
            best_val = -1.0
            for b_idx in range(b_std.shape[1]):
                if b_idx in used_b:
                    continue
                v = float(abs_corr[a_idx, b_idx])
                if v > best_val:
                    best_val = v
                    best_j = b_idx
            if best_j == -1:
                per_component.append(0.0)
                continue
            used_b.add(best_j)
            matches.append(
                ComponentMatch(
                    a_index=a_idx,
                    b_index=best_j,
                    abs_cosine=float(abs_corr[a_idx, best_j]),
                    signed_cosine=float(corr[a_idx, best_j]),
                )
            )
            per_component.append(float(abs_corr[a_idx, best_j]))
    else:
        for a_idx in range(a_std.shape[1]):
            b_idx = int(np.argmax(abs_corr[a_idx]))
            matches.append(
                ComponentMatch(
                    a_index=a_idx,
                    b_index=b_idx,
                    abs_cosine=float(abs_corr[a_idx, b_idx]),
                    signed_cosine=float(corr[a_idx, b_idx]),
                )
            )
            per_component.append(float(abs_corr[a_idx, b_idx]))

    stability = (
        float(np.mean([m.abs_cosine for m in matches])) if matches else 0.0
    )
    return AlignmentResult(
        matches=matches, stability=stability, per_component=per_component
    )


def _standardise_columns(matrix: np.ndarray) -> np.ndarray:
    means = matrix.mean(axis=0, keepdims=True)
    stds = matrix.std(axis=0, keepdims=True, ddof=1)
    stds = np.where(stds < 1e-12, 1.0, stds)
    return (matrix - means) / stds
