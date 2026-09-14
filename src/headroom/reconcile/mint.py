"""MinT reconciliation with a shrunk covariance of the forecasts' own past errors.

Base forecasts made one series at a time do not sum: the city's median is not the sum of
the boroughs' medians. MinT (Wickramasuriya, Athanasopoulos and Hyndman, 2019) replaces
them with the coherent set closest to them in the metric of their error covariance ``W``:

    reconciled = S G base,   G = (S' W^-1 S)^-1 S' W^-1

so a series whose errors are large and correlated with its parent's is moved more than a
series the model forecasts well. ``S G`` is a projection onto coherent vectors, so an input
that is already coherent comes back unchanged.

## W is shrunk, and it comes from out-of-sample errors

The sample covariance of 37 series from a year of weekly errors is too noisy to invert
well, so it is shrunk toward its own diagonal by the Schafer and Strimmer estimator.
:func:`shrunk_covariance` is written to match hierarchicalforecast's ``mint_shrink``
exactly (centred errors, ``n - 1`` denominator, the same shrinkage intensity), and
`tests/test_reconcile.py` checks the projection against the library's to 1e-6.

The textbook estimates ``W`` from in-sample one-step residuals. Here it is estimated from
**each model's own out-of-sample errors at the same horizon step over the previous 52
origins**, with the feedback rule of :func:`headroom.conformal.split.available_upto`.
Two reasons: the checkpoints hold forecasts, not in-sample fits, and an error covariance
for fourteen days ahead is better estimated from fourteen-day-ahead errors than from
one-day-ahead ones. The cost is the same burn-in as the conformal distribution: origins
before a full window are not reconciled.

## Only the median is reconciled

The median of every node is reconciled, and each node's whole quantile forecast is moved
by the same amount its median moved, so its spread is the base model's. That makes the
medians coherent at every origin, which is the claim the coherence check enforces.
Quantiles cannot sum in general (the 90th percentile of a sum is not the sum of 90th
percentiles), so nothing here claims they do. A reconciliation of the whole distribution,
through coherent sample paths, is PLAN.md's probabilistic step and is not built.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from headroom.conformal.predictive import first_valid_origin
from headroom.conformal.split import WINDOW, available_upto
from headroom.hierarchy.spec import Hierarchy

#: Added to the diagonal of ``W`` before inverting, as hierarchicalforecast does.
RIDGE: float = 2e-8


def shrunk_covariance(errors: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], float]:
    """Shrink the sample covariance of errors toward its diagonal (Schafer and Strimmer).

    Args:
        errors: One row per observation and one column per node, shape
            ``(n_samples, n_nodes)``. At least two rows.

    Returns:
        ``(W, intensity)``: the shrunk covariance with :data:`RIDGE` on its diagonal, and
        the shrinkage intensity between 0 (sample covariance) and 1 (diagonal only).

    Raises:
        ValueError: Fewer than two observations, or a node whose errors never vary.
    """
    n = errors.shape[0]
    if errors.ndim != 2 or n < 2:
        raise ValueError(f"need a 2-D array with at least two rows, got {errors.shape}")
    centred = errors - errors.mean(axis=0)
    covariance = centred.T @ centred / (n - 1)
    scale = np.sqrt(np.diag(covariance))
    if np.any(scale == 0.0):
        raise ValueError("a node's errors never vary, so its correlations are undefined")

    standardised = centred / scale
    correlation = standardised.T @ standardised / (n - 1)
    products = standardised[:, :, np.newaxis] * standardised[:, np.newaxis, :]
    variance = n / (n - 1) ** 3 * ((products - products.mean(axis=0)) ** 2).sum(axis=0)
    off = ~np.eye(errors.shape[1], dtype=np.bool_)
    denominator = float((correlation[off] ** 2).sum())
    intensity = 1.0 if denominator == 0.0 else float(variance[off].sum()) / denominator
    intensity = min(max(intensity, 0.0), 1.0)

    shrunk = intensity * np.diag(np.diag(covariance)) + (1.0 - intensity) * covariance
    shrunk[np.diag_indices_from(shrunk)] += RIDGE
    return shrunk, intensity


def mint_matrix(
    s_matrix: npt.NDArray[np.float64], w: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return ``G``, which maps base forecasts of every node to reconciled leaves.

    Args:
        s_matrix: The summing matrix, shape ``(n_nodes, n_leaves)``.
        w: The error covariance, shape ``(n_nodes, n_nodes)``.

    Returns:
        ``(S' W^-1 S)^-1 S' W^-1``, shape ``(n_leaves, n_nodes)``.
    """
    w_inv_s = np.linalg.solve(w, s_matrix)
    result: npt.NDArray[np.float64] = np.linalg.solve(s_matrix.T @ w_inv_s, w_inv_s.T)
    return result


@dataclass(frozen=True, slots=True)
class ReconciledMedians:
    """Reconciled medians over a backtest, for the origins that have a full window.

    Attributes:
        medians: Shape ``(n_origins, n_nodes, horizon)``; ``nan`` before
            :attr:`first_valid`.
        first_valid: The first reconciled origin.
        intensity: Shrinkage intensity per origin and horizon step, ``nan`` before
            :attr:`first_valid`.
        coherence_error: The largest breach of the summing constraints over every
            reconciled origin and step, in incidents. Zero to rounding.
    """

    medians: npt.NDArray[np.float64]
    first_valid: int
    intensity: npt.NDArray[np.float64]
    coherence_error: float


def reconcile_backtest(
    median: npt.NDArray[np.float64],
    actual: npt.NDArray[np.float64],
    hierarchy: Hierarchy,
    origin_step: int,
    window: int = WINDOW,
) -> ReconciledMedians:
    """Reconcile a backtest's medians with MinT, estimating ``W`` from past errors.

    Args:
        median: Base medians, shape ``(n_origins, n_nodes, horizon)``.
        actual: What happened, same shape.
        hierarchy: The hierarchy, whose summing matrix defines coherence.
        origin_step: Days between origins, for the feedback rule.
        window: Past origins whose errors estimate ``W``.

    Returns:
        The reconciled medians, their shrinkage intensities and the coherence check.

    Raises:
        ValueError: The shapes disagree with each other or with the hierarchy, or no
            origin has a full window.
    """
    if median.shape != actual.shape or median.ndim != 3:
        raise ValueError(f"median {median.shape} and actual {actual.shape} must match, 3-D")
    n_origins, n_nodes, horizon = median.shape
    if n_nodes != hierarchy.n_nodes:
        raise ValueError(f"{n_nodes} nodes, hierarchy has {hierarchy.n_nodes}")
    first = first_valid_origin(horizon, origin_step, window)
    if first >= n_origins:
        raise ValueError(f"{n_origins} origins is too few for a {window}-origin window")

    s = hierarchy.s_matrix
    errors = actual - median
    out = np.full(median.shape, np.nan)
    intensity = np.full((n_origins, horizon), np.nan)
    worst = 0.0
    for step in range(1, horizon + 1):
        for origin in range(first, n_origins):
            end = available_upto(origin, step, origin_step)
            w, intensity[origin, step - 1] = shrunk_covariance(
                errors[end - window : end, :, step - 1]
            )
            reconciled = s @ (mint_matrix(s, w) @ median[origin, :, step - 1])
            out[origin, :, step - 1] = reconciled
            worst = max(worst, hierarchy.coherence_error(reconciled))
    return ReconciledMedians(
        medians=out, first_valid=first, intensity=intensity, coherence_error=worst
    )


def shift_quantiles(
    quantiles: npt.NDArray[np.float64],
    base_median: npt.NDArray[np.float64],
    reconciled_median: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Move each node's quantile forecast by the amount reconciliation moved its median.

    Args:
        quantiles: Base quantiles, shape ``(..., n_levels)``.
        base_median: Base medians, shape ``(...)``.
        reconciled_median: Reconciled medians, same shape.

    Returns:
        Shifted quantiles, floored at zero. The floor can only bind on a node whose
        reconciled lower quantiles fall below zero; the medians themselves are not floored,
        so their coherence is untouched.
    """
    shifted = quantiles + (reconciled_median - base_median)[..., np.newaxis]
    return np.clip(shifted, 0.0, None)
