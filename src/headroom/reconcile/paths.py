"""Probabilistic reconciliation: a whole distribution that lives on the coherent subspace.

`PLAN.md` section 2.5. :mod:`headroom.reconcile.mint` reconciles the median and shifts each
node's quantiles with it, which makes the **medians** coherent and says nothing about the
rest of the distribution. This builds the distribution itself out of coherent pieces.

## How

At each origin and horizon step, take the model's own out-of-sample errors over the
previous 52 origins, which is the same window, the same feedback rule and the same errors
:mod:`headroom.reconcile.mint` estimates ``W`` from. Each of those 52 errors is a vector
over all 37 nodes at once, so it carries the cross-sectional dependence the marginal
distributions throw away. Add each error vector to the base forecast to get 52 incoherent
futures, then put every one of them through the MinT projection ``S G``:

    path_i = S G (base + error_i)

``S G`` is a projection onto the coherent subspace, so **every path is coherent exactly**,
and the predictive distribution is the empirical distribution of those paths. That is the
claim this module makes and the one :func:`reconcile_paths` verifies at every origin.

## The quantiles still do not sum, and that is not a defect

A coherent distribution does not have coherent quantiles. The 90th percentile of the city
is not the sum of the boroughs' 90th percentiles, because the boroughs do not all have
their bad days together. `tests/test_paths.py` asserts that the marginals do not sum, so
that nobody later "fixes" it into something false. What is coherent is every draw, which is
what a planner needs: any total computed from a single path agrees with that path's parts,
and a decision taken over the whole hierarchy is taken on one consistent future.

## Why the 52 error vectors are used as they are, and not resampled

`PLAN.md` says "bootstrap of reconciled in-sample errors". Resampling with replacement from
52 vectors adds no information to those 52, and it would cost the exact convention the rest
of this package uses: the ``q`` quantile as the ``k``-th smallest with
``k = ceil((n + 1) q)``, which is what :mod:`headroom.conformal.predictive` and
:mod:`headroom.conformal.split` already do. So the full sample is used once, the order
statistics are read the same way as everywhere else, and there is no seed to record. Two
consequences are inherited rather than new: the distribution cannot be wider than the
window's widest error, and above ``n / (n + 1)`` there are too few errors to certify a
level, so the widest is used.

## Nothing is floored at zero

Clipping a path at zero would break its coherence, which is the one property this exists to
provide. The paths are left alone, the coherence check is therefore exact, and the share of
path values that fall below zero is measured and reported instead of hidden. It is small
and it is concentrated in the dispatch areas, where a quiet day is near zero anyway.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from headroom.conformal.predictive import first_valid_origin, order_statistic_ranks
from headroom.conformal.split import WINDOW, available_upto
from headroom.hierarchy.spec import Hierarchy
from headroom.reconcile.mint import mint_matrix, shrunk_covariance
from headroom.score.levels import check_levels


@dataclass(frozen=True, slots=True)
class ReconciledPaths:
    """A coherent predictive distribution over a backtest.

    Attributes:
        quantiles: Shape ``(n_origins, n_nodes, horizon, n_levels)``; ``nan`` before
            :attr:`first_valid`. Read off the coherent paths, so the marginals do not sum.
        medians: The point MinT reconciliation, shape ``(n_origins, n_nodes, horizon)``,
            computed in the same pass so a caller needs only one. ``nan`` before
            :attr:`first_valid`.
        first_valid: The first origin with a full window, and so the first with a
            distribution.
        intensity: Shrinkage intensity per origin and horizon step.
        coherence_error: The largest breach of the summing constraints over **every path**
            at every origin and step, in incidents. Zero to rounding.
        negative_share: The share of path values below zero, which is what not flooring
            them costs.
    """

    quantiles: npt.NDArray[np.float64]
    medians: npt.NDArray[np.float64]
    first_valid: int
    intensity: npt.NDArray[np.float64]
    coherence_error: float
    negative_share: float


def reconcile_paths(
    median: npt.NDArray[np.float64],
    actual: npt.NDArray[np.float64],
    hierarchy: Hierarchy,
    levels: npt.NDArray[np.float64],
    origin_step: int,
    window: int = WINDOW,
) -> ReconciledPaths:
    """Reconcile a backtest's whole distribution through coherent sample paths.

    Args:
        median: Base medians, shape ``(n_origins, n_nodes, horizon)``. Only the median is
            needed: the spread comes from the error window, not from the base model's own
            quantiles, so a median-only model is reconciled the same way as any other.
        actual: What happened, same shape.
        hierarchy: The hierarchy, whose summing matrix defines coherence.
        levels: The quantile grid to report the paths on.
        origin_step: Days between origins, for the feedback rule.
        window: Past origins whose errors give both ``W`` and the paths.

    Returns:
        The coherent distribution, the point reconciliation beside it, and the two checks.

    Raises:
        ValueError: The shapes disagree with each other or with the hierarchy, or the
            backtest is too short for any origin to have a full window.
    """
    check_levels(levels)
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
    ranks = order_statistic_ranks(levels, window)
    out = np.full((n_origins, n_nodes, horizon, levels.size), np.nan)
    medians = np.full(median.shape, np.nan)
    intensity = np.full((n_origins, horizon), np.nan)
    worst = 0.0
    negative = 0
    counted = 0
    for step in range(1, horizon + 1):
        for origin in range(first, n_origins):
            end = available_upto(origin, step, origin_step)
            window_errors = errors[end - window : end, :, step - 1]
            w, intensity[origin, step - 1] = shrunk_covariance(window_errors)
            # S G, the projection onto coherent vectors. Every column of anything it
            # multiplies comes out coherent, which is the whole mechanism.
            projection = s @ mint_matrix(s, w)
            base = median[origin, :, step - 1]
            medians[origin, :, step - 1] = projection @ base
            paths = (projection @ (base + window_errors).T).T
            out[origin, :, step - 1, :] = np.sort(paths, axis=0)[ranks, :].T
            worst = max(worst, hierarchy.coherence_error(paths.T))
            negative += int((paths < 0.0).sum())
            counted += paths.size
    return ReconciledPaths(
        quantiles=out,
        medians=medians,
        first_valid=first,
        intensity=intensity,
        coherence_error=worst,
        negative_share=negative / counted,
    )
