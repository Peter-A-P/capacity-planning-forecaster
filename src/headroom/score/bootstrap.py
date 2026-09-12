"""The moving block bootstrap, over forecast origins.

Every reported number carries a confidence interval; this is where they come from.

## Why blocks

The ordinary bootstrap resamples origins independently, which assumes a method's score on
one origin says nothing about its score on the next. That is false here and obviously so:
a model that was badly calibrated in the first week of April 2020 was badly calibrated in
the second. Resampling independently breaks that dependence, the resamples look more
varied than the data, and the interval comes out too narrow. The interval would then be a
statement about a world where the errors were independent, which is not the world the
forecast has to work in.

The moving block bootstrap resamples **contiguous runs of origins** instead, so whatever
dependence exists inside a run of :data:`BLOCK` origins is carried into the resample
intact. It is the standard device for a dependent series and it is the same reason the
conformal work in :mod:`headroom.conformal` cannot assume exchangeability.

## What the block length costs

Too short and the dependence is broken again; too long and there are too few distinct
blocks for the resample to vary. :data:`BLOCK` is set at 28 origins, four weeks, which
is longer than the weekly seasonality in this data and long enough to hold most of a
regime change. `docs/methods.md` reports how the intervals move as it changes, because a
block length that the answer is sensitive to is a result about the block length rather
than about the forecast.
"""

from collections.abc import Callable
from typing import Final

import numpy as np
import numpy.typing as npt

#: Block length in origins. Four weeks: longer than the weekly cycle, short enough that a
#: twenty-year backtest still has hundreds of distinct blocks to draw from.
BLOCK: Final[int] = 28

#: Resamples per interval. 2,000 is enough that the interval's own Monte Carlo error is
#: small next to the width it is reporting, and cheap enough to run for every cell of
#: every table.
RESAMPLES: Final[int] = 2_000

#: Default interval. PLAN.md section 1 asks for 95 percent.
ALPHA: Final[float] = 0.05


def block_indices(n: int, block: int, rng: np.random.Generator) -> npt.NDArray[np.intp]:
    """Draw one moving-block resample's worth of indices.

    The indices are contiguous inside each block and in increasing order there, which is
    the whole point: `tests/test_score.py` asserts it, because a bug that shuffled within
    a block would silently turn this back into the independent bootstrap and quietly
    narrow every interval in the project.

    Args:
        n: Number of origins.
        block: Block length.
        rng: The random generator.

    Returns:
        ``n`` indices into the origins, as ``ceil(n / block)`` contiguous blocks laid end
        to end and truncated to length ``n``.

    Raises:
        ValueError: The block length does not fit the number of origins.
    """
    if n < 1:
        raise ValueError("no origins to resample")
    if not 1 <= block <= n:
        raise ValueError(f"block must be between 1 and {n}, got {block}")

    n_blocks = -(-n // block)  # ceiling division
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    offsets = np.arange(block)
    return (starts[:, np.newaxis] + offsets[np.newaxis, :]).ravel()[:n]


def block_bootstrap(
    values: npt.NDArray[np.float64],
    statistic: Callable[[npt.NDArray[np.float64]], np.float64 | float] = np.mean,
    block: int = BLOCK,
    resamples: int = RESAMPLES,
    seed: int = 0,
) -> npt.NDArray[np.float64]:
    """Return the bootstrap distribution of a statistic over origins.

    Args:
        values: Per-origin values, shape ``(n_origins,)``, in origin order.
        statistic: What to compute on each resample. The mean by default.
        block: Block length in origins.
        resamples: Number of resamples.
        seed: Seed, so every published interval reproduces exactly.

    Returns:
        The statistic on each resample, shape ``(resamples,)``.

    Raises:
        ValueError: ``values`` is not one-dimensional.
    """
    if values.ndim != 1:
        raise ValueError("bootstrap takes one series of per-origin values at a time")
    rng = np.random.default_rng(seed)
    n = values.size
    return np.array(
        [float(statistic(values[block_indices(n, block, rng)])) for _ in range(resamples)]
    )


def confidence_interval(
    values: npt.NDArray[np.float64],
    statistic: Callable[[npt.NDArray[np.float64]], np.float64 | float] = np.mean,
    block: int = BLOCK,
    resamples: int = RESAMPLES,
    alpha: float = ALPHA,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return a statistic and its block-bootstrap confidence interval.

    The percentile interval, which is the right choice here: the statistics being
    bootstrapped are means and differences of means of scores, whose distributions are
    close to symmetric, and a percentile interval makes no assumption the data has to
    earn.

    Args:
        values: Per-origin values, in origin order.
        statistic: What to compute. The mean by default.
        block: Block length in origins.
        resamples: Number of resamples.
        alpha: ``1 - alpha`` is the coverage; 0.05 gives a 95 percent interval.
        seed: Seed, so every published interval reproduces exactly.

    Returns:
        ``(point, lower, upper)``, the statistic on the data itself and the interval.

    Raises:
        ValueError: ``alpha`` is not strictly between 0 and 1.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    draws = block_bootstrap(values, statistic, block, resamples, seed)
    lower, upper = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(statistic(values)), float(lower), float(upper)


def skill_interval(
    score: npt.NDArray[np.float64],
    baseline: npt.NDArray[np.float64],
    block: int = BLOCK,
    resamples: int = RESAMPLES,
    alpha: float = ALPHA,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return skill against a baseline, with a paired block-bootstrap interval.

    The pair matters. Both methods are resampled on the **same** drawn origins, so the
    origin-to-origin variation that dwarfs the difference between methods cancels. An
    interval built by bootstrapping the two separately would be several times wider and
    would report no difference where there is one.

    Args:
        score: The method's per-origin scores, in origin order.
        baseline: The baseline's per-origin scores, same origins and order.
        block: Block length in origins.
        resamples: Number of resamples.
        alpha: ``1 - alpha`` is the coverage.
        seed: Seed, so every published interval reproduces exactly.

    Returns:
        ``(skill, lower, upper)``.

    Raises:
        ValueError: The two score series are not the same shape or are not 1-D.
    """
    if score.shape != baseline.shape:
        raise ValueError(
            f"paired bootstrap needs the same origins, got {score.shape} and {baseline.shape}"
        )
    if score.ndim != 1:
        raise ValueError("bootstrap takes one series of per-origin values at a time")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    rng = np.random.default_rng(seed)
    n = score.size
    draws = np.empty(resamples)
    for i in range(resamples):
        idx = block_indices(n, block, rng)
        draws[i] = 1.0 - score[idx].mean() / baseline[idx].mean()
    lower, upper = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    point = 1.0 - score.mean() / baseline.mean()
    return float(point), float(lower), float(upper)
