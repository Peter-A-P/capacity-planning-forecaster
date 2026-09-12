"""Conformal prediction intervals under distribution shift.

The headline of this project (PLAN.md section 2.4). Exchangeability does not hold for a
time series, so ordinary split conformal has no guarantee here and is expected to fail
through the March 2020 shift. It is kept as the comparison rather than quietly dropped.

The assumption each method makes, and what it does and does not buy, is stated in
`docs/methods.md` and is repeated in the README beside the coverage chart. The one thing
adaptive conformal inference does **not** give is per-period coverage; saying otherwise
would be the single most likely way for this project to be wrong in public.
"""

from headroom.conformal.aci import AdaptiveConformal
from headroom.conformal.agaci import AggregatedConformal
from headroom.conformal.split import SplitConformal, conformal_quantile

__all__ = [
    "AdaptiveConformal",
    "AggregatedConformal",
    "SplitConformal",
    "conformal_quantile",
]
