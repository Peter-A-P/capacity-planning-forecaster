"""The rolling-origin backtest.

:mod:`headroom.backtest.origins` defines where the forecasts are made from and what each
one is allowed to see. Every guarantee this project makes about its numbers rests on that
module being right, so it is small, it is pure, and it is tested for look-ahead directly.
"""

from headroom.backtest.origins import Origin, Origins

__all__ = ["Origin", "Origins"]
