"""Shared types.

The vocabulary is deliberately small: a *series* is one node of the hierarchy through
time, a *level* is a rank of the hierarchy (``city``, ``borough``, ``area``), and a
*node* is one series identifier at one level.
"""

from datetime import date
from typing import Any, Final, Literal

#: A rank of the hierarchy. ``area`` holds the leaves; ``city`` is the single root.
type Level = Literal["city", "borough", "area"]

#: Levels from root to leaf. The order matters: the summing matrix is built in it.
LEVELS: Final[tuple[Level, ...]] = ("city", "borough", "area")

#: A node identifier, unique across the whole hierarchy (e.g. ``"NYC"``,
#: ``"NYC/BROOKLYN"``, ``"NYC/BROOKLYN/K5"``).
type NodeId = str

#: The root node's identifier. The hierarchy is one city, so it is a constant.
ROOT: Final[str] = "NYC"

#: Separator between hierarchy ranks inside a :data:`NodeId`.
SEP: Final[str] = "/"


def as_date(value: Any, what: str) -> date:  # noqa: ANN401 - polars aggregates are Any
    """Narrow a polars aggregate to a date, or say which one was not.

    ``Series.min()`` and friends are typed ``Any``, so every caller that reads a window
    boundary out of a frame has to narrow it. Doing that here keeps the narrowing in one
    place and turns a wrong dtype into a message naming the column rather than an
    AttributeError three frames later.

    Args:
        value: The aggregate to narrow.
        what: What it was, for the error message.

    Returns:
        The value as a date.

    Raises:
        TypeError: It is not a date, which means the frame's dtype is wrong.
    """
    if not isinstance(value, date):
        raise TypeError(f"{what} is {type(value).__name__}, expected a date")
    return value
