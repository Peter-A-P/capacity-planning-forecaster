"""Forecast origins: where each forecast is made from, and what it may see.

A forecast origin is a day on which a forecaster is allowed to know everything up to and
including that day, and nothing after it. Every number this project reports comes from a
forecast made at an origin and scored against days the forecaster had not seen.

The whole credibility of the results tables rests on that being enforced rather than
intended, so the rule lives in one small pure module with a test that asserts it directly:
the training slice ends at the origin, the target slice starts the day after, and they
never touch. A subtle look-ahead is the most common way a forecasting result turns out to
be worthless, and it is invisible in the output.

## The three choices this module fixes

**Horizon.** Fourteen days. It is the horizon a rota is actually built over: far enough
ahead that a staffing decision can be acted on, close enough that the forecast is not
fiction. It spans two weekly cycles, so a method that only gets the weekly shape right is
visibly not enough.

**Spacing.** Origins every seven days. Consecutive daily origins would produce fourteen
times more overlapping forecasts that are almost the same forecast, which costs fourteen
times the compute and adds almost no information, while making the dependence between
origins that the block bootstrap has to handle far worse. Weekly spacing also holds the
day of week fixed across origins, so no method is flattered by having more Mondays.

**Refitting.** Every fourth origin, so roughly monthly. Refitting at every origin is the
ideal and is what a fair comparison wants; refitting monthly is what the CPU budget in
PLAN.md section 6 allows for the neural models, and applying the same schedule to the
statistical models keeps the comparison fair rather than giving the cheap models an
advantage the expensive ones were denied.

**How much history each forecast sees.** A trailing window of
:data:`TRAIN_WINDOW_DAYS`, not everything back to 2005. Two reasons and they point the
same way.

The statistical reason: by the last origin an expanding window would be fitting on
twenty-one years spanning two regime changes, and a model that averages 2008 and 2025
demand is describing a city that no longer exists. Three years is long enough for three
annual cycles and short enough to have left the last regime behind.

The practical reason, which is why this is a fixed window rather than an afterthought:
under an expanding window the training length runs from 1,095 days at the first origin to
7,829 at the last, so **every fit gets steadily more expensive** and a backtest's total
cost cannot be estimated from its first origins. Measured, the same three models cost
about 120 seconds an origin early and far more late. A trailing window makes the cost per
origin flat and the total predictable.

Set ``train_window`` to None for the expanding behaviour, which is what the cheap models
can afford and what a reader may expect to see compared.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from typing import Final

#: Days ahead each forecast covers.
HORIZON: Final[int] = 14

#: Days between consecutive origins.
STEP: Final[int] = 7

#: Origins between refits. 4 origins at a 7-day step is roughly monthly.
REFIT_EVERY: Final[int] = 4

#: Smallest training history before the first origin. Three years covers three annual
#: cycles, which is the least a model that claims to know the annual shape can be fitted
#: on, and it still leaves eighteen years of the record to be scored over.
MIN_TRAIN_DAYS: Final[int] = 3 * 365

#: Days of history each forecast may see. Equal to :data:`MIN_TRAIN_DAYS`, so every origin
#: in the backtest sees exactly the same amount of history as the first one and no model
#: is advantaged by where in the record it happened to be asked.
TRAIN_WINDOW_DAYS: Final[int] = MIN_TRAIN_DAYS


@dataclass(frozen=True, slots=True)
class Origin:
    """One forecast origin.

    Attributes:
        number: Position in the backtest, counting from zero.
        index: The origin's position in the panel's day axis. The forecaster may see
            every day up to and including this one.
        day: The origin's date.
        horizon: Days forecast ahead.
        refit: Whether a model should be refitted at this origin rather than reused.
        train_window: Days of history the forecaster may see, or None for everything up
            to the origin.
    """

    number: int
    index: int
    day: date
    horizon: int
    refit: bool
    train_window: int | None = None

    @property
    def train(self) -> slice:
        """The days a forecaster made at this origin may see.

        Ends at ``index + 1`` because a slice bound is exclusive and the origin day
        itself is observed. Starts at the window, or at the beginning of the record when
        there is no window.
        """
        start = 0 if self.train_window is None else max(0, self.index + 1 - self.train_window)
        return slice(start, self.index + 1)

    @property
    def target(self) -> slice:
        """The days this forecast is scored against, starting the day after the origin."""
        return slice(self.index + 1, self.index + 1 + self.horizon)

    @property
    def target_days(self) -> range:
        """The target day indices, as a range."""
        return range(self.index + 1, self.index + 1 + self.horizon)


@dataclass(frozen=True, slots=True)
class Origins:
    """The schedule of forecast origins for a backtest.

    Attributes:
        days: The panel's days, ascending and contiguous.
        horizon: Days forecast ahead at each origin.
        step: Days between origins.
        refit_every: Origins between refits.
        min_train: Days of history required before the first origin.
        train_window: Days of history each forecast may see, or None to let it expand.
    """

    days: tuple[date, ...]
    horizon: int = HORIZON
    step: int = STEP
    refit_every: int = REFIT_EVERY
    min_train: int = MIN_TRAIN_DAYS
    train_window: int | None = TRAIN_WINDOW_DAYS

    def __post_init__(self) -> None:
        """Check the schedule can produce at least one scorable origin.

        Raises:
            ValueError: A parameter is not positive, or the record is too short to hold
                the training history, one origin and a full horizon after it.
        """
        for name, value in (
            ("horizon", self.horizon),
            ("step", self.step),
            ("refit_every", self.refit_every),
            ("min_train", self.min_train),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        if self.train_window is not None:
            if self.train_window < 1:
                raise ValueError(f"train_window must be at least 1, got {self.train_window}")
            if self.train_window > self.min_train:
                raise ValueError(
                    f"train_window of {self.train_window} is longer than the {self.min_train} "
                    "days of history guaranteed before the first origin, so the first "
                    "forecasts would see less history than the later ones"
                )
        needed = self.min_train + self.horizon
        if len(self.days) < needed:
            raise ValueError(
                f"{len(self.days)} days is too short: {self.min_train} of training history "
                f"plus a {self.horizon}-day horizon needs {needed}"
            )

    @property
    def first_index(self) -> int:
        """Day index of the first origin."""
        return self.min_train - 1

    @property
    def last_index(self) -> int:
        """Day index of the last origin whose whole horizon is inside the record.

        A forecast whose horizon runs off the end of the data would be scored on fewer
        days than the others, which would quietly weight the end of the record
        differently. The schedule stops before that rather than truncating.
        """
        return len(self.days) - 1 - self.horizon

    def __iter__(self) -> Iterator[Origin]:
        """Yield every origin in the schedule, in order."""
        indices = range(self.first_index, self.last_index + 1, self.step)
        for number, index in enumerate(indices):
            yield Origin(
                number=number,
                index=index,
                day=self.days[index],
                horizon=self.horizon,
                refit=number % self.refit_every == 0,
                train_window=self.train_window,
            )

    def __len__(self) -> int:
        """The number of origins in the schedule."""
        span = self.last_index - self.first_index
        return span // self.step + 1 if span >= 0 else 0

    @property
    def n_refits(self) -> int:
        """How many model fits the schedule costs."""
        return -(-len(self) // self.refit_every)

    def covering(self, day: date) -> list[int]:
        """Return the numbers of the origins whose horizon covers a day.

        Used by the coverage chart, which has to say which forecasts were in flight when
        the March 2020 shift arrived.

        Args:
            day: The day to look for.

        Returns:
            Origin numbers, ascending. Empty if no origin's horizon reaches it.
        """
        try:
            index = self.days.index(day)
        except ValueError:
            return []
        return [o.number for o in self if index in o.target_days]
