"""Checkpointing, so a long backtest survives being interrupted.

The statistical models cost roughly 190 seconds per origin for all four on this panel
(`docs/methods.md` has the measured table), which puts a full-record backtest in the range
of hours rather than minutes. A run that has to start again from the beginning because a
laptop slept is a run that never finishes.

So the driver writes each origin's forecasts as it produces them and picks up where it
stopped. The checkpoint holds **forecasts, not scores**: scoring is cheap and changing a
score should not mean refitting a model for five hours.

The file records the settings the run was started with and refuses to resume into a
checkpoint built under different ones. Silently mixing forecasts from two different
schedules would produce a results table that never existed.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from headroom.backtest.origins import Origins


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Partial forecasts for one model over one origin schedule.

    Attributes:
        path: Where this checkpoint lives.
        settings: The run's settings, compared on resume.
        quantiles: Forecasts so far, shape ``(n_origins, n_nodes, horizon, n_levels)``.
            Origins not yet computed are ``nan``.
        done: Whether each origin has been computed, shape ``(n_origins,)``.
        seconds: Model time spent per origin, shape ``(n_origins,)``.
    """

    path: Path
    settings: dict[str, Any]
    quantiles: npt.NDArray[np.float64]
    done: npt.NDArray[np.bool_]
    seconds: npt.NDArray[np.float64]

    @property
    def n_done(self) -> int:
        """How many origins are complete."""
        return int(self.done.sum())

    @property
    def complete(self) -> bool:
        """Whether every origin is complete."""
        return bool(self.done.all())

    def next_origin(self) -> int | None:
        """Return the first origin still to compute, or None if there are none.

        Returns:
            The origin number, or None.
        """
        remaining = np.flatnonzero(~self.done)
        return int(remaining[0]) if remaining.size else None

    def record(self, origin: int, quantiles: npt.NDArray[np.float64], seconds: float) -> None:
        """Store one origin's forecasts in memory.

        Args:
            origin: The origin number.
            quantiles: That origin's forecasts, shape ``(n_nodes, horizon, n_levels)``.
            seconds: Model time spent.
        """
        self.quantiles[origin] = quantiles
        self.done[origin] = True
        self.seconds[origin] = seconds

    def save(self) -> None:
        """Write the checkpoint to disk atomically.

        Written to a temporary file and moved into place, so an interrupt during the
        write cannot leave a half-written checkpoint that resumes into nonsense.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".partial")
        # Written through an open handle rather than by name: np.savez_compressed appends
        # ".npz" to a path that does not already end in it, so passing the temporary path
        # by name would write somewhere else and the rename below would not find it.
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                quantiles=self.quantiles,
                done=self.done,
                seconds=self.seconds,
                settings=np.array(json.dumps(self.settings)),
            )
        temporary.replace(self.path)


def settings_of(
    model: str, origins: Origins, n_nodes: int, levels: npt.NDArray[np.float64]
) -> dict[str, Any]:
    """Describe a run, for the compatibility check on resume.

    Args:
        model: The model's name.
        origins: The origin schedule.
        n_nodes: Nodes in the hierarchy.
        levels: The quantile grid.

    Returns:
        A JSON-serialisable description.
    """
    return {
        "model": model,
        "n_origins": len(origins),
        "horizon": origins.horizon,
        "step": origins.step,
        "min_train": origins.min_train,
        "first_day": origins.days[0].isoformat(),
        "last_day": origins.days[-1].isoformat(),
        "n_nodes": n_nodes,
        "n_levels": int(levels.size),
    }


def open_checkpoint(
    path: Path,
    model: str,
    origins: Origins,
    n_nodes: int,
    levels: npt.NDArray[np.float64],
) -> Checkpoint:
    """Load a checkpoint, or start a new one.

    Args:
        path: Where the checkpoint lives.
        model: The model's name.
        origins: The origin schedule.
        n_nodes: Nodes in the hierarchy.
        levels: The quantile grid.

    Returns:
        The checkpoint, resumed if the file exists and matches.

    Raises:
        ValueError: The existing checkpoint was built under different settings.
    """
    wanted = settings_of(model, origins, n_nodes, levels)
    shape = (len(origins), n_nodes, origins.horizon, levels.size)

    if not path.exists():
        return Checkpoint(
            path=path,
            settings=wanted,
            quantiles=np.full(shape, np.nan),
            done=np.zeros(len(origins), dtype=np.bool_),
            seconds=np.zeros(len(origins)),
        )

    with np.load(path, allow_pickle=False) as stored:
        found = json.loads(str(stored["settings"]))
        if found != wanted:
            differences = {
                key: (found.get(key), wanted.get(key))
                for key in set(found) | set(wanted)
                if found.get(key) != wanted.get(key)
            }
            raise ValueError(
                f"{path} was built under different settings and cannot be resumed into. "
                f"Differences as (stored, requested): {differences}. Delete it to start again."
            )
        return Checkpoint(
            path=path,
            settings=found,
            quantiles=stored["quantiles"],
            done=stored["done"],
            seconds=stored["seconds"],
        )
