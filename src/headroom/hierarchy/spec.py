"""The hierarchy and its summing matrix.

The summing matrix ``S`` is the whole structure in one object: a 0/1 matrix with one row
per node and one column per leaf, where ``S @ leaf_values`` is every node's value. MinT
reconciliation in :mod:`headroom.reconcile` is defined in terms of it, and the coherence
test is the statement that a set of forecasts equals ``S`` times its own leaves.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from headroom.types import LEVELS, ROOT, SEP, Level, NodeId

#: Tolerance for the coherence check. Reconciliation is a projection, so the residual is
#: rounding only; anything above this is a bug, not numerical noise.
COHERENCE_ATOL: Final[float] = 1e-8


@dataclass(frozen=True, slots=True)
class Hierarchy:
    """A hierarchy of series, with the matrix that sums its leaves to its nodes.

    Attributes:
        nodes: Every node identifier, ordered root first, then boroughs, then areas.
            The order is the row order of :attr:`s_matrix` and is stable across runs.
        leaves: The leaf identifiers, in the column order of :attr:`s_matrix`. They are
            the tail of :attr:`nodes`.
        levels: The level of each node, in the same order as :attr:`nodes`.
        s_matrix: The summing matrix, shape ``(len(nodes), len(leaves))``.
    """

    nodes: tuple[NodeId, ...]
    leaves: tuple[NodeId, ...]
    levels: tuple[Level, ...]
    s_matrix: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        """Check the invariants a caller would otherwise have to trust.

        Raises:
            ValueError: The pieces do not describe one consistent hierarchy.
        """
        if len(self.nodes) != len(self.levels):
            raise ValueError("one level per node")
        if self.s_matrix.shape != (len(self.nodes), len(self.leaves)):
            raise ValueError(
                f"S is {self.s_matrix.shape}, nodes and leaves say "
                f"{(len(self.nodes), len(self.leaves))}"
            )
        if self.nodes[-len(self.leaves) :] != self.leaves:
            raise ValueError("leaves must be the tail of nodes, in order")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("node identifiers must be unique")

    @property
    def n_nodes(self) -> int:
        """The number of series in the hierarchy, leaves included."""
        return len(self.nodes)

    @property
    def n_leaves(self) -> int:
        """The number of leaf series."""
        return len(self.leaves)

    def index(self, node: NodeId) -> int:
        """Return a node's row in :attr:`s_matrix`.

        Args:
            node: The node identifier.

        Returns:
            The row index.

        Raises:
            KeyError: The node is not in this hierarchy.
        """
        try:
            return self.nodes.index(node)
        except ValueError as exc:
            raise KeyError(node) from exc

    def rows_at(self, level: Level) -> npt.NDArray[np.intp]:
        """Return the row indices of every node at one level.

        Args:
            level: The level to select.

        Returns:
            Row indices into :attr:`nodes` and :attr:`s_matrix`.
        """
        return np.flatnonzero(np.asarray(self.levels) == level)

    def aggregate(self, leaf_values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Sum leaf values up to every node.

        Args:
            leaf_values: Leaf values, shape ``(n_leaves,)`` or ``(n_leaves, k)`` where
                ``k`` indexes anything held fixed across the hierarchy, such as time or
                a quantile.

        Returns:
            Values for every node, shape ``(n_nodes,)`` or ``(n_nodes, k)``.
        """
        return self.s_matrix @ leaf_values

    def coherence_error(self, values: npt.NDArray[np.float64]) -> float:
        """Return the largest absolute breach of the summing constraints.

        A coherent set of forecasts satisfies ``values == S @ values[leaf rows]``. This
        returns how far from that the input is, which the backtest records at every
        origin.

        Args:
            values: Values for every node, shape ``(n_nodes,)`` or ``(n_nodes, k)``.

        Returns:
            The maximum absolute difference. Zero to floating-point precision for
            anything built by summing leaves, and for anything MinT has projected.
        """
        leaf_rows = values[-self.n_leaves :]
        return float(np.max(np.abs(values - self.aggregate(leaf_rows))))

    def is_coherent(
        self, values: npt.NDArray[np.float64], atol: float = COHERENCE_ATOL
    ) -> bool:
        """Report whether values satisfy the summing constraints.

        Args:
            values: Values for every node.
            atol: Absolute tolerance; defaults to :data:`COHERENCE_ATOL`.

        Returns:
            True if every constraint holds within the tolerance.
        """
        return self.coherence_error(values) <= atol


def node_id(*parts: str) -> NodeId:
    """Build a node identifier from its path.

    Args:
        *parts: Path below the root, for example ``("BROOKLYN", "K5")``. No arguments
            gives the root.

    Returns:
        The identifier, for example ``"NYC/BROOKLYN/K5"``.
    """
    return SEP.join((ROOT, *parts))


def from_paths(paths: list[tuple[str, str]]) -> Hierarchy:
    """Build a three-level hierarchy from ``(borough, area)`` pairs.

    Every pair becomes a leaf, every distinct borough becomes an aggregate, and the root
    is the city. Boroughs and areas are each sorted, so the node order is a function of
    the input set alone and does not depend on the order it arrives in.

    Args:
        paths: One ``(borough, area)`` pair per leaf. Duplicates are collapsed.

    Returns:
        The hierarchy.

    Raises:
        ValueError: ``paths`` is empty, or an area appears under two boroughs.
    """
    if not paths:
        raise ValueError("a hierarchy needs at least one leaf")

    owner: dict[str, str] = {}
    for borough, area in paths:
        if owner.setdefault(area, borough) != borough:
            raise ValueError(
                f"area {area!r} appears under both {owner[area]!r} and {borough!r}; "
                "use headroom.data.nyc_ems.assign_areas_to_boroughs first"
            )

    boroughs = sorted({borough for borough, _ in paths})
    leaves_by_borough = {b: sorted(a for a, o in owner.items() if o == b) for b in boroughs}

    nodes: list[NodeId] = [ROOT]
    levels: list[Level] = ["city"]
    for borough in boroughs:
        nodes.append(node_id(borough))
        levels.append("borough")
    leaves: list[NodeId] = []
    for borough in boroughs:
        for area in leaves_by_borough[borough]:
            leaves.append(node_id(borough, area))
    nodes.extend(leaves)
    levels.extend(["area"] * len(leaves))

    s_matrix = np.zeros((len(nodes), len(leaves)), dtype=np.float64)
    leaf_column = {leaf: j for j, leaf in enumerate(leaves)}
    for i, (node, level) in enumerate(zip(nodes, levels, strict=True)):
        if level == "city":
            s_matrix[i, :] = 1.0
        elif level == "borough":
            for leaf in leaves:
                if leaf.startswith(node + SEP):
                    s_matrix[i, leaf_column[leaf]] = 1.0
        else:
            s_matrix[i, leaf_column[node]] = 1.0

    assert LEVELS == ("city", "borough", "area")
    return Hierarchy(
        nodes=tuple(nodes), leaves=tuple(leaves), levels=tuple(levels), s_matrix=s_matrix
    )
