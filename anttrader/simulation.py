"""Deterministic Langton's Ant on independent binary 16 x 16 tori."""

from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

DIRECTIONS = ("north", "east", "south", "west")
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


@dataclass(frozen=True)
class SimulationConfig:
    """Approved defaults; shorter runs and other valid poses support checks/replay."""

    steps: int = 2048
    start_row: int = 15
    start_col: int = 0
    initial_direction: str = "north"
    boundary: str = "toroidal"
    checkpoints: tuple[int, ...] = (256, 512, 1024, 2048)
    ants: int = 1
    reset_each_window: bool = True

    def __post_init__(self) -> None:
        if not _integer(self.steps) or self.steps < 1:
            raise ValueError("simulation.steps must be a positive integer.")
        for key in ("start_row", "start_col"):
            value = getattr(self, key)
            if not _integer(value) or not 0 <= value < 16:
                raise ValueError(f"simulation.{key} must be an integer in 0..15.")
        if self.initial_direction not in DIRECTIONS:
            raise ValueError(
                f"simulation.initial_direction must be one of {DIRECTIONS}."
            )
        if self.boundary != "toroidal":
            raise ValueError("simulation.boundary must be toroidal.")
        if not _integer(self.ants) or self.ants != 1:
            raise ValueError("simulation.ants must be 1.")
        if self.reset_each_window is not True:
            raise ValueError("simulation.reset_each_window must be true.")
        points = self.checkpoints
        if (
            not isinstance(points, (list, tuple))
            or not points
            or any(not _integer(s) or not 1 <= s <= self.steps for s in points)
            or any(a >= b for a, b in pairwise(points))
            or points[-1] != self.steps
        ):
            raise ValueError(
                "simulation.checkpoints must be increasing positive integers, "
                "ending at simulation.steps."
            )
        object.__setattr__(self, "checkpoints", tuple(points))


DEFAULT_SIMULATION = SimulationConfig()


@dataclass(frozen=True)
class Checkpoint:
    """State after steps transitions; direction codes are N=0, E=1, S=2, W=3."""

    steps: int
    row: int
    col: int
    direction: int
    horizontal_displacement: int
    vertical_displacement: int
    right_turns: int
    unique_cells_visited: int
    black_cells: int
    changed_cells: int
    grid: NDArray[np.uint8]
    visit_counts: NDArray[np.int64]

    @property
    def heading(self) -> str:
        return DIRECTIONS[self.direction]

    @property
    def left_turns(self) -> int:
        return self.steps - self.right_turns

    @property
    def revisits(self) -> int:
        return self.steps - self.unique_cells_visited


@dataclass(frozen=True)
class SimulationResult:
    initial_grid: NDArray[np.uint8]
    config: SimulationConfig
    checkpoints: tuple[Checkpoint, ...]
    # Initial pose, then pose after each move. Flip each departing pose's cell.
    trajectory: tuple[tuple[int, int, int], ...] | None


def simulate(
    grid: NDArray,
    config: SimulationConfig = DEFAULT_SIMULATION,
    *,
    record_trajectory: bool = False,
) -> SimulationResult:
    """READ, TURN, FLIP, MOVE; count reads, not arrivals. Never mutate grid.

    Use native integer lists in the scalar loop. Only checkpoints copy the grid
    and visit matrix. Unwrapped x increases east and unwrapped y increases south.
    """
    initial = np.asarray(grid)
    if initial.shape != (16, 16) or not np.isin(initial, [0, 1]).all():
        raise ValueError("Simulation requires a binary 16 x 16 grid.")
    if not isinstance(record_trajectory, bool):
        raise TypeError("record_trajectory must be a boolean.")
    initial = initial.astype(np.uint8, copy=True)
    cells = initial.tolist()
    visits = [[0] * 16 for _ in range(16)]
    row, col = config.start_row, config.start_col
    direction = DIRECTIONS.index(config.initial_direction)
    x = y = right_turns = unique = completed = 0
    trajectory = [(row, col, direction)] if record_trajectory else None
    checkpoints = []
    for stop in config.checkpoints:
        for _ in range(completed, stop):
            cell = cells[row][col]
            if visits[row][col] == 0:
                unique += 1
            visits[row][col] += 1
            if cell == 0:
                direction = (direction + 1) % 4
                right_turns += 1
            else:
                direction = (direction - 1) % 4
            cells[row][col] = 1 - cell
            dr, dc = DELTAS[direction]
            row = (row + dr) % 16
            col = (col + dc) % 16
            x += dc
            y += dr
            if trajectory is not None:
                trajectory.append((row, col, direction))
        completed = stop
        snapshot = np.array(cells, dtype=np.uint8)
        checkpoints.append(
            Checkpoint(
                steps=stop,
                row=row,
                col=col,
                direction=direction,
                horizontal_displacement=x,
                vertical_displacement=y,
                right_turns=right_turns,
                unique_cells_visited=unique,
                black_cells=int(snapshot.sum()),
                changed_cells=int(np.count_nonzero(snapshot != initial)),
                grid=snapshot,
                visit_counts=np.array(visits, dtype=np.int64),
            )
        )
    return SimulationResult(
        initial,
        config,
        tuple(checkpoints),
        tuple(trajectory) if trajectory is not None else None,
    )
