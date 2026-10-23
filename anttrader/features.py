"""Numerical ant behavior only: no labels, predictions, or trading logic."""

from collections.abc import Iterable, Iterator

import numpy as np

from anttrader.grid import HistoricalGrid
from anttrader.simulation import (
    DEFAULT_SIMULATION,
    DIRECTIONS,
    Checkpoint,
    SimulationConfig,
    SimulationResult,
    simulate,
)

FEATURE_NAMES = (
    "final_row",
    "final_col",
    "heading_north",
    "heading_east",
    "heading_south",
    "heading_west",
    "unique_cells_visited",
    "revisit_fraction",
    "horizontal_displacement",
    "vertical_displacement",
    "right_turn_fraction",
    "final_black_fraction",
    "changed_cell_fraction",
    "visit_entropy",
    "max_visit_concentration",
)


def checkpoint_features(state: Checkpoint) -> dict[str, int | float]:
    """Use cumulative read probabilities and natural-log entropy / ln(cell count)."""
    if state.steps < 1:
        raise ValueError("Feature extraction requires at least one completed step.")
    probabilities = state.visit_counts[state.visit_counts > 0] / state.steps
    cells = state.grid.size
    entropy = (
        -float(np.sum(probabilities * np.log(probabilities))) / np.log(cells)
        if cells > 1
        else 0.0
    )
    values = (
        state.row,
        state.col,
        *(int(state.heading == heading) for heading in DIRECTIONS),
        state.unique_cells_visited,
        state.revisits / state.steps,
        state.horizontal_displacement,
        state.vertical_displacement,
        state.right_turns / state.steps,
        state.black_cells / cells,
        state.changed_cells / cells,
        float(entropy),
        int(state.visit_counts.max()) / state.steps,
    )
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite feature at checkpoint {state.steps}.")
    return dict(zip(FEATURE_NAMES, values, strict=True))


def feature_columns(config: SimulationConfig) -> list[str]:
    return [
        f"s{step:04d}_{name}" for step in config.checkpoints for name in FEATURE_NAMES
    ]


def extract_features(result: SimulationResult) -> dict[str, int | float]:
    """Stable ascending checkpoint prefixes; omit human-readable heading in CSV."""
    return {
        f"s{state.steps:04d}_{name}": value
        for state in result.checkpoints
        for name, value in checkpoint_features(state).items()
    }


def iter_feature_rows(
    grids: Iterable[HistoricalGrid], config: SimulationConfig = DEFAULT_SIMULATION
) -> Iterator[dict[str, str | int | float]]:
    """One independent simulation per window; fail with its timestamp on error."""
    previous = None
    for window in grids:
        timestamp = window.decision_timestamp
        try:
            if timestamp.tzinfo is None or (
                previous is not None and timestamp <= previous
            ):
                raise ValueError(
                    "Decision timestamps must be timezone-aware and increasing."
                )
            features = extract_features(simulate(window.grid, config))
        except Exception as exc:
            raise ValueError(
                f"Feature simulation failed at {timestamp}: {exc}"
            ) from exc
        yield {
            "decision_timestamp": timestamp.tz_convert("UTC").isoformat(),
            **features,
        }
        previous = timestamp
