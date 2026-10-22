"""Known feature mathematics, stable schema, finite values, and replay parity."""

from dataclasses import replace

import numpy as np
import pytest

from anttrader.features import (
    FEATURE_NAMES,
    checkpoint_features,
    extract_features,
    feature_columns,
)
from anttrader.simulation import SimulationConfig, simulate


def test_known_eight_step_features():
    config = SimulationConfig(steps=8, checkpoints=(8,))
    state = simulate(np.zeros((16, 16)), config).checkpoints[0]
    assert checkpoint_features(state) == {
        "final_row": 15,
        "final_col": 0,
        "heading_north": 0,
        "heading_east": 0,
        "heading_south": 1,
        "heading_west": 0,
        "unique_cells_visited": 7,
        "revisit_fraction": 1 / 8,
        "horizontal_displacement": 0,
        "vertical_displacement": 0,
        "right_turn_fraction": 7 / 8,
        "final_black_fraction": 6 / 256,
        "changed_cell_fraction": 6 / 256,
        "visit_entropy": pytest.approx(2.75 / 8),
        "max_visit_concentration": 1 / 4,
    }
    assert FEATURE_NAMES == tuple(checkpoint_features(state))


@pytest.mark.parametrize(
    "occupied,entropy,concentration", [(1, 0, 1), (256, 1, 1 / 256)]
)
def test_entropy_endpoints(occupied, entropy, concentration):
    state = simulate(np.zeros((16, 16))).checkpoints[0]
    visits = np.zeros((16, 16), dtype=np.int64)
    visits.flat[:occupied] = 256 // occupied
    features = checkpoint_features(replace(state, visit_counts=visits))
    assert features["visit_entropy"] == pytest.approx(entropy)
    assert features["max_visit_concentration"] == concentration


@pytest.mark.parametrize("direction", ["north", "east", "south", "west"])
def test_heading_displacement_and_one_hot(direction):
    config = SimulationConfig(steps=1, checkpoints=(1,), initial_direction=direction)
    state = simulate(np.zeros((16, 16)), config).checkpoints[0]
    values = checkpoint_features(state)
    assert sum(values[f"heading_{h}"] for h in ("north", "east", "south", "west")) == 1
    assert values[f"heading_{state.heading}"] == 1
    assert values["horizontal_displacement"] == state.horizontal_displacement
    assert values["vertical_displacement"] == state.vertical_displacement


def test_feature_schema_bounds_and_recording_parity():
    grid = np.random.default_rng(13).integers(0, 2, (16, 16), dtype=np.uint8)
    features = extract_features(simulate(grid))
    assert features == extract_features(simulate(grid, record_trajectory=True))
    assert list(features) == feature_columns(SimulationConfig())
    assert len(features) == 60
    assert list(features)[:2] == ["s0256_final_row", "s0256_final_col"]
    assert list(features)[-1] == "s2048_max_visit_concentration"
    assert np.isfinite(list(features.values())).all()
    for name, value in features.items():
        if "fraction" in name or "concentration" in name or "entropy" in name:
            assert -1e-15 <= value <= 1 + 1e-15
        if "unique_cells_visited" in name:
            assert 1 <= value <= 256


def test_zero_step_features_rejected():
    state = simulate(np.zeros((16, 16))).checkpoints[0]
    with pytest.raises(ValueError, match="at least one"):
        checkpoint_features(replace(state, steps=0))
