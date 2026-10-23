"""Independent short trajectories, toroidal movement, checkpoints, and replay."""

from dataclasses import replace

import numpy as np
import pytest

from anttrader.simulation import SimulationConfig, simulate


def short_config(steps=1, **kwargs):
    return SimulationConfig(steps=steps, checkpoints=(steps,), **kwargs)


@pytest.mark.parametrize(
    "cell, col, heading, dx", [(0, 1, "east", 1), (1, 15, "west", -1)]
)
def test_approved_first_step_and_input_copy(cell, col, heading, dx):
    grid = np.zeros((16, 16), dtype=np.uint8)
    grid[15, 0] = cell
    original = grid.copy()
    result = simulate(grid, short_config())
    state = result.checkpoints[0]
    assert (state.row, state.col, state.heading) == (15, col, heading)
    assert state.grid[15, 0] == 1 - cell
    assert state.grid[15, col] == 0  # Flip the departing cell, not the arrival.
    assert state.horizontal_displacement == dx
    assert state.vertical_displacement == 0
    assert state.right_turns == 1 - cell
    assert state.left_turns == cell
    assert state.visit_counts.sum() == state.unique_cells_visited == 1
    assert state.visit_counts[15, 0] == 1
    np.testing.assert_array_equal(grid, original)
    np.testing.assert_array_equal(result.initial_grid, original)
    assert result.trajectory is None


@pytest.mark.parametrize(
    "row,col,direction,expected,displacement",
    [
        (0, 5, "west", (15, 5, "north"), (0, -1)),
        (15, 5, "east", (0, 5, "south"), (0, 1)),
        (5, 0, "south", (5, 15, "west"), (-1, 0)),
        (5, 15, "north", (5, 0, "east"), (1, 0)),
    ],
)
def test_wrap_all_edges(row, col, direction, expected, displacement):
    config = short_config(start_row=row, start_col=col, initial_direction=direction)
    state = simulate(np.zeros((16, 16)), config).checkpoints[0]
    assert (state.row, state.col, state.heading) == expected
    assert (state.horizontal_displacement, state.vertical_displacement) == displacement


def test_hand_calculated_eight_steps_and_replay():
    # Four right turns return to the start. The fifth read flips it back to zero.
    expected_poses = (
        (15, 0, 0),
        (15, 1, 1),
        (0, 1, 2),
        (0, 0, 3),
        (15, 0, 0),
        (15, 15, 3),
        (14, 15, 0),
        (14, 0, 1),
        (15, 0, 2),
    )
    config = short_config(8)
    grid = np.zeros((16, 16), dtype=np.uint8)
    result = simulate(grid, config, record_trajectory=True)
    assert result.trajectory == expected_poses
    state = result.checkpoints[0]
    expected_grid = np.zeros_like(grid)
    for position in [(15, 1), (0, 1), (0, 0), (15, 15), (14, 15), (14, 0)]:
        expected_grid[position] = 1
    np.testing.assert_array_equal(state.grid, expected_grid)
    assert (state.right_turns, state.left_turns, state.unique_cells_visited) == (
        7,
        1,
        7,
    )
    assert state.revisits == 1
    assert (state.horizontal_displacement, state.vertical_displacement) == (0, 0)
    assert state.black_cells == state.changed_cells == 6
    assert state.visit_counts[15, 0] == 2
    assert np.count_nonzero(state.visit_counts) == 7
    assert state.visit_counts.sum() == 8
    replay = result.initial_grid.copy()
    for row, col, _ in result.trajectory[:-1]:
        replay[row, col] ^= 1
    np.testing.assert_array_equal(replay, state.grid)


def test_continuous_checkpoints_determinism_and_independence():
    grid = np.random.default_rng(42).integers(0, 2, (16, 16), dtype=np.uint8)
    original = grid.copy()
    full = simulate(grid)
    recorded = simulate(grid, record_trajectory=True)
    assert [s.steps for s in full.checkpoints] == [256, 512, 1024, 2048]
    for state, replayed in zip(full.checkpoints, recorded.checkpoints, strict=True):
        shorter = simulate(grid, short_config(state.steps)).checkpoints[0]
        for other in (shorter, replayed):
            for name in state.__dataclass_fields__:
                np.testing.assert_array_equal(
                    getattr(state, name), getattr(other, name)
                )
        assert state.visit_counts.sum() == state.steps
        assert state.unique_cells_visited == np.count_nonzero(state.visit_counts)
        assert state.revisits == state.steps - state.unique_cells_visited
        assert state.right_turns + state.left_turns == state.steps
        assert 0 <= state.row < 16 and 0 <= state.col < 16
        assert state.row == (15 + state.vertical_displacement) % 16
        assert state.col == state.horizontal_displacement % 16
        assert state.black_cells == state.grid.sum()
        assert state.changed_cells == np.count_nonzero(state.grid != grid)
        assert recorded.trajectory[state.steps] == (
            state.row,
            state.col,
            state.direction,
        )
    assert all(0 <= r < 16 and 0 <= c < 16 for r, c, _ in recorded.trajectory)
    reconstructed = recorded.initial_grid.copy()
    for row, col, _ in recorded.trajectory[:-1]:
        reconstructed[row, col] ^= 1
    np.testing.assert_array_equal(reconstructed, recorded.checkpoints[-1].grid)
    assert not np.shares_memory(full.checkpoints[0].grid, full.checkpoints[1].grid)
    full.checkpoints[0].grid[:] = 9
    full.checkpoints[0].visit_counts[:] = 0
    full.initial_grid[:] = 9
    np.testing.assert_array_equal(grid, original)
    np.testing.assert_array_equal(recorded.initial_grid, original)
    np.testing.assert_array_equal(simulate(grid).checkpoints[-1].grid, reconstructed)


@pytest.mark.parametrize(
    "changes",
    [
        {"steps": 0},
        {"steps": True},
        {"steps": 1.5},
        {"start_row": -1},
        {"start_col": False},
        {"initial_direction": "up"},
        {"boundary": "closed"},
        {"ants": 2},
        {"ants": True},
        {"reset_each_window": False},
        {"reset_each_window": 1},
        {"checkpoints": []},
        {"checkpoints": [256, 256, 2048]},
        {"checkpoints": [512, 256, 2048]},
        {"checkpoints": [0, 2048]},
        {"checkpoints": [256, 2049]},
        {"checkpoints": [256, 512]},
        {"checkpoints": [True, 2048]},
        {"checkpoints": [256.0, 2048]},
    ],
)
def test_invalid_config(changes):
    with pytest.raises(ValueError, match="simulation"):
        replace(SimulationConfig(), **changes)


@pytest.mark.parametrize(
    "grid",
    [
        np.zeros((4, 4)),
        np.zeros((16, 16, 1)),
        np.full((16, 16), 2),
        np.full((16, 16), np.nan),
        np.full((16, 16), -1),
        np.full((16, 16), "0"),
    ],
)
def test_invalid_input_grid(grid):
    with pytest.raises(ValueError, match="binary rectangular|starting position"):
        simulate(grid)
