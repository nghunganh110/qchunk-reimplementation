import numpy as np
import pytest

from qchunk.dataset import TransitionDataset


def make_dataset():
    # Two episodes: [0, 1, 2] and [3, 4, 5]. The latter ends in a time limit.
    return TransitionDataset(
        {
            "observations": np.arange(6)[:, None],
            "actions": np.arange(10, 16)[:, None],
            "rewards": np.arange(1, 7),
            "next_observations": np.arange(1, 7)[:, None],
            "masks": np.array([1, 1, 0, 1, 1, 1]),
            "terminals": np.array([0, 0, 1, 0, 0, 1]),
        }
    )


def test_complete_chunk_and_discounted_return():
    batch = make_dataset().get_chunks(np.array([0]), horizon=3)
    np.testing.assert_array_equal(batch.actions[0, :, 0], [10, 11, 12])
    np.testing.assert_array_equal(batch.valid, [[1, 1, 1]])
    np.testing.assert_array_equal(batch.next_observations[0, -1], [3])
    returns = (batch.rewards * batch.valid * 0.9 ** np.arange(3)).sum(axis=1)
    np.testing.assert_allclose(returns, [1 + 0.9 * 2 + 0.9**2 * 3])
    np.testing.assert_array_equal(batch.masks.prod(axis=1), [0])


def test_padding_never_uses_next_episode():
    batch = make_dataset().get_chunks(np.array([1, 2]), horizon=4)
    np.testing.assert_array_equal(batch.valid, [[1, 1, 0, 0], [1, 0, 0, 0]])
    np.testing.assert_array_equal(batch.actions[:, :, 0], [[11, 12, 0, 0], [12, 0, 0, 0]])
    np.testing.assert_array_equal(batch.rewards, [[2, 3, 0, 0], [3, 0, 0, 0]])
    np.testing.assert_array_equal(batch.masks, [[1, 0, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(batch.next_observations[:, :, 0], [[2, 3, 3, 3], [3, 3, 3, 3]])


def test_trajectory_boundary_preserves_bootstrap_but_stops_chunk():
    dataset = make_dataset()
    batch = dataset.get_chunks(np.array([3]), horizon=3)
    np.testing.assert_array_equal(batch.valid, [[1, 1, 1]])
    np.testing.assert_array_equal(batch.masks.prod(axis=1), [1])
    # Move the trajectory boundary earlier while retaining bootstrap eligibility.
    dataset.data["terminals"][4] = 1
    batch = dataset.get_chunks(np.array([3]), horizon=3)
    np.testing.assert_array_equal(batch.actions[0, :, 0], [13, 14, 0])
    np.testing.assert_array_equal(batch.valid, [[1, 1, 0]])
    np.testing.assert_array_equal(batch.masks, [[1, 1, 0]])


def test_dataset_end_without_terminal_is_still_padded():
    dataset = make_dataset()
    dataset.data["terminals"][-1] = 0
    batch = dataset.get_chunks(np.array([5]), horizon=3)
    np.testing.assert_array_equal(batch.valid, [[1, 0, 0]])
    np.testing.assert_array_equal(batch.actions[0, :, 0], [15, 0, 0])
    np.testing.assert_array_equal(batch.next_observations[0, :, 0], [6, 6, 6])


def test_horizon_one_includes_terminal_transition():
    batch = make_dataset().get_chunks(np.array([2]), horizon=1)
    np.testing.assert_array_equal(batch.valid, [[1]])
    np.testing.assert_array_equal(batch.actions, [[[12]]])
    np.testing.assert_array_equal(batch.masks, [[0]])


def test_random_sampling_is_reproducible():
    dataset = make_dataset()
    first = dataset.sample_chunks(8, 3, np.random.default_rng(42))
    second = dataset.sample_chunks(8, 3, np.random.default_rng(42))
    np.testing.assert_array_equal(first.observations, second.observations)
    np.testing.assert_array_equal(first.actions, second.actions)
    assert first.actions.shape == (8, 3, 1)


@pytest.mark.parametrize("starts", [[], [[0]], [0.5], [-1], [6]])
def test_invalid_starts(starts):
    with pytest.raises(ValueError):
        make_dataset().get_chunks(np.asarray(starts), horizon=3)


@pytest.mark.parametrize("horizon", [0, -1, 1.5])
def test_invalid_horizon(horizon):
    with pytest.raises(ValueError):
        make_dataset().get_chunks(np.array([0]), horizon)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5])
def test_invalid_batch_size(batch_size):
    with pytest.raises(ValueError):
        make_dataset().sample_chunks(batch_size, 3, np.random.default_rng(0))
