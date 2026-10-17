import numpy as np
import pytest

from qchunk.config import QCConfig
from qchunk.dataset import TransitionDataset
from qchunk.replay_buffer import ReplayBuffer
from qchunk.training import QCTrainState, batch_to_jax


def transition(index, *, terminal=False, mask=1.0):
    return {
        "observations": np.array([index, -index], dtype=np.float32),
        "actions": np.array([index / 10], dtype=np.float32),
        "rewards": float(index),
        "next_observations": np.array([index + 1, -index - 1], dtype=np.float32),
        "masks": mask,
        "terminals": float(terminal),
    }


def make_dataset(size=4):
    transitions = [transition(index, terminal=index == size - 1) for index in range(size)]
    return TransitionDataset(
        {
            key: np.stack([item[key] for item in transitions])
            if key in ("observations", "actions", "next_observations")
            else np.array([item[key] for item in transitions])
            for key in TransitionDataset.REQUIRED_FIELDS
        }
    )


def test_add_and_convert_to_chronological_dataset():
    buffer = ReplayBuffer(capacity=4, observation_dim=2, action_dim=1)
    for index in range(3):
        buffer.add_transition(transition(index, terminal=index == 2, mask=index != 2))

    dataset = buffer.as_dataset()
    assert len(buffer) == 3
    assert buffer.pointer == 3
    np.testing.assert_array_equal(dataset.data["observations"][:, 0], [0, 1, 2])
    np.testing.assert_array_equal(dataset.data["masks"], [1, 1, 0])
    np.testing.assert_array_equal(dataset.data["terminals"], [0, 0, 1])


def test_wraparound_overwrites_oldest_and_preserves_chunk_order():
    buffer = ReplayBuffer(capacity=4, observation_dim=2, action_dim=1)
    for index in range(6):
        buffer.add_transition(
            transition(index, terminal=index in (3, 5), mask=float(index not in (3, 5)))
        )

    dataset = buffer.as_dataset()
    np.testing.assert_array_equal(dataset.data["observations"][:, 0], [2, 3, 4, 5])

    batch = buffer.get_chunks(np.array([0, 2]), horizon=3)
    np.testing.assert_allclose(batch.actions[:, :, 0], [[0.2, 0.3, 0.0], [0.4, 0.5, 0.0]])
    np.testing.assert_array_equal(batch.valid, [[1, 1, 0], [1, 1, 0]])
    np.testing.assert_array_equal(batch.next_observations[:, :, 0], [[3, 4, 4], [5, 6, 6]])


def test_create_from_dataset_then_overwrite_oldest():
    buffer = ReplayBuffer.from_dataset(make_dataset(), capacity=4)
    buffer.add_transition(transition(4, terminal=True, mask=0.0))

    dataset = buffer.as_dataset()
    np.testing.assert_array_equal(dataset.data["observations"][:, 0], [1, 2, 3, 4])
    assert buffer.pointer == 1


def test_initial_dataset_end_is_boundary_before_online_data():
    dataset = make_dataset()
    dataset.data["terminals"][-1] = 0.0
    buffer = ReplayBuffer.from_dataset(dataset, capacity=5)
    buffer.add_transition(transition(4))

    batch = buffer.get_chunks(np.array([3]), horizon=2)
    np.testing.assert_array_equal(batch.valid, [[1, 0]])
    np.testing.assert_array_equal(batch.masks, [[1, 0]])


def test_sampling_is_reproducible_after_wraparound():
    buffer = ReplayBuffer(capacity=4, observation_dim=2, action_dim=1)
    for index in range(6):
        buffer.add_transition(transition(index))

    first = buffer.sample_chunks(10, 2, np.random.default_rng(42))
    second = buffer.sample_chunks(10, 2, np.random.default_rng(42))
    np.testing.assert_array_equal(first.observations, second.observations)
    np.testing.assert_array_equal(first.actions, second.actions)


def test_clear_makes_buffer_empty_and_reusable():
    buffer = ReplayBuffer.from_dataset(make_dataset(), capacity=5)
    buffer.clear()
    assert len(buffer) == 0
    with pytest.raises(ValueError, match="empty"):
        buffer.sample_chunks(1, 2, np.random.default_rng(0))

    buffer.add_transition(transition(9))
    np.testing.assert_array_equal(buffer.as_dataset().data["observations"][:, 0], [9])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"capacity": 0, "observation_dim": 2, "action_dim": 1}, "capacity"),
        ({"capacity": 2, "observation_dim": 0, "action_dim": 1}, "observation_dim"),
        ({"capacity": 2, "observation_dim": 2, "action_dim": 0}, "action_dim"),
    ],
)
def test_invalid_dimensions_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ReplayBuffer(**kwargs)


def test_invalid_transition_is_rejected_without_advancing_pointer():
    buffer = ReplayBuffer(capacity=2, observation_dim=2, action_dim=1)
    bad_transition = transition(0)
    bad_transition["actions"] = np.zeros(2)
    with pytest.raises(ValueError, match="actions"):
        buffer.add_transition(bad_transition)
    assert len(buffer) == 0
    assert buffer.pointer == 0


def test_initial_dataset_must_fit_capacity():
    with pytest.raises(ValueError, match="capacity"):
        ReplayBuffer.from_dataset(make_dataset(), capacity=3)


def test_sampled_replay_chunk_can_update_qc_state():
    buffer = ReplayBuffer.from_dataset(make_dataset(), capacity=5)
    batch = buffer.sample_chunks(4, 2, np.random.default_rng(0))
    state = QCTrainState.create(
        seed=0,
        observation_dim=2,
        config=QCConfig(
            horizon=2,
            action_dim=1,
            hidden_dims=(8,),
            num_candidates=2,
            num_flow_steps=2,
        ),
    )

    state, metrics = state.update(batch_to_jax(batch))

    assert int(state.flow.step) == 1
    assert all(np.isfinite(float(value)) for value in metrics.values())
