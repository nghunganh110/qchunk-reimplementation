from dataclasses import asdict

import jax
import numpy as np
from flax import serialization

from qchunk.config import QCConfig
from qchunk.dataset import TransitionDataset
from qchunk.online import OnlineTrainingProgress
from qchunk.online_checkpoint import load_online_checkpoint, save_online_checkpoint
from qchunk.replay_buffer import ReplayBuffer
from qchunk.training import QCTrainState


def make_state(seed=0):
    return QCTrainState.create(
        seed=seed,
        observation_dim=2,
        config=QCConfig(
            horizon=2,
            action_dim=1,
            hidden_dims=(8,),
            num_candidates=2,
            num_flow_steps=2,
        ),
    )


def make_buffer():
    dataset = TransitionDataset(
        {
            "observations": np.array([[0, 0], [1, -1]], dtype=np.float32),
            "actions": np.array([[0.1], [0.2]], dtype=np.float32),
            "rewards": np.array([0, 1], dtype=np.float32),
            "next_observations": np.array([[1, -1], [2, -2]], dtype=np.float32),
            "masks": np.array([1, 0], dtype=np.float32),
            "terminals": np.array([0, 1], dtype=np.float32),
        }
    )
    return ReplayBuffer.from_dataset(dataset, capacity=5)


def trees_equal(left, right):
    left = serialization.to_state_dict(left)
    right = serialization.to_state_dict(right)
    comparisons = jax.tree.map(
        lambda x, y: np.array_equal(np.asarray(x), np.asarray(y)), left, right
    )
    return all(jax.tree.leaves(comparisons))


def test_online_checkpoint_round_trip(tmp_path):
    state = make_state()
    replay_rng = np.random.default_rng(8)
    replay_rng.integers(100, size=4)
    progress = OnlineTrainingProgress(
        state=state,
        replay_buffer=make_buffer(),
        environment_steps=17,
        policy_rng=jax.random.PRNGKey(9),
        replay_rng_state=replay_rng.bit_generator.state,
        observation=np.array([2.0, -2.0], dtype=np.float32),
        queued_actions=np.array([[0.3]], dtype=np.float32),
        episode_return=4.5,
        episode_length=3,
        episode_success=1.0,
        completed_episodes=2,
        update_count=13,
        environment_state={"physics": np.array([1.0, 2.0]), "elapsed_steps": 3},
    )

    path = save_online_checkpoint(tmp_path / "online.msgpack", progress)
    assert path.is_dir()
    assert (path / "metadata.msgpack").is_file()
    assert (path / "replay" / "observations.npy").is_file()
    restored = load_online_checkpoint(path, make_state(seed=999))

    assert trees_equal(restored.state, progress.state)
    assert restored.environment_steps == 17
    assert restored.replay_rng_state == progress.replay_rng_state
    assert asdict(restored.state.config) == asdict(progress.state.config)
    np.testing.assert_array_equal(restored.policy_rng, progress.policy_rng)
    np.testing.assert_array_equal(restored.observation, progress.observation)
    np.testing.assert_array_equal(restored.queued_actions, progress.queued_actions)
    np.testing.assert_array_equal(
        restored.environment_state["physics"], progress.environment_state["physics"]
    )
    for key in TransitionDataset.REQUIRED_FIELDS:
        np.testing.assert_array_equal(
            restored.replay_buffer.as_dataset().data[key],
            progress.replay_buffer.as_dataset().data[key],
        )
