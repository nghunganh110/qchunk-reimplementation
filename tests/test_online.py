from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np
import pytest

from qchunk.config import QCConfig
from qchunk.dataset import TransitionDataset
from qchunk.online import OnlineTrainingConfig, train_online
from qchunk.training import QCTrainState


@dataclass(frozen=True)
class FakeAgentConfig:
    horizon: int = 2
    action_dim: int = 1


@dataclass(frozen=True)
class FakeState:
    config: FakeAgentConfig = FakeAgentConfig()
    updates: int = 0

    def select_action_chunks(self, observations, _rng):
        assert observations.shape == (1, 1)
        return jnp.array([[0.25, -0.5]]), jnp.array([0])

    def update(self, batch):
        assert batch["observations"].shape == (2, 1)
        assert batch["actions"].shape == (2, 2, 1)
        new_state = replace(self, updates=self.updates + 1)
        return new_state, {"loss": jnp.asarray(float(new_state.updates))}


class FakeEnv:
    def __init__(self, episode_length=3, truncate=False):
        self.episode_length = episode_length
        self.truncate = truncate
        self.episode_step = 0
        self.actions = []
        self.reset_seeds = []

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.episode_step = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        self.actions.append(float(action[0]))
        self.episode_step += 1
        done = self.episode_step == self.episode_length
        return (
            np.array([self.episode_step], dtype=np.float32),
            1.0,
            done and not self.truncate,
            done and self.truncate,
            {"success": float(done)},
        )


def capture_fake_env(env):
    return {"episode_step": env.episode_step}


def restore_fake_env(env, state):
    env.reset()
    env.episode_step = int(state["episode_step"])


def make_dataset(size=4):
    return TransitionDataset(
        {
            "observations": np.arange(size, dtype=np.float32)[:, None],
            "actions": np.zeros((size, 1), dtype=np.float32),
            "rewards": np.zeros(size, dtype=np.float32),
            "next_observations": np.arange(1, size + 1, dtype=np.float32)[:, None],
            "masks": np.ones(size, dtype=np.float32),
            "terminals": np.array([0] * (size - 1) + [1], dtype=np.float32),
        }
    )


def test_online_loop_executes_chunks_collects_data_and_updates():
    env = FakeEnv(episode_length=3)
    evaluations = []

    def evaluate_state(step, state):
        evaluations.append((step, state.updates))
        return {"score": float(state.updates)}

    result = train_online(
        FakeState(),
        env,
        make_dataset(),
        OnlineTrainingConfig(
            steps=5,
            buffer_capacity=10,
            batch_size=2,
            start_training=3,
            log_interval=2,
            evaluation_interval=2,
            environment_seed=7,
        ),
        evaluate_fn=evaluate_state,
    )

    assert env.actions == [0.25, -0.5, 0.25, 0.25, -0.5]
    assert env.reset_seeds == [7, None]
    assert result.state.updates == 3
    assert result.environment_steps == 5
    assert [record.step for record in result.records] == [1, 2, 4, 5]
    assert [record.step for record in result.evaluations] == [2, 4, 5]
    assert evaluations == [(2, 0), (4, 2), (5, 3)]
    assert len(result.episodes) == 1
    assert result.episodes[0].episode_return == 3.0
    assert result.episodes[0].length == 3
    assert result.episodes[0].success == 1.0

    data = result.replay_buffer.as_dataset().data
    np.testing.assert_array_equal(data["rewards"][-5:], np.ones(5))
    np.testing.assert_array_equal(data["terminals"][-5:], [0, 0, 1, 0, 0])
    np.testing.assert_array_equal(data["masks"][-5:], [1, 1, 0, 1, 1])


def test_truncation_ends_episode_but_preserves_bootstrap_mask():
    result = train_online(
        FakeState(),
        FakeEnv(episode_length=1, truncate=True),
        make_dataset(),
        OnlineTrainingConfig(
            steps=1,
            buffer_capacity=5,
            batch_size=2,
            start_training=2,
            evaluation_interval=None,
        ),
    )

    data = result.replay_buffer.as_dataset().data
    assert data["terminals"][-1] == 1.0
    assert data["masks"][-1] == 1.0


def test_buffer_expands_to_keep_complete_offline_dataset():
    result = train_online(
        FakeState(),
        FakeEnv(episode_length=10),
        make_dataset(size=4),
        OnlineTrainingConfig(
            steps=1,
            buffer_capacity=2,
            batch_size=2,
            start_training=2,
            evaluation_interval=None,
        ),
    )
    assert result.replay_buffer.capacity == 5
    assert len(result.replay_buffer) == 5


def test_online_loop_updates_real_qc_state():
    state = QCTrainState.create(
        seed=0,
        observation_dim=1,
        config=QCConfig(
            horizon=2,
            action_dim=1,
            hidden_dims=(8,),
            num_candidates=2,
            num_flow_steps=2,
        ),
    )
    result = train_online(
        state,
        FakeEnv(episode_length=3),
        make_dataset(),
        OnlineTrainingConfig(
            steps=2,
            buffer_capacity=6,
            batch_size=2,
            start_training=1,
            evaluation_interval=None,
        ),
    )

    assert int(result.state.flow.step) == 2
    assert int(result.state.critic.step) == 2
    assert all(np.isfinite(value) for value in result.records[-1].metrics.values())


def test_interrupted_online_training_resumes_mid_chunk():
    common = {
        "buffer_capacity": 12,
        "batch_size": 2,
        "start_training": 1,
        "log_interval": 10,
        "evaluation_interval": None,
        "checkpoint_interval": None,
        "replay_seed": 4,
        "policy_seed": 5,
        "environment_seed": 6,
    }
    full_env = FakeEnv(episode_length=4)
    full = train_online(
        FakeState(),
        full_env,
        make_dataset(),
        OnlineTrainingConfig(steps=6, **common),
        capture_environment_state=capture_fake_env,
    )

    first_env = FakeEnv(episode_length=4)
    first = train_online(
        FakeState(),
        first_env,
        make_dataset(),
        OnlineTrainingConfig(steps=3, **common),
        capture_environment_state=capture_fake_env,
    )
    assert first.progress.queued_actions[:, 0].tolist() == [-0.5]

    resumed_env = FakeEnv(episode_length=4)
    resumed = train_online(
        FakeState(),
        resumed_env,
        make_dataset(),
        OnlineTrainingConfig(steps=6, **common),
        resume=first.progress,
        capture_environment_state=capture_fake_env,
        restore_environment_state=restore_fake_env,
    )

    assert first_env.actions + resumed_env.actions == full_env.actions
    assert resumed.state == full.state
    assert resumed.progress.completed_episodes == full.progress.completed_episodes
    assert resumed.progress.update_count == full.progress.update_count
    assert resumed.replay_rng_state == full.replay_rng_state
    np.testing.assert_array_equal(resumed.policy_rng, full.policy_rng)
    for key in TransitionDataset.REQUIRED_FIELDS:
        np.testing.assert_array_equal(
            resumed.replay_buffer.as_dataset().data[key],
            full.replay_buffer.as_dataset().data[key],
        )


def test_checkpoint_callback_requires_environment_capture():
    with pytest.raises(ValueError, match="capture_environment_state"):
        train_online(
            FakeState(),
            FakeEnv(),
            make_dataset(),
            OnlineTrainingConfig(
                steps=1,
                buffer_capacity=5,
                batch_size=2,
                checkpoint_interval=1,
                evaluation_interval=None,
            ),
            on_checkpoint=lambda _progress: None,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"steps": 0},
        {"buffer_capacity": 0},
        {"batch_size": 0},
        {"start_training": 0},
        {"updates_per_step": 0},
        {"log_interval": 0},
        {"evaluation_interval": 0},
        {"checkpoint_interval": 0},
    ],
)
def test_invalid_online_config_is_rejected(overrides):
    values = {
        "steps": 1,
        "buffer_capacity": 5,
        "batch_size": 2,
        "evaluation_interval": None,
    }
    values.update(overrides)
    with pytest.raises(ValueError):
        train_online(
            FakeState(),
            FakeEnv(),
            make_dataset(),
            OnlineTrainingConfig(**values),
        )
