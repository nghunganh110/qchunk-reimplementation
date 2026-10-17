from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest

from qchunk.evaluation import EvaluationConfig, evaluate, rollout_episode


@dataclass(frozen=True)
class FakeConfig:
    horizon: int = 2
    action_dim: int = 1


class FakeState:
    config = FakeConfig()

    def __init__(self):
        self.calls = 0

    def select_action_chunks(self, observations, _rng):
        self.calls += 1
        assert observations.shape == (1, 1)
        return jnp.array([[0.25, -0.5]]), jnp.array([0])


class FakeEnv:
    def __init__(self, terminal_step=3, success=True):
        self.terminal_step = terminal_step
        self.success = success
        self.actions = []
        self.reset_seeds = []
        self.steps = 0

    def reset(self, seed):
        self.reset_seeds.append(seed)
        self.steps = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        self.actions.append(float(action[0]))
        self.steps += 1
        terminated = self.steps == self.terminal_step
        info = {"success": float(self.success and terminated)}
        return (
            np.array([self.steps], dtype=np.float32),
            1.0,
            terminated,
            False,
            info,
        )


def test_rollout_executes_chunks_open_loop_and_stops_mid_chunk():
    state = FakeState()
    env = FakeEnv(terminal_step=3)
    result, _ = rollout_episode(
        state,
        env,
        jnp.array([0, 1], dtype=jnp.uint32),
        reset_seed=7,
    )

    assert state.calls == 2
    assert env.actions == [0.25, -0.5, 0.25]
    assert result.length == 3
    assert result.episode_return == 3.0
    assert result.success == 1.0
    assert result.chunks_sampled == 2


def test_evaluate_aggregates_episode_metrics_and_seeds_resets():
    state = FakeState()
    env = FakeEnv(terminal_step=2, success=False)
    result = evaluate(state, env, EvaluationConfig(episodes=3, seed=10))

    assert result.success_rate == 0.0
    assert result.mean_return == 2.0
    assert result.mean_length == 2.0
    assert env.reset_seeds == [10, 11, 12]


def test_rollout_respects_explicit_step_limit():
    result, _ = rollout_episode(
        FakeState(),
        FakeEnv(terminal_step=100),
        jnp.array([0, 1], dtype=jnp.uint32),
        reset_seed=0,
        max_episode_steps=3,
    )
    assert result.length == 3
    assert result.success == 0.0


@pytest.mark.parametrize(
    "config",
    [EvaluationConfig(episodes=0), EvaluationConfig(max_episode_steps=0)],
)
def test_invalid_evaluation_config_is_rejected(config):
    with pytest.raises(ValueError):
        evaluate(FakeState(), FakeEnv(), config)
