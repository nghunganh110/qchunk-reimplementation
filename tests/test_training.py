import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qchunk.config import QCConfig
from qchunk.dataset import TransitionDataset
from qchunk.training import QCTrainState, batch_to_jax


def make_batch(batch_size=8, horizon=2, action_dim=2, observation_dim=3):
    size = max(batch_size + horizon, 12)
    rng = np.random.default_rng(30)
    dataset = TransitionDataset(
        {
            "observations": rng.normal(size=(size, observation_dim)),
            "actions": np.tanh(rng.normal(size=(size, action_dim))),
            "rewards": rng.normal(size=size),
            "next_observations": rng.normal(size=(size, observation_dim)),
            "masks": np.ones(size),
            "terminals": np.zeros(size),
        }
    )
    starts = np.arange(batch_size)
    return dataset.get_chunks(starts, horizon)


def tree_allclose(left, right):
    comparisons = jax.tree.map(lambda x, y: np.allclose(np.asarray(x), np.asarray(y)), left, right)
    return all(jax.tree.leaves(comparisons))


def tree_any_different(left, right):
    comparisons = jax.tree.map(
        lambda x, y: not np.array_equal(np.asarray(x), np.asarray(y)), left, right
    )
    return any(jax.tree.leaves(comparisons))


def small_config(**overrides):
    values = {
        "horizon": 2,
        "action_dim": 2,
        "hidden_dims": (16, 16),
        "num_critics": 2,
        "num_candidates": 4,
        "num_flow_steps": 2,
        "learning_rate": 1e-3,
        "tau": 0.25,
    }
    values.update(overrides)
    return QCConfig(**values)


def test_create_initializes_matching_online_and_target_critics():
    state = QCTrainState.create(seed=0, observation_dim=3, config=small_config())
    assert state.flow.step == 0
    assert state.critic.step == 0
    assert tree_allclose(state.critic.params, state.target_critic_params)


def test_batch_to_jax_preserves_shapes_and_values():
    numpy_batch = make_batch()
    batch = batch_to_jax(numpy_batch)
    assert batch["observations"].shape == (8, 3)
    assert batch["actions"].shape == (8, 2, 2)
    assert isinstance(batch["actions"], jax.Array)
    np.testing.assert_allclose(batch["actions"], numpy_batch.actions)


def test_update_changes_online_networks_and_polyak_updates_target():
    config = small_config()
    state = QCTrainState.create(seed=1, observation_dim=3, config=config)
    old_flow_params = state.flow.params
    old_critic_params = state.critic.params
    old_target_params = state.target_critic_params

    new_state, metrics = state.update(batch_to_jax(make_batch()))

    assert new_state.flow.step == 1
    assert new_state.critic.step == 1
    assert tree_any_different(old_flow_params, new_state.flow.params)
    assert tree_any_different(old_critic_params, new_state.critic.params)
    assert not np.array_equal(
        np.asarray(jax.random.key_data(state.rng)),
        np.asarray(jax.random.key_data(new_state.rng)),
    )
    assert set(metrics) == {
        "flow_loss",
        "critic_loss",
        "q_mean",
        "target_q_mean",
        "complete_fraction",
    }
    assert all(bool(jnp.isfinite(value)) for value in metrics.values())

    expected_target = jax.tree.map(
        lambda old, online: (1 - config.tau) * old + config.tau * online,
        old_target_params,
        new_state.critic.params,
    )
    assert tree_allclose(new_state.target_critic_params, expected_target)


def test_update_is_reusable_after_jit_compilation():
    state = QCTrainState.create(seed=2, observation_dim=3, config=small_config())
    batch = batch_to_jax(make_batch())
    state, first_metrics = state.update(batch)
    state, second_metrics = state.update(batch)
    assert state.flow.step == 2
    assert state.critic.step == 2
    assert all(bool(jnp.isfinite(value)) for value in first_metrics.values())
    assert all(bool(jnp.isfinite(value)) for value in second_metrics.values())


@pytest.mark.parametrize(
    "overrides",
    [
        {"horizon": 0},
        {"action_dim": 0},
        {"hidden_dims": ()},
        {"num_critics": 0},
        {"num_candidates": 0},
        {"num_flow_steps": 0},
        {"discount": 1.1},
        {"tau": -0.1},
        {"learning_rate": 0.0},
        {"q_aggregation": "median"},
    ],
)
def test_invalid_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        QCTrainState.create(seed=0, observation_dim=3, config=small_config(**overrides))
