import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qchunk.agent import (
    aggregate_q_values,
    best_of_n_action_chunks,
    chunk_bootstrap_mask,
    critic_mse_loss,
    critic_td_target,
    discounted_chunk_return,
    select_best_action_chunks,
    soft_update,
)
from qchunk.flow import sample_flow_actions


def test_discounted_chunk_return_uses_discount_and_valid():
    rewards = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 100.0]])
    valid = jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    returns = discounted_chunk_return(rewards, valid, discount=0.9)
    np.testing.assert_allclose(returns, [1 + 0.9 * 2 + 0.9**2 * 3, 4 + 0.9 * 5])


def test_bootstrap_mask_distinguishes_success_boundary_and_incomplete_chunk():
    masks = jnp.array(
        [
            [1.0, 1.0, 0.0],  # task success disables bootstrap
            [1.0, 1.0, 1.0],  # trajectory boundary may still bootstrap
            [1.0, 0.0, 0.0],  # incomplete padded chunk
        ]
    )
    valid = jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(chunk_bootstrap_mask(masks, valid), [0, 1, 0])


def test_critic_target_uses_gamma_to_horizon_power():
    rewards = jnp.array([[1.0, 2.0, 3.0]])
    masks = jnp.ones_like(rewards)
    valid = jnp.ones_like(rewards)
    target = critic_td_target(rewards, masks, valid, jnp.array([10.0]), discount=0.5)
    expected = 1 + 0.5 * 2 + 0.5**2 * 3 + 0.5**3 * 10
    np.testing.assert_allclose(target, [expected])


def test_critic_target_stops_gradient_through_next_q():
    rewards = jnp.ones((1, 2))
    masks = jnp.ones_like(rewards)
    valid = jnp.ones_like(rewards)

    gradient = jax.grad(
        lambda next_q: critic_td_target(rewards, masks, valid, next_q, discount=0.9).sum()
    )(jnp.array([3.0]))

    np.testing.assert_array_equal(gradient, [0.0])


def test_critic_loss_ignores_incomplete_chunks():
    predictions = jnp.array([[2.0, 100.0], [4.0, -100.0]])
    targets = jnp.array([1.0, 0.0])
    valid = jnp.array([[1.0, 1.0], [1.0, 0.0]])
    loss = critic_mse_loss(predictions, targets, valid)
    # Four K*B entries remain in the denominator, matching the reference code.
    np.testing.assert_allclose(loss, ((2 - 1) ** 2 + (4 - 1) ** 2) / 4)


def test_q_aggregation():
    values = jnp.array([[1.0, 5.0], [3.0, 2.0]])
    np.testing.assert_allclose(aggregate_q_values(values, "mean"), [2.0, 3.5])
    np.testing.assert_allclose(aggregate_q_values(values, "min"), [1.0, 2.0])
    with pytest.raises(ValueError):
        aggregate_q_values(values, "median")


def test_select_best_action_chunks_uses_ensemble_mean():
    candidates = jnp.array(
        [
            [[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]],
            [[40.0, 0.0], [50.0, 0.0], [60.0, 0.0]],
        ]
    )
    q_values = jnp.array(
        [
            [[1.0, 5.0, 3.0], [8.0, 1.0, 2.0]],
            [[3.0, 1.0, 3.0], [2.0, 3.0, 4.0]],
        ]
    )
    selected, indices = select_best_action_chunks(candidates, q_values)
    np.testing.assert_array_equal(indices, [1, 0])
    np.testing.assert_array_equal(selected, [[20.0, 0.0], [40.0, 0.0]])


def zero_vector_field(_variables, _observations, actions, _times):
    return jnp.zeros_like(actions)


def sum_critic(_variables, _observations, actions):
    scores = jnp.sum(actions, axis=-1)
    return jnp.stack((scores, scores + 1.0), axis=0)


def test_best_of_n_matches_explicit_candidate_ranking():
    observations = jnp.ones((3, 4))
    key = jax.random.key(20)
    selected, indices = best_of_n_action_chunks(
        zero_vector_field,
        {},
        sum_critic,
        {},
        observations,
        key,
        action_chunk_dim=6,
        num_candidates=5,
        num_flow_steps=2,
    )

    repeated_observations = jnp.broadcast_to(observations[:, None, :], (3, 5, 4))
    candidates = sample_flow_actions(
        zero_vector_field, {}, repeated_observations, key, 6, num_steps=2
    )
    expected_indices = jnp.argmax(jnp.sum(candidates, axis=-1), axis=-1)
    expected = jnp.take_along_axis(candidates, expected_indices[:, None, None], axis=1).squeeze(1)

    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(selected, expected)


def test_soft_update_endpoints_and_interpolation():
    target = {"weight": jnp.array([0.0, 2.0])}
    online = {"weight": jnp.array([10.0, 6.0])}
    np.testing.assert_allclose(soft_update(target, online, 0.0)["weight"], [0, 2])
    np.testing.assert_allclose(soft_update(target, online, 1.0)["weight"], [10, 6])
    np.testing.assert_allclose(soft_update(target, online, 0.25)["weight"], [2.5, 3])
