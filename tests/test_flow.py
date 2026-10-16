import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qchunk.flow import (
    euler_sample,
    flatten_action_chunks,
    flow_matching_loss,
    flow_path,
    masked_flow_mse,
    sample_flow_actions,
)
from qchunk.networks import FlowVectorField


def test_flatten_action_chunks_preserves_values_and_leading_axes():
    chunks = jnp.arange(2 * 3 * 4).reshape(2, 3, 4)
    flattened = flatten_action_chunks(chunks)
    assert flattened.shape == (2, 12)
    np.testing.assert_array_equal(flattened[0], jnp.arange(12))


def test_flow_path_has_correct_endpoints_and_velocity():
    target = jnp.array([[1.0, 3.0]])
    noise = jnp.array([[-1.0, 1.0]])

    at_zero, velocity = flow_path(target, noise, jnp.array([[0.0]]))
    at_one, _ = flow_path(target, noise, jnp.array([[1.0]]))
    at_half, _ = flow_path(target, noise, jnp.array([[0.5]]))

    np.testing.assert_allclose(at_zero, noise)
    np.testing.assert_allclose(at_one, target)
    np.testing.assert_allclose(at_half, (target + noise) / 2)
    np.testing.assert_allclose(velocity, target - noise)


def test_masked_flow_mse_ignores_padded_actions():
    prediction = jnp.zeros((1, 6))
    target = jnp.array([[1.0, 1.0, 2.0, 2.0, 100.0, 100.0]])
    valid = jnp.array([[1.0, 1.0, 0.0]])

    loss = masked_flow_mse(prediction, target, valid, action_dim=2)

    # The reference implementation averages over all six coordinates after
    # masking, so the two invalid errors contribute zero to the numerator.
    np.testing.assert_allclose(loss, (1 + 1 + 4 + 4) / 6)


def zero_vector_field(_variables, _observations, actions, _times):
    return jnp.zeros_like(actions)


def constant_vector_field(_variables, _observations, actions, _times):
    return jnp.full_like(actions, 0.4)


def test_euler_zero_field_keeps_noise_unchanged():
    observations = jnp.ones((2, 3))
    noise = jnp.array([[0.1, -0.2], [0.3, -0.4]])
    actions = euler_sample(zero_vector_field, {}, observations, noise, num_steps=10)
    np.testing.assert_allclose(actions, noise)


def test_euler_constant_field_has_known_solution():
    observations = jnp.ones((2, 3))
    noise = jnp.zeros((2, 4))
    actions = euler_sample(
        constant_vector_field,
        {},
        observations,
        noise,
        num_steps=5,
        clip_actions=False,
    )
    np.testing.assert_allclose(actions, jnp.full_like(noise, 0.4), rtol=1e-6)


def test_euler_clips_only_the_final_actions():
    observations = jnp.ones((1, 3))
    noise = jnp.full((1, 2), 0.9)
    actions = euler_sample(constant_vector_field, {}, observations, noise, num_steps=2)
    np.testing.assert_allclose(actions, [[1.0, 1.0]])


def test_sample_flow_actions_is_reproducible_and_supports_candidate_axis():
    observations = jnp.ones((2, 5, 3))
    key = jax.random.key(10)
    first = sample_flow_actions(zero_vector_field, {}, observations, key, 4)
    second = sample_flow_actions(zero_vector_field, {}, observations, key, 4)
    assert first.shape == (2, 5, 4)
    np.testing.assert_array_equal(first, second)
    assert bool(jnp.all(first >= -1.0) & jnp.all(first <= 1.0))


def test_flow_matching_loss_is_differentiable():
    observations = jnp.ones((4, 3))
    chunks = jnp.ones((4, 2, 2))
    valid = jnp.array([[1, 1], [1, 1], [1, 0], [1, 0]], dtype=jnp.float32)
    model = FlowVectorField(action_chunk_dim=4, hidden_dims=(8, 8))
    variables = model.init(
        jax.random.key(11),
        observations,
        jnp.ones((4, 4)),
        jnp.ones((4, 1)),
    )

    def loss(params):
        return flow_matching_loss(
            model.apply,
            params,
            observations,
            chunks,
            valid,
            jax.random.key(12),
        )

    value, gradients = jax.jit(jax.value_and_grad(loss))(variables["params"])
    assert value.shape == ()
    assert bool(jnp.isfinite(value))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(gradients))


@pytest.mark.parametrize("num_steps", [0, -1, 1.5])
def test_euler_rejects_invalid_step_count(num_steps):
    with pytest.raises(ValueError):
        euler_sample(
            zero_vector_field,
            {},
            jnp.ones((1, 3)),
            jnp.ones((1, 2)),
            num_steps=num_steps,
        )
