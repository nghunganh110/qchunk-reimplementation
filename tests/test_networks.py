import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qchunk.networks import MLP, ChunkCritic, CriticEnsemble, FlowVectorField


def test_mlp_preserves_leading_dimensions():
    model = MLP(hidden_dims=(8, 8), output_dim=3)
    inputs = jnp.ones((2, 5, 4))
    variables = model.init(jax.random.key(0), inputs)
    outputs = model.apply(variables, inputs)
    assert outputs.shape == (2, 5, 3)


def test_flow_vector_field_matches_action_chunk_shape():
    batch_size, observation_dim, horizon, action_dim = 4, 6, 3, 2
    chunk_dim = horizon * action_dim
    observations = jnp.ones((batch_size, observation_dim))
    noisy_actions = jnp.ones((batch_size, chunk_dim))
    times = jnp.full((batch_size, 1), 0.5)
    model = FlowVectorField(action_chunk_dim=chunk_dim, hidden_dims=(16, 16))

    variables = model.init(jax.random.key(1), observations, noisy_actions, times)
    velocities = model.apply(variables, observations, noisy_actions, times)

    assert velocities.shape == noisy_actions.shape
    assert bool(jnp.all(jnp.isfinite(velocities)))


def test_flow_vector_field_supports_candidate_axis():
    model = FlowVectorField(action_chunk_dim=6, hidden_dims=(8,))
    observations = jnp.ones((2, 5, 4))
    noisy_actions = jnp.ones((2, 5, 6))
    times = jnp.zeros((2, 5, 1))
    variables = model.init(jax.random.key(2), observations, noisy_actions, times)
    assert model.apply(variables, observations, noisy_actions, times).shape == (2, 5, 6)


def test_chunk_critic_outputs_one_value_per_input():
    model = ChunkCritic(action_chunk_dim=6, hidden_dims=(16, 16))
    observations = jnp.ones((4, 7))
    action_chunks = jnp.ones((4, 6))
    variables = model.init(jax.random.key(3), observations, action_chunks)
    assert model.apply(variables, observations, action_chunks).shape == (4,)


def test_critic_ensemble_has_independent_parameters_and_expected_shape():
    model = CriticEnsemble(action_chunk_dim=6, num_critics=2, hidden_dims=(8,))
    observations = jnp.ones((4, 7))
    action_chunks = jnp.ones((4, 6))
    variables = model.init(jax.random.key(4), observations, action_chunks)
    values = model.apply(variables, observations, action_chunks)

    assert values.shape == (2, 4)
    first_kernel = variables["params"]["VmapChunkCritic_0"]["MLP_0"]["Dense_0"]["kernel"]
    assert first_kernel.shape[0] == 2
    assert not np.array_equal(first_kernel[0], first_kernel[1])


def test_networks_are_differentiable_and_jittable():
    model = FlowVectorField(action_chunk_dim=4, hidden_dims=(8,))
    observations = jnp.ones((3, 5))
    noisy_actions = jnp.ones((3, 4))
    times = jnp.zeros((3, 1))
    variables = model.init(jax.random.key(5), observations, noisy_actions, times)

    def loss(params):
        predictions = model.apply({"params": params}, observations, noisy_actions, times)
        return jnp.mean(predictions**2)

    gradients = jax.jit(jax.grad(loss))(variables["params"])
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(gradients))


@pytest.mark.parametrize(
    ("noisy_shape", "time_shape"),
    [((2, 5), (2, 1)), ((2, 6), (2, 2)), ((3, 6), (2, 1))],
)
def test_flow_vector_field_rejects_invalid_shapes(noisy_shape, time_shape):
    model = FlowVectorField(action_chunk_dim=6, hidden_dims=(8,))
    with pytest.raises(ValueError):
        model.init(
            jax.random.key(6),
            jnp.ones((2, 4)),
            jnp.ones(noisy_shape),
            jnp.ones(time_shape),
        )
