import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qchunk.agent import best_of_n_action_chunks
from qchunk.refinement import (
    nearest_candidate_directions,
    project_onto_direction_span,
    refine_action_chunks,
    select_geometric_refined_action_chunks,
)


def linear_critic(_variables, _observations, actions):
    values = actions[..., 0] + 0.5 * actions[..., 1]
    return jnp.stack((values, values), axis=0)


def test_refinement_improves_known_q_within_action_and_trust_region():
    initial = jnp.array([[0.0, 0.0], [1.2, -0.5]])
    refined = refine_action_chunks(
        linear_critic,
        {},
        jnp.zeros((2, 3)),
        initial,
        steps=3,
        step_size=0.2,
        radius=0.3,
    )
    center = np.clip(np.asarray(initial), -1, 1)
    actual = np.asarray(refined)
    assert np.all(actual <= 1.0) and np.all(actual >= -1.0)
    assert np.all(np.linalg.norm(actual - center, axis=-1) <= 0.300001)
    assert np.all(actual[:, 0] + 0.5 * actual[:, 1] > center[:, 0] + 0.5 * center[:, 1])


def test_refinement_rejects_invalid_radius():
    with pytest.raises(ValueError):
        refine_action_chunks(
            linear_critic,
            {},
            jnp.zeros((1, 3)),
            jnp.zeros((1, 2)),
            steps=2,
            step_size=0.1,
            radius=0.0,
        )


def test_nearest_directions_exclude_selected_and_project_gradient():
    candidates = jnp.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [9.0, 0.0]]])
    directions = nearest_candidate_directions(
        candidates, jnp.array([[0.0, 0.0]]), jnp.array([0]), neighbors=2
    )
    np.testing.assert_allclose(directions, [[[1.0, 0.0], [0.0, 2.0]]])
    projected = project_onto_direction_span(
        jnp.array([[1.0, 100.0]]), jnp.array([[[1.0, 0.0]]])
    )
    np.testing.assert_allclose(projected[0, 0], 1.0, atol=1e-4)
    np.testing.assert_allclose(projected[0, 1], 0.0, atol=1e-6)


def test_geometric_refinement_changes_only_supported_direction():
    refined = refine_action_chunks(
        linear_critic,
        {},
        jnp.zeros((1, 3)),
        jnp.zeros((1, 2)),
        steps=2,
        step_size=0.2,
        radius=0.3,
        directions=jnp.array([[[1.0, 0.0]]]),
    )
    np.testing.assert_allclose(refined, [[0.3, 0.0]], atol=1e-6)


def test_geometric_selection_uses_same_qc_candidates_and_respects_radius():
    def zero_flow(_variables, _observations, actions, _times):
        return jnp.zeros_like(actions)

    observations = jnp.zeros((2, 3))
    rng = jax.random.PRNGKey(7)
    common = (zero_flow, {}, linear_critic, {}, observations, rng)
    baseline, baseline_indices = best_of_n_action_chunks(
        *common, action_chunk_dim=2, num_candidates=8, num_flow_steps=2
    )
    refined, geometric_indices = select_geometric_refined_action_chunks(
        *common,
        action_chunk_dim=2,
        num_candidates=8,
        num_flow_steps=2,
        neighbors=3,
        steps=2,
        step_size=0.05,
        radius=0.1,
    )
    np.testing.assert_array_equal(geometric_indices, baseline_indices)
    assert np.all(np.linalg.norm(np.asarray(refined - baseline), axis=-1) <= 0.100001)
    assert np.all(np.asarray(refined) <= 1.0) and np.all(np.asarray(refined) >= -1.0)
