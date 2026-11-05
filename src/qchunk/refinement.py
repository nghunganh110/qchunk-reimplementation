"""Inference-only, locally constrained gradient refinement of QC action chunks."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from qchunk.agent import (
    QAggregation,
    aggregate_q_values,
    best_of_n_action_chunks,
    sample_and_score_action_chunks,
    select_best_action_chunks,
)

Array = jax.Array


def refine_action_chunks(
    critic_apply_fn: Callable,
    critic_params: Any,
    observations: Array,
    initial_chunks: Array,
    *,
    steps: int,
    step_size: float,
    radius: float,
    q_aggregation: QAggregation = "mean",
    directions: Array | None = None,
) -> Array:
    """Ascend the critic inside an L2 ball around the selected, clipped chunk.

    The radius is for the *whole flattened chunk*. Both the center and every
    iterate lie in the environment's [-1, 1] action box. No model parameters
    or optimizer state are changed.
    """
    if steps <= 0 or step_size <= 0 or radius <= 0:
        raise ValueError("steps, step_size and radius must be positive")
    if observations.shape[:-1] != initial_chunks.shape[:-1]:
        raise ValueError("observations and initial_chunks must share batch dimensions")
    if directions is not None and (
        directions.shape[:-2] != initial_chunks.shape[:-1]
        or directions.shape[-1] != initial_chunks.shape[-1]
        or directions.shape[-2] == 0
    ):
        raise ValueError("directions must have shape [..., k, action_chunk_dim] with k > 0")

    center = jnp.clip(initial_chunks, -1.0, 1.0)

    def q_sum(chunks: Array) -> Array:
        q_values = critic_apply_fn({"params": critic_params}, observations, chunks)
        return jnp.sum(aggregate_q_values(q_values, q_aggregation))

    def update(_index: int, chunks: Array) -> Array:
        gradient = jax.grad(q_sum)(chunks)
        if directions is not None:
            gradient = project_onto_direction_span(gradient, directions)
        direction = gradient / jnp.maximum(
            jnp.linalg.norm(gradient, axis=-1, keepdims=True), 1e-8
        )
        if directions is not None:
            direction = jnp.where(
                jnp.linalg.norm(gradient, axis=-1, keepdims=True) > 1e-6,
                direction,
                0.0,
            )
        proposed = jnp.clip(chunks + step_size * direction, -1.0, 1.0)
        delta = proposed - center
        scale = jnp.minimum(
            1.0, radius / jnp.maximum(jnp.linalg.norm(delta, axis=-1, keepdims=True), 1e-8)
        )
        return center + scale * delta

    return jax.lax.fori_loop(0, steps, update, center)


def nearest_candidate_directions(
    candidates: Array, selected: Array, selected_indices: Array, *, neighbors: int
) -> Array:
    """Directions from the QC-selected chunk to its nearest flow samples."""
    if candidates.ndim < 3 or candidates.shape[:-2] != selected.shape[:-1]:
        raise ValueError("candidates and selected must share batch dimensions")
    if candidates.shape[-1] != selected.shape[-1]:
        raise ValueError("candidate and selected action dimensions must match")
    if selected_indices.shape != selected.shape[:-1]:
        raise ValueError("selected_indices must match the batch dimensions")
    if not 0 < neighbors < candidates.shape[-2]:
        raise ValueError("neighbors must be between 1 and N-1")

    differences = candidates - selected[..., None, :]
    distances = jnp.sum(jnp.square(differences), axis=-1)
    is_selected = jnp.arange(candidates.shape[-2]) == selected_indices[..., None]
    distances = jnp.where(is_selected, jnp.inf, distances)
    _, neighbor_indices = jax.lax.top_k(-distances, neighbors)
    return jnp.take_along_axis(differences, neighbor_indices[..., None], axis=-2)


def project_onto_direction_span(gradient: Array, directions: Array) -> Array:
    """Ridge-stabilized projection onto the span of nearby flow chunks."""
    if directions.shape[:-2] != gradient.shape[:-1] or directions.shape[-1] != gradient.shape[-1]:
        raise ValueError("directions and gradient must share batch and action dimensions")
    k = directions.shape[-2]
    if k == 0:
        raise ValueError("directions must contain at least one vector")
    gram = jnp.einsum("...kd,...ld->...kl", directions, directions)
    rhs = jnp.einsum("...kd,...d->...k", directions, gradient)
    ridge_scale = 1e-5 * jnp.maximum(
        jnp.trace(gram, axis1=-2, axis2=-1) / k, 1.0
    )
    regularized = gram + ridge_scale[..., None, None] * jnp.eye(k, dtype=gram.dtype)
    coefficients = jnp.linalg.solve(regularized, rhs[..., None]).squeeze(-1)
    return jnp.einsum("...kd,...k->...d", directions, coefficients)


def select_refined_action_chunks(
    flow_apply_fn: Callable,
    flow_params: Any,
    critic_apply_fn: Callable,
    critic_params: Any,
    observations: Array,
    rng: Array,
    *,
    action_chunk_dim: int,
    num_candidates: int,
    num_flow_steps: int,
    steps: int,
    step_size: float,
    radius: float,
    q_aggregation: QAggregation = "mean",
) -> tuple[Array, Array]:
    """Apply local Q ascent to the same best-of-N choice used by QC."""
    selected, indices = best_of_n_action_chunks(
        flow_apply_fn,
        flow_params,
        critic_apply_fn,
        critic_params,
        observations,
        rng,
        action_chunk_dim=action_chunk_dim,
        num_candidates=num_candidates,
        num_flow_steps=num_flow_steps,
        q_aggregation=q_aggregation,
    )
    refined = refine_action_chunks(
        critic_apply_fn,
        critic_params,
        observations,
        selected,
        steps=steps,
        step_size=step_size,
        radius=radius,
        q_aggregation=q_aggregation,
    )
    return refined, indices


def select_geometric_refined_action_chunks(
    flow_apply_fn: Callable,
    flow_params: Any,
    critic_apply_fn: Callable,
    critic_params: Any,
    observations: Array,
    rng: Array,
    *,
    action_chunk_dim: int,
    num_candidates: int,
    num_flow_steps: int,
    neighbors: int,
    steps: int,
    step_size: float,
    radius: float,
    q_aggregation: QAggregation = "mean",
) -> tuple[Array, Array]:
    """Refine QC's selected chunk only along nearby flow-sample directions."""
    if not 0 < neighbors < num_candidates:
        raise ValueError("neighbors must be between 1 and N-1")
    candidates, candidate_q = sample_and_score_action_chunks(
        flow_apply_fn,
        flow_params,
        critic_apply_fn,
        critic_params,
        observations,
        rng,
        action_chunk_dim=action_chunk_dim,
        num_candidates=num_candidates,
        num_flow_steps=num_flow_steps,
    )
    selected, indices = select_best_action_chunks(candidates, candidate_q, q_aggregation)
    directions = nearest_candidate_directions(
        candidates, selected, indices, neighbors=neighbors
    )
    refined = refine_action_chunks(
        critic_apply_fn,
        critic_params,
        observations,
        selected,
        steps=steps,
        step_size=step_size,
        radius=radius,
        q_aggregation=q_aggregation,
        directions=directions,
    )
    return refined, indices
