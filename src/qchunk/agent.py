from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp

from qchunk.flow import sample_flow_actions

Array = jax.Array
QAggregation = Literal["mean", "min"]


def discounted_chunk_return(rewards: Array, valid: Array, discount: float) -> Array:
    """Compute a discounted return along the final horizon axis."""
    if rewards.shape != valid.shape or rewards.ndim < 2:
        raise ValueError("rewards and valid must have identical shape [..., H]")
    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must lie in [0, 1]")

    powers = jnp.power(discount, jnp.arange(rewards.shape[-1], dtype=rewards.dtype))
    return jnp.sum(rewards * valid * powers, axis=-1)


def chunk_bootstrap_mask(masks: Array, valid: Array) -> Array:
    """Return the bootstrap mask for each complete chunk.

    Incomplete chunks return zero because their padded masks are not meaningful
    TD endpoints. The critic loss separately excludes those chunks.
    """
    if masks.shape != valid.shape or masks.ndim < 2:
        raise ValueError("masks and valid must have identical shape [..., H]")
    complete = valid[..., -1]
    return complete * jnp.prod(masks, axis=-1)


def critic_td_target(
    rewards: Array,
    masks: Array,
    valid: Array,
    next_q: Array,
    discount: float,
) -> Array:
    """Compute R_H + gamma**H * mask * Q_target at the next chunk state."""
    expected_shape = rewards.shape[:-1]
    if masks.shape != rewards.shape or valid.shape != rewards.shape:
        raise ValueError("rewards, masks, and valid must have identical shapes")
    if next_q.shape != expected_shape:
        raise ValueError(f"next_q must have shape {expected_shape}, got {next_q.shape}")

    returns = discounted_chunk_return(rewards, valid, discount)
    bootstrap = chunk_bootstrap_mask(masks, valid)
    horizon = rewards.shape[-1]
    target = returns + discount**horizon * bootstrap * next_q
    return jax.lax.stop_gradient(target)


def aggregate_q_values(q_values: Array, method: QAggregation = "mean") -> Array:
    """Aggregate the leading critic-ensemble axis."""
    if q_values.ndim < 2:
        raise ValueError("q_values must have shape [K, ...]")
    if method == "mean":
        return jnp.mean(q_values, axis=0)
    if method == "min":
        return jnp.min(q_values, axis=0)
    raise ValueError(f"unsupported Q aggregation: {method}")


def critic_mse_loss(predicted_q: Array, target_q: Array, valid: Array) -> Array:
    """Mean TD error across critics and batches, excluding incomplete chunks."""
    if predicted_q.ndim < 2:
        raise ValueError("predicted_q must have shape [K, ...]")
    if predicted_q.shape[1:] != target_q.shape:
        raise ValueError("target_q must match predicted_q without the ensemble axis")
    if valid.shape[:-1] != target_q.shape:
        raise ValueError("valid batch dimensions must match target_q")

    complete = valid[..., -1]
    squared_error = jnp.square(predicted_q - target_q[None, ...])
    return jnp.mean(squared_error * complete[None, ...])


def select_best_action_chunks(
    candidate_chunks: Array,
    candidate_q: Array,
    aggregation: QAggregation = "mean",
) -> tuple[Array, Array]:
    """Select the highest-valued candidate along the penultimate axis.

    candidate_chunks has shape [..., N, H*A], while candidate_q has shape
    [K, ..., N]. The returned chunks have shape [..., H*A].
    """
    if candidate_chunks.ndim < 3:
        raise ValueError("candidate_chunks must have shape [..., N, H*A]")
    if candidate_q.shape[1:] != candidate_chunks.shape[:-1]:
        raise ValueError("candidate_q must have shape [K, ..., N]")

    scores = aggregate_q_values(candidate_q, aggregation)
    indices = jnp.argmax(scores, axis=-1)
    gather_indices = indices[..., None, None]
    selected = jnp.take_along_axis(candidate_chunks, gather_indices, axis=-2)
    return selected.squeeze(-2), indices


def best_of_n_action_chunks(
    flow_apply_fn: Callable,
    flow_params: Any,
    critic_apply_fn: Callable,
    critic_params: Any,
    observations: Array,
    rng: Array,
    action_chunk_dim: int,
    num_candidates: int,
    num_flow_steps: int = 10,
    q_aggregation: QAggregation = "mean",
) -> tuple[Array, Array]:
    """Generate N behavior chunks and return the candidate with maximum Q."""
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
    return select_best_action_chunks(candidates, candidate_q, q_aggregation)


def sample_and_score_action_chunks(
    flow_apply_fn: Callable,
    flow_params: Any,
    critic_apply_fn: Callable,
    critic_params: Any,
    observations: Array,
    rng: Array,
    action_chunk_dim: int,
    num_candidates: int,
    num_flow_steps: int = 10,
) -> tuple[Array, Array]:
    """Expose the QC candidate set and critic scores for inference experiments."""
    if not isinstance(num_candidates, int) or num_candidates <= 0:
        raise ValueError("num_candidates must be a positive integer")

    candidate_observations = jnp.broadcast_to(
        observations[..., None, :],
        (*observations.shape[:-1], num_candidates, observations.shape[-1]),
    )
    candidates = sample_flow_actions(
        flow_apply_fn,
        flow_params,
        candidate_observations,
        rng,
        action_chunk_dim,
        num_steps=num_flow_steps,
    )
    candidate_q = critic_apply_fn({"params": critic_params}, candidate_observations, candidates)
    return candidates, candidate_q


def soft_update(target_params: Any, online_params: Any, tau: float) -> Any:
    """Polyak-update target parameters toward online parameters."""
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must lie in [0, 1]")
    return jax.tree.map(
        lambda target, online: (1.0 - tau) * target + tau * online,
        target_params,
        online_params,
    )
