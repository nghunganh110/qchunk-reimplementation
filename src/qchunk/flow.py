from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

Array = jax.Array
ApplyFn = Callable[[dict[str, Any], Array, Array, Array], Array]


def flatten_action_chunks(action_chunks: Array) -> Array:
    """Flatten the final horizon and action axes: [..., H, A] -> [..., H*A]."""
    if action_chunks.ndim < 3:
        raise ValueError("action_chunks must have shape [..., H, A]")
    return action_chunks.reshape((*action_chunks.shape[:-2], -1))


def flow_path(target_actions: Array, noise: Array, times: Array) -> tuple[Array, Array]:
    """Construct the linear flow path and its constant target velocity."""
    if target_actions.shape != noise.shape:
        raise ValueError("target_actions and noise must have identical shapes")
    if times.shape != (*target_actions.shape[:-1], 1):
        raise ValueError("times must have shape [..., 1] matching the action leading axes")

    interpolated_actions = (1.0 - times) * noise + times * target_actions
    target_velocity = target_actions - noise
    return interpolated_actions, target_velocity


def masked_flow_mse(
    predicted_velocity: Array,
    target_velocity: Array,
    valid: Array,
    action_dim: int,
) -> Array:
    """Mean squared velocity error with invalid chunk positions zeroed out.

    The denominator remains the full B*H*A tensor size, matching the reference
    implementation. Therefore, an incomplete chunk contributes proportionally
    less loss than a complete chunk.
    """
    if predicted_velocity.shape != target_velocity.shape:
        raise ValueError("predicted_velocity and target_velocity must have identical shapes")
    if action_dim <= 0 or predicted_velocity.shape[-1] % action_dim != 0:
        raise ValueError("action_dim must divide the flattened action dimension")

    horizon = predicted_velocity.shape[-1] // action_dim
    expected_valid_shape = (*predicted_velocity.shape[:-1], horizon)
    if valid.shape != expected_valid_shape:
        raise ValueError(f"valid must have shape {expected_valid_shape}, got {valid.shape}")

    squared_error = jnp.square(predicted_velocity - target_velocity)
    coordinate_valid = jnp.repeat(valid, action_dim, axis=-1)
    return jnp.mean(squared_error * coordinate_valid)


def flow_matching_loss(
    apply_fn: ApplyFn,
    params: Any,
    observations: Array,
    action_chunks: Array,
    valid: Array,
    rng: Array,
) -> Array:
    """Compute the flow-matching objective for a batch of action chunks."""
    if observations.shape[:-1] != action_chunks.shape[:-2]:
        raise ValueError("observations and action_chunks must share batch dimensions")
    if valid.shape != action_chunks.shape[:-1]:
        raise ValueError("valid must match action_chunks without the action dimension")

    flat_actions = flatten_action_chunks(action_chunks)
    noise_rng, time_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, flat_actions.shape, dtype=flat_actions.dtype)
    times = jax.random.uniform(
        time_rng,
        (*flat_actions.shape[:-1], 1),
        dtype=flat_actions.dtype,
    )
    interpolated_actions, target_velocity = flow_path(flat_actions, noise, times)
    predicted_velocity = apply_fn({"params": params}, observations, interpolated_actions, times)
    return masked_flow_mse(
        predicted_velocity,
        target_velocity,
        valid,
        action_dim=action_chunks.shape[-1],
    )


def euler_sample(
    apply_fn: ApplyFn,
    params: Any,
    observations: Array,
    noise: Array,
    num_steps: int = 10,
    clip_actions: bool = True,
) -> Array:
    """Integrate the learned vector field from time zero to one."""
    if not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")
    if observations.shape[:-1] != noise.shape[:-1]:
        raise ValueError("observations and noise must share leading dimensions")

    actions = noise
    step_size = 1.0 / num_steps
    for step in range(num_steps):
        times = jnp.full(
            (*actions.shape[:-1], 1),
            step * step_size,
            dtype=actions.dtype,
        )
        velocity = apply_fn({"params": params}, observations, actions, times)
        if velocity.shape != actions.shape:
            raise ValueError("the vector field output must match the noise shape")
        actions = actions + step_size * velocity

    if clip_actions:
        actions = jnp.clip(actions, -1.0, 1.0)
    return actions


def sample_flow_actions(
    apply_fn: ApplyFn,
    params: Any,
    observations: Array,
    rng: Array,
    action_chunk_dim: int,
    num_steps: int = 10,
    clip_actions: bool = True,
) -> Array:
    """Sample Gaussian noise and transform it into flattened action chunks."""
    if not isinstance(action_chunk_dim, int) or action_chunk_dim <= 0:
        raise ValueError("action_chunk_dim must be a positive integer")
    noise = jax.random.normal(
        rng,
        (*observations.shape[:-1], action_chunk_dim),
        dtype=observations.dtype,
    )
    return euler_sample(
        apply_fn,
        params,
        observations,
        noise,
        num_steps=num_steps,
        clip_actions=clip_actions,
    )
