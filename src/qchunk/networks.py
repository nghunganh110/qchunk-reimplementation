from collections.abc import Callable, Sequence

import jax.numpy as jnp
from flax import linen as nn

Array = jnp.ndarray


class MLP(nn.Module):
    """A feed-forward network whose final layer is linear."""

    hidden_dims: Sequence[int]
    output_dim: int
    layer_norm: bool = False
    activation: Callable[[Array], Array] = nn.gelu

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        x = inputs
        for width in self.hidden_dims:
            x = nn.Dense(width)(x)
            x = self.activation(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
        return nn.Dense(self.output_dim)(x)


class FlowVectorField(nn.Module):
    """Predict velocity in flattened action-chunk space.

    observations has shape [..., S], noisy_actions [..., H*A], and times
    [..., 1]. The output has the same shape as noisy_actions.
    """

    action_chunk_dim: int
    hidden_dims: Sequence[int] = (512, 512, 512, 512)
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations: Array, noisy_actions: Array, times: Array) -> Array:
        if noisy_actions.shape[-1] != self.action_chunk_dim:
            raise ValueError(
                "noisy_actions last dimension must equal action_chunk_dim "
                f"({self.action_chunk_dim}), got {noisy_actions.shape[-1]}"
            )
        if times.shape[-1] != 1:
            raise ValueError("times must have last dimension 1")
        if not (observations.shape[:-1] == noisy_actions.shape[:-1] == times.shape[:-1]):
            raise ValueError("observations, noisy_actions, and times must share leading dimensions")

        inputs = jnp.concatenate((observations, noisy_actions, times), axis=-1)
        return MLP(
            hidden_dims=self.hidden_dims,
            output_dim=self.action_chunk_dim,
            layer_norm=self.layer_norm,
        )(inputs)


class ChunkCritic(nn.Module):
    """Estimate Q(s, a_t, ..., a_{t+H-1}) for one critic."""

    action_chunk_dim: int
    hidden_dims: Sequence[int] = (512, 512, 512, 512)
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations: Array, action_chunks: Array) -> Array:
        if action_chunks.shape[-1] != self.action_chunk_dim:
            raise ValueError(
                "action_chunks last dimension must equal action_chunk_dim "
                f"({self.action_chunk_dim}), got {action_chunks.shape[-1]}"
            )
        if observations.shape[:-1] != action_chunks.shape[:-1]:
            raise ValueError("observations and action_chunks must share leading dimensions")

        inputs = jnp.concatenate((observations, action_chunks), axis=-1)
        values = MLP(
            hidden_dims=self.hidden_dims,
            output_dim=1,
            layer_norm=self.layer_norm,
        )(inputs)
        return values.squeeze(-1)


class CriticEnsemble(nn.Module):
    """An ensemble of independently initialized chunk critics.

    The ensemble axis is first, so the result has shape [K, ...].
    """

    action_chunk_dim: int
    num_critics: int = 2
    hidden_dims: Sequence[int] = (512, 512, 512, 512)
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations: Array, action_chunks: Array) -> Array:
        if self.num_critics <= 0:
            raise ValueError("num_critics must be positive")

        ensemble_type = nn.vmap(
            ChunkCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=(None, None),
            out_axes=0,
            axis_size=self.num_critics,
        )
        ensemble = ensemble_type(
            action_chunk_dim=self.action_chunk_dim,
            hidden_dims=self.hidden_dims,
            layer_norm=self.layer_norm,
        )
        return ensemble(observations, action_chunks)
