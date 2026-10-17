from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training import train_state

from qchunk.agent import (
    aggregate_q_values,
    best_of_n_action_chunks,
    critic_mse_loss,
    critic_td_target,
    soft_update,
)
from qchunk.config import QCConfig
from qchunk.dataset import ChunkBatch
from qchunk.flow import flatten_action_chunks, flow_matching_loss
from qchunk.networks import CriticEnsemble, FlowVectorField

Array = jax.Array
JaxBatch = dict[str, Array]


def batch_to_jax(batch: ChunkBatch) -> JaxBatch:
    """Move a NumPy ChunkBatch into JAX arrays."""
    return {
        "observations": jnp.asarray(batch.observations),
        "actions": jnp.asarray(batch.actions),
        "rewards": jnp.asarray(batch.rewards),
        "next_observations": jnp.asarray(batch.next_observations),
        "masks": jnp.asarray(batch.masks),
        "valid": jnp.asarray(batch.valid),
    }


@struct.dataclass
class QCTrainState:
    """All mutable state required for one QC training process."""

    rng: Array
    flow: train_state.TrainState
    critic: train_state.TrainState
    target_critic_params: Any
    config: QCConfig = struct.field(pytree_node=False)

    @classmethod
    def create(cls, seed: int, observation_dim: int, config: QCConfig) -> "QCTrainState":
        """Initialize networks, Adam optimizers, target parameters, and RNG."""
        config.validate()
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive")

        # Flax serialization cannot currently encode JAX's typed PRNG keys.
        # The legacy uint32 representation has identical random semantics and
        # allows the complete training state to be checkpointed.
        rng = jax.random.PRNGKey(seed)
        flow_rng, critic_rng, state_rng = jax.random.split(rng, 3)
        observations = jnp.zeros((1, observation_dim), dtype=jnp.float32)
        action_chunks = jnp.zeros((1, config.action_chunk_dim), dtype=jnp.float32)
        times = jnp.zeros((1, 1), dtype=jnp.float32)

        flow_model = FlowVectorField(
            action_chunk_dim=config.action_chunk_dim,
            hidden_dims=config.hidden_dims,
            layer_norm=config.flow_layer_norm,
        )
        critic_model = CriticEnsemble(
            action_chunk_dim=config.action_chunk_dim,
            num_critics=config.num_critics,
            hidden_dims=config.hidden_dims,
            layer_norm=config.critic_layer_norm,
        )
        flow_params = flow_model.init(flow_rng, observations, action_chunks, times)["params"]
        critic_params = critic_model.init(critic_rng, observations, action_chunks)["params"]

        flow_tx = optax.adam(config.learning_rate)
        critic_tx = optax.adam(config.learning_rate)
        return cls(
            rng=state_rng,
            flow=train_state.TrainState.create(
                apply_fn=flow_model.apply,
                params=flow_params,
                tx=flow_tx,
            ),
            critic=train_state.TrainState.create(
                apply_fn=critic_model.apply,
                params=critic_params,
                tx=critic_tx,
            ),
            target_critic_params=critic_params,
            config=config,
        )

    @jax.jit
    def select_action_chunks(self, observations: Array, rng: Array) -> tuple[Array, Array]:
        """Select one best-of-N flattened action chunk per observation."""
        return best_of_n_action_chunks(
            self.flow.apply_fn,
            self.flow.params,
            self.critic.apply_fn,
            self.critic.params,
            observations,
            rng,
            action_chunk_dim=self.config.action_chunk_dim,
            num_candidates=self.config.num_candidates,
            num_flow_steps=self.config.num_flow_steps,
            q_aggregation=self.config.q_aggregation,
        )

    @jax.jit
    def update(self, batch: JaxBatch) -> tuple["QCTrainState", dict[str, Array]]:
        """Apply one flow update, one critic update, and one target update."""
        expected_action_shape = (self.config.horizon, self.config.action_dim)
        if batch["actions"].shape[-2:] != expected_action_shape:
            raise ValueError(
                f"actions must end with shape {expected_action_shape}, "
                f"got {batch['actions'].shape[-2:]}"
            )

        next_rng, flow_rng, candidate_rng = jax.random.split(self.rng, 3)

        def flow_loss_fn(params):
            return flow_matching_loss(
                self.flow.apply_fn,
                params,
                batch["observations"],
                batch["actions"],
                batch["valid"],
                flow_rng,
            )

        flow_loss, flow_grads = jax.value_and_grad(flow_loss_fn)(self.flow.params)
        new_flow = self.flow.apply_gradients(grads=flow_grads)

        next_observations = batch["next_observations"][..., -1, :]
        next_actions, _ = best_of_n_action_chunks(
            self.flow.apply_fn,
            self.flow.params,
            self.critic.apply_fn,
            self.critic.params,
            next_observations,
            candidate_rng,
            action_chunk_dim=self.config.action_chunk_dim,
            num_candidates=self.config.num_candidates,
            num_flow_steps=self.config.num_flow_steps,
            q_aggregation=self.config.q_aggregation,
        )
        target_q_ensemble = self.critic.apply_fn(
            {"params": self.target_critic_params},
            next_observations,
            next_actions,
        )
        next_q = aggregate_q_values(target_q_ensemble, self.config.q_aggregation)
        target_q = critic_td_target(
            batch["rewards"],
            batch["masks"],
            batch["valid"],
            next_q,
            self.config.discount,
        )
        flat_actions = flatten_action_chunks(batch["actions"])

        def critic_loss_fn(params):
            predicted_q = self.critic.apply_fn(
                {"params": params}, batch["observations"], flat_actions
            )
            loss = critic_mse_loss(predicted_q, target_q, batch["valid"])
            return loss, predicted_q

        (critic_loss, predicted_q), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
            self.critic.params
        )
        new_critic = self.critic.apply_gradients(grads=critic_grads)
        new_target_critic_params = soft_update(
            self.target_critic_params,
            new_critic.params,
            self.config.tau,
        )

        new_state = self.replace(
            rng=next_rng,
            flow=new_flow,
            critic=new_critic,
            target_critic_params=new_target_critic_params,
        )
        metrics = {
            "flow_loss": flow_loss,
            "critic_loss": critic_loss,
            "q_mean": jnp.mean(predicted_q),
            "target_q_mean": jnp.mean(target_q),
            "complete_fraction": jnp.mean(batch["valid"][..., -1]),
        }
        return new_state, metrics
