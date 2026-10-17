from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from qchunk.training import QCTrainState


@dataclass(frozen=True)
class EvaluationConfig:
    episodes: int = 50
    seed: int = 42
    max_episode_steps: int | None = None

    def validate(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.max_episode_steps is not None and self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive when provided")


@dataclass(frozen=True)
class EpisodeResult:
    episode_return: float
    length: int
    success: float
    chunks_sampled: int


@dataclass(frozen=True)
class EvaluationResult:
    episodes: tuple[EpisodeResult, ...]
    success_rate: float
    mean_return: float
    mean_length: float

    def as_metrics(self) -> dict[str, float]:
        return {
            "success_rate": self.success_rate,
            "mean_return": self.mean_return,
            "mean_length": self.mean_length,
        }


def rollout_episode(
    state: QCTrainState,
    env: Any,
    rng: jax.Array,
    reset_seed: int,
    max_episode_steps: int | None = None,
) -> tuple[EpisodeResult, jax.Array]:
    """Run one episode, resampling only after each open-loop action chunk."""
    observation, _ = env.reset(seed=reset_seed)
    episode_return = 0.0
    episode_length = 0
    success = 0.0
    chunks_sampled = 0
    done = False

    while not done:
        rng, action_rng = jax.random.split(rng)
        observations = jnp.asarray(observation, dtype=jnp.float32)[None, :]
        flat_chunk, _ = state.select_action_chunks(observations, action_rng)
        action_chunk = np.asarray(jax.device_get(flat_chunk[0])).reshape(
            state.config.horizon, state.config.action_dim
        )
        chunks_sampled += 1

        for action in action_chunk:
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            episode_length += 1
            success = max(success, float(info.get("success", 0.0)))
            reached_limit = max_episode_steps is not None and episode_length >= max_episode_steps
            done = bool(terminated or truncated or reached_limit)
            if done:
                break

    return (
        EpisodeResult(
            episode_return=episode_return,
            length=episode_length,
            success=success,
            chunks_sampled=chunks_sampled,
        ),
        rng,
    )


def evaluate(
    state: QCTrainState,
    env: Any,
    config: EvaluationConfig | None = None,
) -> EvaluationResult:
    """Evaluate a QC policy over multiple episodes."""
    if config is None:
        config = EvaluationConfig()
    config.validate()
    rng = jax.random.PRNGKey(config.seed)
    episode_results = []
    for episode_index in range(config.episodes):
        result, rng = rollout_episode(
            state,
            env,
            rng,
            reset_seed=config.seed + episode_index,
            max_episode_steps=config.max_episode_steps,
        )
        episode_results.append(result)

    success_rate = float(np.mean([result.success for result in episode_results]))
    mean_return = float(np.mean([result.episode_return for result in episode_results]))
    mean_length = float(np.mean([result.length for result in episode_results]))
    return EvaluationResult(
        episodes=tuple(episode_results),
        success_rate=success_rate,
        mean_return=mean_return,
        mean_length=mean_length,
    )
