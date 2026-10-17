from collections import deque
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from qchunk.dataset import TransitionDataset
from qchunk.replay_buffer import ReplayBuffer
from qchunk.training import QCTrainState, batch_to_jax

Metrics = dict[str, float]


@dataclass(frozen=True)
class OnlineTrainingConfig:
    steps: int
    buffer_capacity: int
    batch_size: int
    start_training: int = 5_000
    updates_per_step: int = 1
    log_interval: int = 5_000
    evaluation_interval: int | None = 100_000
    checkpoint_interval: int | None = 50_000
    replay_seed: int = 1
    policy_seed: int = 1
    environment_seed: int = 1

    def validate(self) -> None:
        for name in (
            "steps",
            "buffer_capacity",
            "batch_size",
            "start_training",
            "updates_per_step",
            "log_interval",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.evaluation_interval is not None and self.evaluation_interval <= 0:
            raise ValueError("evaluation_interval must be positive when provided")
        if self.checkpoint_interval is not None and self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive when provided")


@dataclass(frozen=True)
class OnlineMetricRecord:
    step: int
    metrics: Metrics


@dataclass(frozen=True)
class OnlineEpisodeRecord:
    end_step: int
    episode_return: float
    length: int
    success: float


@dataclass(frozen=True)
class OnlineTrainingResult:
    state: QCTrainState
    replay_buffer: ReplayBuffer
    records: tuple[OnlineMetricRecord, ...]
    evaluations: tuple[OnlineMetricRecord, ...]
    episodes: tuple[OnlineEpisodeRecord, ...]
    environment_steps: int
    policy_rng: jax.Array
    replay_rng_state: dict[str, Any]
    progress: "OnlineTrainingProgress"


@dataclass(frozen=True)
class OnlineTrainingProgress:
    state: QCTrainState
    replay_buffer: ReplayBuffer
    environment_steps: int
    policy_rng: jax.Array
    replay_rng_state: dict[str, Any]
    observation: np.ndarray
    queued_actions: np.ndarray
    episode_return: float
    episode_length: int
    episode_success: float
    completed_episodes: int
    update_count: int
    environment_state: dict[str, Any]


def _host_metrics(metrics: dict[str, jax.Array]) -> Metrics:
    host_values = jax.device_get(metrics)
    return {name: float(value) for name, value in host_values.items()}


def train_online(
    state: QCTrainState,
    env: Any,
    offline_dataset: TransitionDataset,
    config: OnlineTrainingConfig,
    on_log: Callable[[OnlineMetricRecord], None] | None = None,
    evaluate_fn: Callable[[int, QCTrainState], Metrics] | None = None,
    on_evaluate: Callable[[OnlineMetricRecord], None] | None = None,
    *,
    resume: OnlineTrainingProgress | None = None,
    capture_environment_state: Callable[[Any], dict[str, Any]] | None = None,
    restore_environment_state: Callable[[Any, dict[str, Any]], None] | None = None,
    on_checkpoint: Callable[[OnlineTrainingProgress], None] | None = None,
) -> OnlineTrainingResult:
    """Collect online experience and fine-tune QC from a combined replay buffer."""
    config.validate()
    observation_dim = offline_dataset.data["observations"].shape[-1]
    action_dim = offline_dataset.data["actions"].shape[-1]
    if action_dim != state.config.action_dim:
        raise ValueError("offline dataset action dimension does not match QCConfig")

    if on_checkpoint is not None and capture_environment_state is None:
        raise ValueError("capture_environment_state is required for online checkpoints")

    replay_rng = np.random.default_rng(config.replay_seed)
    if resume is None:
        capacity = max(config.buffer_capacity, len(offline_dataset) + 1)
        replay_buffer = ReplayBuffer.from_dataset(offline_dataset, capacity)
        policy_rng = jax.random.PRNGKey(config.policy_seed)
        observation, _ = env.reset(seed=config.environment_seed)
        observation = _observation(observation, observation_dim)
        action_queue: deque[np.ndarray] = deque()
        initial_step = 0
        episode_return = 0.0
        episode_length = 0
        episode_success = 0.0
        completed_episodes = 0
        update_count = 0
    else:
        if restore_environment_state is None:
            raise ValueError("restore_environment_state is required when resuming online training")
        if not 0 <= resume.environment_steps <= config.steps:
            raise ValueError("resumed environment_steps must lie in [0, config.steps]")
        state = resume.state
        replay_buffer = resume.replay_buffer
        if replay_buffer.observation_dim != observation_dim:
            raise ValueError(
                "resumed replay buffer observation dimension does not match offline dataset"
            )
        if replay_buffer.action_dim != state.config.action_dim:
            raise ValueError("resumed replay buffer action dimension does not match QCConfig")
        replay_rng.bit_generator.state = deepcopy(resume.replay_rng_state)
        policy_rng = resume.policy_rng
        observation = _observation(resume.observation, observation_dim)
        queued_actions = np.asarray(resume.queued_actions, dtype=np.float32)
        if (
            queued_actions.ndim != 2
            or queued_actions.shape[1] != state.config.action_dim
            or queued_actions.shape[0] > state.config.horizon
        ):
            raise ValueError("resumed queued_actions have an invalid shape")
        action_queue = deque(queued_actions)
        initial_step = resume.environment_steps
        episode_return = resume.episode_return
        episode_length = resume.episode_length
        episode_success = resume.episode_success
        completed_episodes = resume.completed_episodes
        update_count = resume.update_count
        restore_environment_state(env, resume.environment_state)

    records = []
    evaluations = []
    episodes = []
    latest_device_metrics = None

    for step in range(initial_step + 1, config.steps + 1):
        if not action_queue:
            policy_rng, action_rng = jax.random.split(policy_rng)
            flat_chunk, _ = state.select_action_chunks(
                jnp.asarray(observation, dtype=jnp.float32)[None, :],
                action_rng,
            )
            action_chunk = np.asarray(jax.device_get(flat_chunk[0])).reshape(
                state.config.horizon, state.config.action_dim
            )
            action_queue.extend(action_chunk)

        action = action_queue.popleft()
        next_observation, reward, terminated, truncated, info = env.step(action)
        next_observation = _observation(next_observation, observation_dim)
        done = bool(terminated or truncated)
        replay_buffer.add_transition(
            {
                "observations": observation,
                "actions": action,
                "rewards": reward,
                "next_observations": next_observation,
                "masks": float(not terminated),
                "terminals": float(done),
            }
        )

        episode_return += float(reward)
        episode_length += 1
        episode_success = max(episode_success, float(info.get("success", 0.0)))

        if done:
            episodes.append(
                OnlineEpisodeRecord(
                    end_step=step,
                    episode_return=episode_return,
                    length=episode_length,
                    success=episode_success,
                )
            )
            completed_episodes += 1
            observation, _ = env.reset()
            observation = _observation(observation, observation_dim)
            action_queue.clear()
            episode_return = 0.0
            episode_length = 0
            episode_success = 0.0
        else:
            observation = next_observation

        if step >= config.start_training:
            for _ in range(config.updates_per_step):
                chunk_batch = replay_buffer.sample_chunks(
                    config.batch_size,
                    state.config.horizon,
                    replay_rng,
                )
                state, latest_device_metrics = state.update(batch_to_jax(chunk_batch))
                update_count += 1

        should_log = (
            step == initial_step + 1 or step % config.log_interval == 0 or step == config.steps
        )
        if should_log:
            metrics = {
                "buffer_size": float(len(replay_buffer)),
                "episodes": float(completed_episodes),
                "updates": float(update_count),
            }
            if latest_device_metrics is not None:
                metrics.update(_host_metrics(latest_device_metrics))
            record = OnlineMetricRecord(step=step, metrics=metrics)
            records.append(record)
            if on_log is not None:
                on_log(record)

        should_evaluate = evaluate_fn is not None and (
            step == config.steps
            or (config.evaluation_interval is not None and step % config.evaluation_interval == 0)
        )
        if should_evaluate:
            evaluation = OnlineMetricRecord(step=step, metrics=evaluate_fn(step, state))
            evaluations.append(evaluation)
            if on_evaluate is not None:
                on_evaluate(evaluation)

        should_checkpoint = (
            on_checkpoint is not None
            and config.checkpoint_interval is not None
            and step % config.checkpoint_interval == 0
        )
        if should_checkpoint:
            on_checkpoint(
                _make_progress(
                    state=state,
                    replay_buffer=replay_buffer,
                    environment_steps=step,
                    policy_rng=policy_rng,
                    replay_rng_state=replay_rng.bit_generator.state,
                    observation=observation,
                    action_queue=action_queue,
                    episode_return=episode_return,
                    episode_length=episode_length,
                    episode_success=episode_success,
                    completed_episodes=completed_episodes,
                    update_count=update_count,
                    environment_state=capture_environment_state(env),
                )
            )

    final_environment_state = (
        capture_environment_state(env) if capture_environment_state is not None else {}
    )
    progress = _make_progress(
        state=state,
        replay_buffer=replay_buffer,
        environment_steps=config.steps,
        policy_rng=policy_rng,
        replay_rng_state=replay_rng.bit_generator.state,
        observation=observation,
        action_queue=action_queue,
        episode_return=episode_return,
        episode_length=episode_length,
        episode_success=episode_success,
        completed_episodes=completed_episodes,
        update_count=update_count,
        environment_state=final_environment_state,
    )
    return OnlineTrainingResult(
        state=state,
        replay_buffer=replay_buffer,
        records=tuple(records),
        evaluations=tuple(evaluations),
        episodes=tuple(episodes),
        environment_steps=config.steps,
        policy_rng=policy_rng,
        replay_rng_state=deepcopy(replay_rng.bit_generator.state),
        progress=progress,
    )


def _make_progress(
    *,
    state: QCTrainState,
    replay_buffer: ReplayBuffer,
    environment_steps: int,
    policy_rng: jax.Array,
    replay_rng_state: dict[str, Any],
    observation: np.ndarray,
    action_queue: deque[np.ndarray],
    episode_return: float,
    episode_length: int,
    episode_success: float,
    completed_episodes: int,
    update_count: int,
    environment_state: dict[str, Any],
) -> OnlineTrainingProgress:
    queued_actions = (
        np.stack(action_queue).astype(np.float32, copy=False)
        if action_queue
        else np.empty((0, state.config.action_dim), dtype=np.float32)
    )
    return OnlineTrainingProgress(
        state=state,
        replay_buffer=replay_buffer,
        environment_steps=environment_steps,
        policy_rng=policy_rng,
        replay_rng_state=deepcopy(replay_rng_state),
        observation=observation.copy(),
        queued_actions=queued_actions,
        episode_return=episode_return,
        episode_length=episode_length,
        episode_success=episode_success,
        completed_episodes=completed_episodes,
        update_count=update_count,
        environment_state=environment_state,
    )


def _observation(value: object, observation_dim: int) -> np.ndarray:
    observation = np.asarray(value, dtype=np.float32)
    if observation.shape != (observation_dim,):
        raise ValueError(f"environment observation must have shape [{observation_dim}]")
    return observation
