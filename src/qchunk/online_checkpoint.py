import json
import shutil
from dataclasses import asdict
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import serialization

from qchunk.online import OnlineTrainingProgress
from qchunk.replay_buffer import ReplayBuffer
from qchunk.training import QCTrainState


def _config_json(state: QCTrainState) -> str:
    return json.dumps(asdict(state.config), sort_keys=True, separators=(",", ":"))


def save_online_checkpoint(
    path: str | Path,
    progress: OnlineTrainingProgress,
) -> Path:
    """Atomically save learner, replay, environment, and online runtime state."""
    if progress.environment_steps < 0:
        raise ValueError("environment_steps must be non-negative")
    if int(progress.state.flow.step) != int(progress.state.critic.step):
        raise ValueError("flow and critic optimizer steps must match")

    replay_storage = progress.replay_buffer.storage_state()
    payload = {
        "format_version": 1,
        "config_json": _config_json(progress.state),
        "state": serialization.to_state_dict(progress.state),
        "online": {
            "environment_steps": progress.environment_steps,
            "policy_rng": np.asarray(progress.policy_rng),
            "replay_rng_state_json": json.dumps(
                progress.replay_rng_state, sort_keys=True, separators=(",", ":")
            ),
            "observation": np.asarray(progress.observation, dtype=np.float32),
            "queued_actions": np.asarray(progress.queued_actions, dtype=np.float32),
            "episode_return": progress.episode_return,
            "episode_length": progress.episode_length,
            "episode_success": progress.episode_success,
            "completed_episodes": progress.completed_episodes,
            "update_count": progress.update_count,
            "environment_state": progress.environment_state,
            "replay_buffer": {
                "capacity": replay_storage["capacity"],
                "size": replay_storage["size"],
                "pointer": replay_storage["pointer"],
            },
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    backup_path = path.with_name(f".{path.name}.backup")
    _recover_interrupted_swap(path, backup_path)
    _remove_path(temporary_path)
    temporary_path.mkdir()

    try:
        (temporary_path / "metadata.msgpack").write_bytes(serialization.msgpack_serialize(payload))
        replay_directory = temporary_path / "replay"
        replay_directory.mkdir()
        replay_data = replay_storage["data"]
        for key in ReplayBuffer.REQUIRED_FIELDS:
            np.save(replay_directory / f"{key}.npy", replay_data[key], allow_pickle=False)

        if path.exists():
            path.replace(backup_path)
        temporary_path.replace(path)
        _remove_path(backup_path)
    except Exception:
        if not path.exists() and backup_path.exists():
            backup_path.replace(path)
        raise
    return path


def load_online_checkpoint(
    path: str | Path,
    template: QCTrainState,
) -> OnlineTrainingProgress:
    """Restore an exact online-training continuation point."""
    path = Path(path)
    backup_path = path.with_name(f".{path.name}.backup")
    _recover_interrupted_swap(path, backup_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    metadata_path = path / "metadata.msgpack"
    if not metadata_path.is_file():
        raise ValueError("Online checkpoint does not contain metadata")
    payload = serialization.msgpack_restore(metadata_path.read_bytes())
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported online checkpoint format")
    if payload.get("config_json") != _config_json(template):
        raise ValueError("Checkpoint configuration does not match the template state")
    if "state" not in payload:
        raise ValueError("Checkpoint does not contain learner state")
    online = payload.get("online")
    if not isinstance(online, dict):
        raise TypeError("Checkpoint does not contain online-training progress")

    try:
        environment_steps = int(online["environment_steps"])
        replay_rng_state = json.loads(online["replay_rng_state_json"])
        environment_state = online["environment_state"]
        replay_buffer_state = online["replay_buffer"]
        episode_length = int(online["episode_length"])
        completed_episodes = int(online["completed_episodes"])
        update_count = int(online["update_count"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("Checkpoint contains invalid online-training progress") from error

    if (
        environment_steps < 0
        or episode_length < 0
        or completed_episodes < 0
        or update_count < 0
        or not isinstance(replay_rng_state, dict)
        or not isinstance(environment_state, dict)
        or not isinstance(replay_buffer_state, dict)
    ):
        raise ValueError("Checkpoint contains invalid online-training progress")

    state = serialization.from_state_dict(template, payload["state"])
    if int(state.flow.step) != int(state.critic.step):
        raise ValueError("Checkpoint optimizer step counters do not match")
    replay_data = {}
    for key in ReplayBuffer.REQUIRED_FIELDS:
        array_path = path / "replay" / f"{key}.npy"
        if not array_path.is_file():
            raise ValueError(f"Online checkpoint is missing replay field: {key}")
        replay_data[key] = np.load(array_path, mmap_mode="r", allow_pickle=False)
    replay_buffer = ReplayBuffer.from_storage_state({**replay_buffer_state, "data": replay_data})
    observation = np.asarray(online["observation"], dtype=np.float32)
    queued_actions = np.asarray(online["queued_actions"], dtype=np.float32)
    policy_rng = np.asarray(online["policy_rng"], dtype=np.uint32)
    if observation.shape != (replay_buffer.observation_dim,):
        raise ValueError("Checkpoint observation has an invalid shape")
    if (
        queued_actions.ndim != 2
        or queued_actions.shape[1] != replay_buffer.action_dim
        or queued_actions.shape[0] > template.config.horizon
    ):
        raise ValueError("Checkpoint queued actions have an invalid shape")
    if policy_rng.shape != (2,):
        raise ValueError("Checkpoint policy RNG has an invalid shape")

    return OnlineTrainingProgress(
        state=state,
        replay_buffer=replay_buffer,
        environment_steps=environment_steps,
        policy_rng=jnp.asarray(policy_rng),
        replay_rng_state=replay_rng_state,
        observation=observation,
        queued_actions=queued_actions,
        episode_return=float(online["episode_return"]),
        episode_length=episode_length,
        episode_success=float(online["episode_success"]),
        completed_episodes=completed_episodes,
        update_count=update_count,
        environment_state=environment_state,
    )


def _recover_interrupted_swap(path: Path, backup_path: Path) -> None:
    if path.exists() and backup_path.exists():
        _remove_path(backup_path)
    elif not path.exists() and backup_path.exists():
        backup_path.replace(path)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
