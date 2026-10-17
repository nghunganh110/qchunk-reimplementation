import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from flax import serialization

from qchunk.training import QCTrainState


@dataclass(frozen=True)
class TrainingCheckpoint:
    state: QCTrainState
    step: int
    dataset_rng_state: dict[str, Any]


def _config_json(state: QCTrainState) -> str:
    return json.dumps(asdict(state.config), sort_keys=True, separators=(",", ":"))


def _write_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = serialization.msgpack_serialize(payload)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_bytes(encoded)
    temporary_path.replace(path)
    return path


def save_checkpoint(path: str | Path, state: QCTrainState) -> Path:
    """Atomically save model, optimizer, target, and RNG state."""
    payload = {
        "config_json": _config_json(state),
        "state": serialization.to_state_dict(state),
    }
    return _write_checkpoint(path, payload)


def save_training_checkpoint(
    path: str | Path,
    state: QCTrainState,
    step: int,
    dataset_rng_state: dict[str, Any],
) -> Path:
    """Atomically save QC state and exact offline-training progress."""
    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
    if int(state.flow.step) != step or int(state.critic.step) != step:
        raise ValueError("checkpoint step must match both optimizer step counters")
    payload = {
        "config_json": _config_json(state),
        "state": serialization.to_state_dict(state),
        "training": {
            "step": step,
            "dataset_rng_state_json": json.dumps(
                dataset_rng_state, sort_keys=True, separators=(",", ":")
            ),
        },
    }
    return _write_checkpoint(path, payload)


def _read_checkpoint(path: str | Path, template: QCTrainState) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    payload = serialization.msgpack_restore(path.read_bytes())
    expected_config = _config_json(template)
    if payload.get("config_json") != expected_config:
        raise ValueError("Checkpoint configuration does not match the template state")
    if "state" not in payload:
        raise ValueError("Checkpoint does not contain training state")
    return payload


def load_checkpoint(path: str | Path, template: QCTrainState) -> QCTrainState:
    """Restore a checkpoint into a state with matching static configuration."""
    payload = _read_checkpoint(path, template)
    return serialization.from_state_dict(template, payload["state"])


def load_training_checkpoint(path: str | Path, template: QCTrainState) -> TrainingCheckpoint:
    """Restore QC state, completed step count, and dataset sampling RNG."""
    payload = _read_checkpoint(path, template)
    training = payload.get("training")
    if not isinstance(training, dict):
        raise TypeError("Checkpoint does not contain offline-training progress")
    try:
        step = int(training["step"])
        dataset_rng_state = json.loads(training["dataset_rng_state_json"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("Checkpoint contains invalid offline-training progress") from error
    if step < 0 or not isinstance(dataset_rng_state, dict):
        raise ValueError("Checkpoint contains invalid offline-training progress")
    state = serialization.from_state_dict(template, payload["state"])
    if int(state.flow.step) != step or int(state.critic.step) != step:
        raise ValueError("checkpoint step does not match optimizer step counters")
    return TrainingCheckpoint(
        state=state,
        step=step,
        dataset_rng_state=dataset_rng_state,
    )
