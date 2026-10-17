import json
from dataclasses import asdict
from pathlib import Path

from flax import serialization

from qchunk.training import QCTrainState


def _config_json(state: QCTrainState) -> str:
    return json.dumps(asdict(state.config), sort_keys=True, separators=(",", ":"))


def save_checkpoint(path: str | Path, state: QCTrainState) -> Path:
    """Atomically save model, optimizer, target, and RNG state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_json": _config_json(state),
        "state": serialization.to_state_dict(state),
    }
    encoded = serialization.msgpack_serialize(payload)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_bytes(encoded)
    temporary_path.replace(path)
    return path


def load_checkpoint(path: str | Path, template: QCTrainState) -> QCTrainState:
    """Restore a checkpoint into a state with matching static configuration."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    payload = serialization.msgpack_restore(path.read_bytes())
    expected_config = _config_json(template)
    if payload.get("config_json") != expected_config:
        raise ValueError("Checkpoint configuration does not match the template state")
    if "state" not in payload:
        raise ValueError("Checkpoint does not contain training state")
    return serialization.from_state_dict(template, payload["state"])
