import argparse
import json
import os
import shutil
import stat
import warnings
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import jax
import numpy as np

from qchunk.checkpoint import load_checkpoint, load_training_checkpoint
from qchunk.evaluation import EvaluationConfig
from qchunk.offline import OfflineTrainingConfig
from qchunk.ogbench import (
    make_ogbench_env,
    ogbench_dataset_paths,
    validate_ogbench_dataset_archive,
)
from qchunk.online import OnlineTrainingConfig
from qchunk.online_checkpoint import load_online_checkpoint
from qchunk.train_ogbench import load_experiment_config, make_agent_config
from qchunk.training import QCTrainState

Status = Literal["pass", "warning", "fail"]
Phase = Literal["offline", "online", "evaluation"]

PINNED_BENCHMARK_DEPENDENCIES = {
    "ogbench": "1.1.2",
    "gymnasium": "1.1.1",
    "mujoco": "3.3.1",
}
REQUIRED_DEPENDENCIES = ("jax", "flax", "optax", "numpy", "PyYAML")


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: Status
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreflightReport:
    phase: Phase
    checks: tuple[PreflightCheck, ...]

    @property
    def ready(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        counts = {
            status: sum(check.status == status for check in self.checks)
            for status in ("pass", "warning", "fail")
        }
        return {
            "phase": self.phase,
            "ready": self.ready,
            "summary": counts,
            "checks": [asdict(check) for check in self.checks],
        }


def _check_dependencies() -> PreflightCheck:
    versions = {}
    errors = []
    for package in (*REQUIRED_DEPENDENCIES, *PINNED_BENCHMARK_DEPENDENCIES):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            errors.append(f"{package} is not installed")
    for package, expected in PINNED_BENCHMARK_DEPENDENCIES.items():
        actual = versions.get(package)
        if actual is not None and actual != expected:
            errors.append(f"{package} is {actual}, expected {expected}")
    if errors:
        return PreflightCheck("dependencies", "fail", "; ".join(errors), versions)
    return PreflightCheck(
        "dependencies",
        "pass",
        "Required packages and pinned benchmark versions are installed.",
        versions,
    )


def _check_devices(require_gpu: bool) -> PreflightCheck:
    try:
        devices = jax.devices()
        platforms = sorted({device.platform for device in devices})
    except Exception as error:  # noqa: BLE001 - report backend failures instead of crashing
        return PreflightCheck("jax_devices", "fail", f"JAX device discovery failed: {error}")

    details = {
        "default_backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
        "platforms": platforms,
    }
    has_gpu = any(platform in {"cuda", "gpu", "rocm"} for platform in platforms)
    if not has_gpu:
        status: Status = "fail" if require_gpu else "warning"
        message = (
            "No GPU backend is available; --require-gpu was requested."
            if require_gpu
            else "No GPU backend is available. Long training will run on CPU."
        )
        return PreflightCheck("jax_devices", status, message, details)
    return PreflightCheck("jax_devices", "pass", "JAX can use a GPU backend.", details)


def _check_environment(experiment: dict[str, Any]) -> tuple[PreflightCheck, int | None, int | None]:
    environment = experiment["environment"]
    dataset_name = environment.get("name")
    if not isinstance(dataset_name, str) or not dataset_name:
        return (
            PreflightCheck("environment", "fail", "environment.name must be a non-empty string"),
            None,
            None,
        )

    env = None
    try:
        with warnings.catch_warnings():
            # MuJoCo can run headlessly even though GLFW cannot connect to a display.
            warnings.filterwarnings("ignore", message=".*Wayland: Failed to connect to display.*")
            warnings.filterwarnings("ignore", message=".*Box (low|high)'s precision lowered.*")
            env = make_ogbench_env(
                dataset_name,
                sparse_reward=bool(environment.get("sparse_reward", False)),
            )
            observation, _ = env.reset(seed=int(experiment.get("seed", 0)))
            observation_shape = tuple(env.observation_space.shape)
            action_shape = tuple(env.action_space.shape)
        if len(observation_shape) != 1 or len(action_shape) != 1:
            raise ValueError(
                "QC currently requires vector observations and continuous vector actions"
            )
        observation = np.asarray(observation)
        if observation.shape != observation_shape or not np.all(np.isfinite(observation)):
            raise ValueError("environment reset returned an invalid observation")
        observation_dim = observation_shape[0]
        action_dim = action_shape[0]
        return (
            PreflightCheck(
                "environment",
                "pass",
                f"Created and reset {dataset_name}.",
                {
                    "observation_shape": list(observation_shape),
                    "action_shape": list(action_shape),
                },
            ),
            observation_dim,
            action_dim,
        )
    except Exception as error:  # noqa: BLE001 - third-party environment errors vary
        return (
            PreflightCheck("environment", "fail", f"Environment smoke check failed: {error}"),
            None,
            None,
        )
    finally:
        if env is not None:
            env.close()


def _check_hyperparameters(
    experiment: dict[str, Any],
    phase: Phase,
    action_dim: int | None,
) -> tuple[PreflightCheck, Any | None]:
    if action_dim is None:
        return (
            PreflightCheck(
                "hyperparameters",
                "fail",
                "Cannot validate agent dimensions because the environment check failed.",
            ),
            None,
        )
    try:
        agent_config = make_agent_config(experiment["agent"], action_dim)
        agent_config.validate()
        OfflineTrainingConfig(**experiment["offline"]).validate()
        if "online" in experiment:
            if not isinstance(experiment["online"], dict):
                raise TypeError("experiment config 'online' must be a mapping")
            OnlineTrainingConfig(**experiment["online"]).validate()
        elif phase == "online":
            raise TypeError("online preflight requires an 'online' mapping")
        if "evaluation" in experiment:
            if not isinstance(experiment["evaluation"], dict):
                raise TypeError("experiment config 'evaluation' must be a mapping")
            EvaluationConfig(**experiment["evaluation"]).validate()
        return (
            PreflightCheck(
                "hyperparameters",
                "pass",
                "Agent and training hyperparameters are valid.",
                {
                    "action_dim": action_dim,
                    "horizon": agent_config.horizon,
                    "action_chunk_dim": agent_config.action_chunk_dim,
                },
            ),
            agent_config,
        )
    except Exception as error:  # noqa: BLE001 - dataclass constructors can reject many input types
        return PreflightCheck("hyperparameters", "fail", str(error)), None


def _check_dataset(experiment: dict[str, Any], phase: Phase) -> PreflightCheck:
    if phase == "evaluation":
        return PreflightCheck(
            "dataset",
            "pass",
            "Evaluation does not load an offline dataset.",
        )
    environment = experiment["environment"]
    train_path, validation_path = ogbench_dataset_paths(
        environment["name"], environment.get("dataset_dir")
    )
    paths = (train_path, validation_path)
    missing = [str(path) for path in paths if not path.is_file()]
    details = {
        "train_path": str(train_path),
        "validation_path": str(validation_path),
    }
    if missing:
        details["missing"] = missing
        partial_downloads = {
            str(path.with_suffix(f"{path.suffix}.tmp")): path.with_suffix(f"{path.suffix}.tmp")
            .stat()
            .st_size
            for path in paths
            if path.with_suffix(f"{path.suffix}.tmp").is_file()
        }
        if partial_downloads:
            details["partial_downloads_bytes"] = partial_downloads
        return PreflightCheck(
            "dataset",
            "fail",
            "OGBench dataset files are missing. The training command downloads or resumes them.",
            details,
        )
    empty = [str(path) for path in paths if path.stat().st_size == 0]
    if empty:
        details["empty"] = empty
        return PreflightCheck("dataset", "fail", "Dataset files are empty.", details)
    try:
        for path in paths:
            validate_ogbench_dataset_archive(path)
    except (OSError, ValueError) as error:
        return PreflightCheck("dataset", "fail", str(error), details)
    details["sizes_bytes"] = {str(path): path.stat().st_size for path in paths}
    return PreflightCheck(
        "dataset",
        "pass",
        "Training and validation archives have valid NPZ directories.",
        details,
    )


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.expanduser().absolute()
    if not candidate.is_dir():
        candidate = candidate.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _filesystem_mode_is_writable(path: Path) -> bool:
    """Check persistent Unix permissions without probing the path with a write."""
    info = path.stat()
    if os.geteuid() == 0:
        return True
    if info.st_uid == os.geteuid():
        return bool(info.st_mode & stat.S_IWUSR)
    if info.st_gid == os.getegid() or info.st_gid in os.getgroups():
        return bool(info.st_mode & stat.S_IWGRP)
    return bool(info.st_mode & stat.S_IWOTH)


def _check_storage(experiment: dict[str, Any], minimum_free_gib: float) -> PreflightCheck:
    if minimum_free_gib < 0:
        return PreflightCheck("storage", "fail", "minimum free space cannot be negative")
    environment = experiment["environment"]
    train_path, _ = ogbench_dataset_paths(environment["name"], environment.get("dataset_dir"))
    output = experiment["output"]
    output_paths = [Path(value) for value in output.values() if isinstance(value, str)]
    roots = {_nearest_existing_parent(train_path), *map(_nearest_existing_parent, output_paths)}
    required_bytes = int(minimum_free_gib * 1024**3)
    details = {"minimum_free_bytes": required_bytes, "filesystems": {}}
    errors = []
    warnings_found = []
    seen_devices = set()
    for root in sorted(roots, key=str):
        device = root.stat().st_dev
        if device in seen_devices:
            continue
        seen_devices.add(device)
        usage = shutil.disk_usage(root)
        # A Codex/container sandbox may make os.access() return false even when
        # the user's persistent Unix permissions permit terminal commands.
        mode_writable = _filesystem_mode_is_writable(root)
        runtime_access = os.access(root, os.W_OK)
        details["filesystems"][str(root)] = {
            "free_bytes": usage.free,
            "unix_mode_writable": mode_writable,
            "runtime_access": runtime_access,
        }
        if not mode_writable:
            errors.append(f"Unix permissions do not allow writing to {root}")
        elif not runtime_access:
            warnings_found.append(
                f"the current sandbox blocks {root}, although Unix permissions allow writing"
            )
        if usage.free < required_bytes:
            errors.append(f"{root} has less than {minimum_free_gib:g} GiB free")
    if errors:
        return PreflightCheck("storage", "fail", "; ".join(errors), details)
    if warnings_found:
        return PreflightCheck("storage", "warning", "; ".join(warnings_found), details)
    return PreflightCheck(
        "storage",
        "pass",
        f"Output and dataset filesystems are writable with at least {minimum_free_gib:g} GiB free.",
        details,
    )


def _checkpoint_path(
    experiment: dict[str, Any],
    phase: Phase,
    checkpoint_path: str | Path | None,
    resume_path: str | Path | None,
) -> tuple[Path | None, str]:
    if resume_path is not None:
        return Path(resume_path), "resume"
    if phase == "offline":
        return None, "none"
    if checkpoint_path is not None:
        return Path(checkpoint_path), "model"
    return Path(experiment["output"]["checkpoint"]), "model"


def _check_checkpoint(
    experiment: dict[str, Any],
    phase: Phase,
    observation_dim: int | None,
    agent_config: Any | None,
    checkpoint_path: str | Path | None,
    resume_path: str | Path | None,
) -> PreflightCheck:
    path, kind = _checkpoint_path(experiment, phase, checkpoint_path, resume_path)
    if path is None:
        return PreflightCheck("checkpoint", "pass", "Offline training will start from scratch.")
    if observation_dim is None or agent_config is None:
        return PreflightCheck(
            "checkpoint",
            "fail",
            "Cannot validate checkpoint compatibility because earlier checks failed.",
            {"path": str(path)},
        )
    if not path.exists():
        return PreflightCheck(
            "checkpoint",
            "fail",
            f"Checkpoint does not exist: {path}",
            {"path": str(path), "kind": kind},
        )
    try:
        template = QCTrainState.create(
            seed=int(experiment.get("seed", 0)),
            observation_dim=observation_dim,
            config=agent_config,
        )
        if kind == "resume" and phase == "online":
            progress = load_online_checkpoint(path, template)
            step = progress.environment_steps
        elif kind == "resume":
            restored = load_training_checkpoint(path, template)
            step = restored.step
        else:
            restored_state = load_checkpoint(path, template)
            step = int(restored_state.flow.step)
    except Exception as error:  # noqa: BLE001 - checkpoint decoders raise heterogeneous errors
        return PreflightCheck(
            "checkpoint",
            "fail",
            f"Checkpoint validation failed: {error}",
            {"path": str(path), "kind": kind},
        )
    return PreflightCheck(
        "checkpoint",
        "pass",
        f"Checkpoint is readable and compatible ({step} completed updates).",
        {"path": str(path), "kind": kind, "step": step},
    )


def run_preflight(
    config_path: str | Path,
    phase: Phase = "offline",
    *,
    checkpoint_path: str | Path | None = None,
    resume_path: str | Path | None = None,
    require_gpu: bool = False,
    minimum_free_gib: float = 5.0,
) -> PreflightReport:
    """Inspect whether the local machine is ready for one experiment phase."""
    checks = [_check_dependencies(), _check_devices(require_gpu)]
    try:
        experiment = load_experiment_config(config_path)
    except Exception as error:  # noqa: BLE001 - configuration loaders may raise I/O or YAML errors
        checks.insert(0, PreflightCheck("configuration", "fail", str(error)))
        return PreflightReport(phase=phase, checks=tuple(checks))

    checks.insert(
        0,
        PreflightCheck(
            "configuration",
            "pass",
            f"Loaded experiment configuration from {Path(config_path)}.",
        ),
    )
    environment_check, observation_dim, action_dim = _check_environment(experiment)
    checks.append(environment_check)
    hyperparameter_check, agent_config = _check_hyperparameters(experiment, phase, action_dim)
    checks.append(hyperparameter_check)
    checks.append(_check_dataset(experiment, phase))
    checks.append(_check_storage(experiment, minimum_free_gib))
    checks.append(
        _check_checkpoint(
            experiment,
            phase,
            observation_dim,
            agent_config,
            checkpoint_path,
            resume_path,
        )
    )
    return PreflightReport(phase=phase, checks=tuple(checks))


def _print_report(report: PreflightReport) -> None:
    labels = {"pass": "PASS", "warning": "WARN", "fail": "FAIL"}
    for check in report.checks:
        print(f"[{labels[check.status]}] {check.name}: {check.message}")
        for key, value in check.details.items():
            print(f"       {key}: {value}")
    print(f"Ready for {report.phase}: {'yes' if report.ready else 'no'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check readiness for a QC experiment")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--phase", choices=("offline", "online", "evaluation"), default="offline")
    parser.add_argument("--checkpoint", type=Path, help="Model checkpoint for online/evaluation")
    parser.add_argument("--resume", type=Path, help="Offline or online resume checkpoint")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--minimum-free-gib", type=float, default=5.0)
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report")
    args = parser.parse_args()
    report = run_preflight(
        args.config,
        args.phase,
        checkpoint_path=args.checkpoint,
        resume_path=args.resume,
        require_gpu=args.require_gpu,
        minimum_free_gib=args.minimum_free_gib,
    )
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        _print_report(report)
    raise SystemExit(0 if report.ready else 1)


if __name__ == "__main__":
    main()
