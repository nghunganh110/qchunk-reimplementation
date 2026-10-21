import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from qchunk.checkpoint import load_training_checkpoint, save_training_checkpoint
from qchunk.config import QCConfig
from qchunk.offline import (
    MetricRecord,
    OfflineTrainingConfig,
    OfflineTrainingResult,
    train_offline,
)
from qchunk.ogbench import load_ogbench
from qchunk.tracking import MetricLogger, record_manifest_event, run_directory
from qchunk.training import QCTrainState


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError("experiment config must contain a YAML mapping")
    for section in ("environment", "agent", "offline", "output"):
        if not isinstance(config.get(section), dict):
            raise TypeError(f"experiment config requires a '{section}' mapping")
    return config


def make_agent_config(values: dict[str, Any], action_dim: int) -> QCConfig:
    values = values.copy()
    configured_action_dim = values.pop("action_dim", action_dim)
    if configured_action_dim != action_dim:
        raise ValueError(
            f"configured action_dim={configured_action_dim} does not match environment "
            f"action_dim={action_dim}"
        )
    if "hidden_dims" in values:
        values["hidden_dims"] = tuple(values["hidden_dims"])
    return QCConfig(action_dim=action_dim, **values)


def print_record(record: MetricRecord) -> None:
    print(json.dumps({"step": record.step, **record.metrics}, sort_keys=True))


def run(
    config_path: str | Path,
    steps_override: int | None = None,
    resume_path: str | Path | None = None,
) -> Path:
    experiment = load_experiment_config(config_path)
    environment_config = experiment["environment"]
    dataset_name = environment_config["name"]
    bundle = load_ogbench(
        dataset_name,
        dataset_dir=environment_config.get("dataset_dir"),
        action_clip_eps=environment_config.get("action_clip_eps", 1e-5),
        sparse_reward=bool(environment_config.get("sparse_reward", False)),
    )
    try:
        observation_dim = bundle.env.observation_space.shape[0]
        action_dim = bundle.env.action_space.shape[0]
        agent_config = make_agent_config(experiment["agent"], action_dim)
        state = QCTrainState.create(
            seed=int(experiment.get("seed", 0)),
            observation_dim=observation_dim,
            config=agent_config,
        )

        offline_values = experiment["offline"].copy()
        if steps_override is not None:
            offline_values["steps"] = steps_override
        training_config = OfflineTrainingConfig(**offline_values)

        initial_step = 0
        dataset_rng_state = None
        if resume_path is not None:
            restored = load_training_checkpoint(resume_path, state)
            state = restored.state
            initial_step = restored.step
            dataset_rng_state = restored.dataset_rng_state

        effective_experiment = deepcopy(experiment)
        effective_experiment["offline"] = offline_values
        run_dir = run_directory(effective_experiment)
        metric_logger = MetricLogger(run_dir)
        metric_logger.prepare_phase("offline", resume_step=initial_step)
        record_manifest_event(
            run_dir,
            effective_experiment,
            "offline",
            resume_checkpoint=resume_path,
            resume_step=initial_step,
        )

        output_config = experiment["output"]
        checkpoint_path = Path(output_config["checkpoint"])
        checkpoint_dir = Path(
            output_config.get("checkpoint_dir", checkpoint_path.parent / "checkpoints")
        )

        def save_periodic_checkpoint(progress: OfflineTrainingResult) -> None:
            path = checkpoint_dir / f"offline_step_{progress.step:07d}.msgpack"
            save_training_checkpoint(
                path,
                progress.state,
                progress.step,
                progress.dataset_rng_state,
            )

        def log_record(record: MetricRecord) -> None:
            print_record(record)
            metric_logger.log("offline", record.step, record.metrics)

        result = train_offline(
            state,
            bundle.train_dataset,
            training_config,
            on_log=log_record,
            initial_step=initial_step,
            dataset_rng_state=dataset_rng_state,
            on_checkpoint=save_periodic_checkpoint,
        )
        return save_training_checkpoint(
            checkpoint_path,
            result.state,
            result.step,
            result.dataset_rng_state,
        )
    finally:
        bundle.env.close()
        bundle.eval_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline-pretrain QC on OGBench")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--steps", type=int, help="Override the total target offline steps")
    parser.add_argument("--resume", type=Path, help="Resume from an offline-training checkpoint")
    args = parser.parse_args()
    checkpoint_path = run(args.config, args.steps, args.resume)
    print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
