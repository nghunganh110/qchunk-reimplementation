import argparse
import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

from qchunk.checkpoint import load_checkpoint, save_checkpoint
from qchunk.evaluation import EvaluationConfig, evaluate
from qchunk.ogbench import (
    capture_ogbench_env_state,
    load_ogbench,
    restore_ogbench_env_state,
)
from qchunk.online import (
    OnlineEpisodeRecord,
    OnlineMetricRecord,
    OnlineTrainingConfig,
    train_online,
)
from qchunk.online_checkpoint import load_online_checkpoint, save_online_checkpoint
from qchunk.tracking import MetricLogger, record_manifest_event, run_directory
from qchunk.train_ogbench import load_experiment_config, make_agent_config
from qchunk.training import QCTrainState


def _print_record(phase: str, record: OnlineMetricRecord) -> None:
    print(json.dumps({"phase": phase, "step": record.step, **record.metrics}, sort_keys=True))


def run(
    config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    steps_override: int | None = None,
    start_training_override: int | None = None,
    evaluation_episodes_override: int | None = None,
    resume_path: str | Path | None = None,
    on_checkpoint_saved: Callable[[Path], None] | None = None,
) -> Path:
    experiment = load_experiment_config(config_path)
    if not isinstance(experiment.get("online"), dict):
        raise TypeError("experiment config requires an 'online' mapping")

    environment_config = experiment["environment"]
    bundle = load_ogbench(
        environment_config["name"],
        dataset_dir=environment_config.get("dataset_dir"),
        action_clip_eps=environment_config.get("action_clip_eps", 1e-5),
    )
    try:
        agent_config = make_agent_config(experiment["agent"], bundle.env.action_space.shape[0])
        template = QCTrainState.create(
            seed=int(experiment.get("seed", 0)),
            observation_dim=bundle.env.observation_space.shape[0],
            config=agent_config,
        )
        resume = None
        if resume_path is not None:
            resume = load_online_checkpoint(resume_path, template)
            state = resume.state
        else:
            if checkpoint_path is None:
                checkpoint_path = experiment["output"]["checkpoint"]
            state = load_checkpoint(checkpoint_path, template)

        online_values = experiment["online"].copy()
        if steps_override is not None:
            online_values["steps"] = steps_override
        if start_training_override is not None:
            online_values["start_training"] = start_training_override
        online_config = OnlineTrainingConfig(**online_values)
        evaluation_values = experiment.get("evaluation", {}).copy()
        if evaluation_episodes_override is not None:
            evaluation_values["episodes"] = evaluation_episodes_override
        evaluation_config = EvaluationConfig(**evaluation_values)

        initial_step = 0 if resume is None else resume.environment_steps
        effective_experiment = deepcopy(experiment)
        effective_experiment["online"] = online_values
        effective_experiment["evaluation"] = evaluation_values
        run_dir = run_directory(effective_experiment)
        metric_logger = MetricLogger(run_dir)
        for phase in ("online", "online_evaluation", "online_episode"):
            metric_logger.prepare_phase(phase, resume_step=initial_step)
        record_manifest_event(
            run_dir,
            effective_experiment,
            "online",
            resume_checkpoint=(resume_path if resume_path is not None else checkpoint_path),
            resume_step=initial_step,
        )

        def evaluate_state(_step: int, current_state: QCTrainState) -> dict[str, float]:
            return evaluate(current_state, bundle.eval_env, evaluation_config).as_metrics()

        def log_record(phase: str, record: OnlineMetricRecord) -> None:
            _print_record(phase, record)
            metric_logger.log(phase, record.step, record.metrics)

        def log_episode(record: OnlineEpisodeRecord) -> None:
            metric_logger.log(
                "online_episode",
                record.end_step,
                {
                    "episode_return": record.episode_return,
                    "length": float(record.length),
                    "success": record.success,
                },
            )

        offline_output_path = Path(experiment["output"]["checkpoint"])
        resume_output_path = Path(
            experiment["output"].get(
                "online_resume_checkpoint",
                offline_output_path.with_name("online_resume"),
            )
        )

        def save_progress(progress):
            path = save_online_checkpoint(resume_output_path, progress)
            if on_checkpoint_saved is not None:
                on_checkpoint_saved(path)
            return path

        result = train_online(
            state,
            bundle.env,
            bundle.train_dataset,
            online_config,
            on_log=lambda record: log_record("online", record),
            evaluate_fn=evaluate_state,
            on_evaluate=lambda record: log_record("online_evaluation", record),
            on_episode=log_episode,
            resume=resume,
            capture_environment_state=capture_ogbench_env_state,
            restore_environment_state=restore_ogbench_env_state,
            on_checkpoint=save_progress,
        )
        save_progress(result.progress)
        output_path = Path(
            experiment["output"].get(
                "online_checkpoint", offline_output_path.with_name("online_final.msgpack")
            )
        )
        return save_checkpoint(output_path, result.state)
    finally:
        bundle.env.close()
        bundle.eval_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Online-fine-tune QC on OGBench")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Offline checkpoint to load")
    parser.add_argument("--resume", type=Path, help="Resume an online-training checkpoint")
    parser.add_argument("--steps", type=int, help="Override total online environment steps")
    parser.add_argument("--start-training", type=int, help="Override first online update step")
    parser.add_argument("--eval-episodes", type=int, help="Override evaluation episode count")
    args = parser.parse_args()
    checkpoint_path = run(
        args.config,
        args.checkpoint,
        args.steps,
        args.start_training,
        args.eval_episodes,
        args.resume,
    )
    print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
