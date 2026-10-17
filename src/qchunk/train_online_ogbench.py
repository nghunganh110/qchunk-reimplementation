import argparse
import json
from pathlib import Path

from qchunk.checkpoint import load_checkpoint, save_checkpoint
from qchunk.evaluation import EvaluationConfig, evaluate
from qchunk.ogbench import load_ogbench
from qchunk.online import OnlineMetricRecord, OnlineTrainingConfig, train_online
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

        def evaluate_state(_step: int, current_state: QCTrainState) -> dict[str, float]:
            return evaluate(current_state, bundle.eval_env, evaluation_config).as_metrics()

        result = train_online(
            state,
            bundle.env,
            bundle.train_dataset,
            online_config,
            on_log=lambda record: _print_record("online", record),
            evaluate_fn=evaluate_state,
            on_evaluate=lambda record: _print_record("evaluation", record),
        )
        offline_output_path = Path(experiment["output"]["checkpoint"])
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
    )
    print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
