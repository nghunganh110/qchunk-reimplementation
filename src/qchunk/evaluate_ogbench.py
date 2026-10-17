import argparse
import json
from pathlib import Path

from qchunk.checkpoint import load_checkpoint
from qchunk.evaluation import EvaluationConfig, evaluate
from qchunk.ogbench import make_ogbench_env
from qchunk.train_ogbench import load_experiment_config, make_agent_config
from qchunk.training import QCTrainState


def run(
    config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    episodes_override: int | None = None,
) -> dict[str, float]:
    experiment = load_experiment_config(config_path)
    env = make_ogbench_env(experiment["environment"]["name"])
    try:
        agent_config = make_agent_config(experiment["agent"], env.action_space.shape[0])
        template = QCTrainState.create(
            seed=int(experiment.get("seed", 0)),
            observation_dim=env.observation_space.shape[0],
            config=agent_config,
        )
        if checkpoint_path is None:
            checkpoint_path = experiment["output"]["checkpoint"]
        state = load_checkpoint(checkpoint_path, template)

        evaluation_values = experiment.get("evaluation", {}).copy()
        if episodes_override is not None:
            evaluation_values["episodes"] = episodes_override
        result = evaluate(state, env, EvaluationConfig(**evaluation_values))
        return result.as_metrics()
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a QC checkpoint on OGBench")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int)
    args = parser.parse_args()
    metrics = run(args.config, args.checkpoint, args.episodes)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
