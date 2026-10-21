import argparse
import json
from pathlib import Path

from qchunk.checkpoint import load_checkpoint
from qchunk.ogbench import make_ogbench_env
from qchunk.simulation import record_episode
from qchunk.train_ogbench import load_experiment_config, make_agent_config
from qchunk.training import QCTrainState


def run(
    config_path: str | Path,
    output_path: str | Path,
    checkpoint_path: str | Path | None = None,
    *,
    seed: int = 42,
    frame_stride: int = 5,
    fps: int = 10,
) -> dict[str, float | int | str]:
    experiment = load_experiment_config(config_path)
    environment = experiment["environment"]
    environment_name = environment["name"]
    env = make_ogbench_env(
        environment_name,
        sparse_reward=bool(environment.get("sparse_reward", False)),
        render_mode="rgb_array",
    )
    try:
        agent_config = make_agent_config(experiment["agent"], env.action_space.shape[0])
        template = QCTrainState.create(
            seed=int(experiment.get("seed", 0)),
            observation_dim=env.observation_space.shape[0],
            config=agent_config,
        )
        if checkpoint_path is None:
            checkpoint_path = experiment["output"].get(
                "online_checkpoint", experiment["output"]["checkpoint"]
            )
        state = load_checkpoint(checkpoint_path, template)
        return record_episode(
            state,
            env,
            output_path,
            seed=seed,
            frame_stride=frame_stride,
            fps=fps,
        ).as_metrics()
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a QC checkpoint in OGBench")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    metrics = run(
        args.config,
        args.output,
        args.checkpoint,
        seed=args.seed,
        frame_stride=args.frame_stride,
        fps=args.fps,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
