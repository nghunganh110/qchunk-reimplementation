import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from qchunk.offline import OfflineTrainingConfig, train_offline
from qchunk.ogbench import load_ogbench
from qchunk.train_ogbench import load_experiment_config, make_agent_config
from qchunk.training import QCTrainState


def run(config_path: str | Path, timed_steps: int = 2_000) -> dict[str, object]:
    """Warm up JAX, then measure offline learner throughput."""
    if timed_steps <= 0:
        raise ValueError("timed_steps must be positive")

    experiment = load_experiment_config(config_path)
    environment = experiment["environment"]
    bundle = load_ogbench(
        environment["name"],
        dataset_dir=environment.get("dataset_dir"),
        action_clip_eps=environment.get("action_clip_eps", 1e-5),
        sparse_reward=bool(environment.get("sparse_reward", False)),
    )
    try:
        agent_config = make_agent_config(experiment["agent"], bundle.env.action_space.shape[0])
        state = QCTrainState.create(
            seed=int(experiment.get("seed", 0)),
            observation_dim=bundle.env.observation_space.shape[0],
            config=agent_config,
        )

        offline_values = experiment["offline"].copy()
        offline_values.update({"steps": 1, "log_interval": 1, "checkpoint_interval": None})
        warmup = train_offline(
            state,
            bundle.train_dataset,
            OfflineTrainingConfig(**offline_values),
        )
        jax.block_until_ready(warmup.state.critic.params)

        offline_values.update(
            {
                "steps": timed_steps + 1,
                "log_interval": timed_steps + 1,
            }
        )
        started = time.perf_counter()
        measured = train_offline(
            warmup.state,
            bundle.train_dataset,
            OfflineTrainingConfig(**offline_values),
            initial_step=1,
            dataset_rng_state=warmup.dataset_rng_state,
        )
        jax.block_until_ready(measured.state.critic.params)
        elapsed = time.perf_counter() - started
        metrics = measured.records[-1].metrics
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"benchmark produced non-finite metrics: {metrics}")
    finally:
        bundle.env.close()
        bundle.eval_env.close()

    seconds_per_update = elapsed / timed_steps
    target_steps = int(experiment["offline"]["steps"])
    return {
        "mode": "benchmark",
        "timed_steps": timed_steps,
        "elapsed_seconds": elapsed,
        "seconds_per_update": seconds_per_update,
        "updates_per_second": 1.0 / seconds_per_update,
        "projection_target_steps": target_steps,
        "projected_offline_hours": seconds_per_update * target_steps / 3600.0,
        "metrics": metrics,
        "projection_scope": "offline learner only after JAX warm-up; excludes data loading",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark warmed-up QC learner throughput")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=2_000)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.steps), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
