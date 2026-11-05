"""Paired QC versus local gradient refinement evaluation on an OGBench task."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from qchunk.agent import best_of_n_action_chunks
from qchunk.checkpoint import load_checkpoint
from qchunk.evaluation import EvaluationConfig, evaluate
from qchunk.ogbench import make_ogbench_env
from qchunk.refinement import (
    select_geometric_refined_action_chunks,
    select_refined_action_chunks,
)
from qchunk.train_ogbench import load_experiment_config, make_agent_config
from qchunk.training import QCTrainState


@dataclass(frozen=True)
class Variant:
    name: str
    candidates: int
    gradient_steps: int = 0
    step_size: float = 0.0
    radius: float = 0.0
    neighbors: int = 0


VARIANTS = {
    "qc_n8": Variant("qc_n8", 8),
    "qc_n8_grad": Variant("qc_n8_grad", 8, gradient_steps=2, step_size=0.25, radius=0.5),
    "qc_n8_grad_small": Variant(
        "qc_n8_grad_small", 8, gradient_steps=2, step_size=0.05, radius=0.1
    ),
    "qc_n8_grad_geo": Variant(
        "qc_n8_grad_geo", 8, gradient_steps=2, step_size=0.05, radius=0.1, neighbors=3
    ),
    "qc_n32": Variant("qc_n32", 32),
    "qc_n64": Variant("qc_n64", 64),
}


@dataclass(frozen=True)
class EvaluationPolicy:
    config: object
    select_action_chunks: object


def make_policy(state: QCTrainState, variant: Variant) -> EvaluationPolicy:
    if variant.candidates <= 0:
        raise ValueError("candidates must be positive")
    config = state.config

    @jax.jit
    def select(observations, rng):
        common = (
            state.flow.apply_fn,
            state.flow.params,
            state.critic.apply_fn,
            state.critic.params,
            observations,
            rng,
        )
        kwargs = {
            "action_chunk_dim": config.action_chunk_dim,
            "num_candidates": variant.candidates,
            "num_flow_steps": config.num_flow_steps,
            "q_aggregation": config.q_aggregation,
        }
        if variant.neighbors:
            return select_geometric_refined_action_chunks(
                *common,
                **kwargs,
                neighbors=variant.neighbors,
                steps=variant.gradient_steps,
                step_size=variant.step_size,
                radius=variant.radius,
            )
        if variant.gradient_steps:
            return select_refined_action_chunks(
                *common,
                **kwargs,
                steps=variant.gradient_steps,
                step_size=variant.step_size,
                radius=variant.radius,
            )
        return best_of_n_action_chunks(*common, **kwargs)

    return EvaluationPolicy(config=config, select_action_chunks=select)


def diagnose_refinement(state: QCTrainState, env, *, episodes: int, seed: int) -> dict:
    """Compare critic Q before/after refinement on states visited by QC N=8."""
    from qchunk.agent import aggregate_q_values
    from qchunk.refinement import refine_action_chunks

    base = make_policy(state, VARIANTS["qc_n8"])
    observations_seen = []
    chunks_seen = []

    class RecordingPolicy:
        config = state.config

        def select_action_chunks(self, observations, rng):
            selected, indices = base.select_action_chunks(observations, rng)
            observations_seen.append(np.asarray(observations[0]))
            chunks_seen.append(np.asarray(selected[0]))
            return selected, indices

    evaluate(RecordingPolicy(), env, EvaluationConfig(episodes=episodes, seed=seed))
    sample_indices = np.linspace(
        0, len(observations_seen) - 1, min(128, len(observations_seen)), dtype=int
    )
    observations = jnp.asarray(np.stack(observations_seen)[sample_indices])
    selected = jnp.asarray(np.stack(chunks_seen)[sample_indices])
    variant = VARIANTS["qc_n8_grad"]

    @jax.jit
    def measure(observations, selected):
        center = jnp.clip(selected, -1.0, 1.0)
        refined = refine_action_chunks(
            state.critic.apply_fn,
            state.critic.params,
            observations,
            selected,
            steps=variant.gradient_steps,
            step_size=variant.step_size,
            radius=variant.radius,
            q_aggregation=state.config.q_aggregation,
        )
        before = state.critic.apply_fn({"params": state.critic.params}, observations, center)
        after = state.critic.apply_fn({"params": state.critic.params}, observations, refined)
        return before, after, jnp.linalg.norm(refined - center, axis=-1)

    before, after, distances = jax.device_get(measure(observations, selected))
    delta = np.asarray(aggregate_q_values(after, state.config.q_aggregation)) - np.asarray(
        aggregate_q_values(before, state.config.q_aggregation)
    )
    return {
        "baseline_episodes": episodes,
        "visited_chunks": len(chunks_seen),
        "sampled_chunks": len(sample_indices),
        "mean_q_before": float(np.mean(before)),
        "mean_q_after": float(np.mean(after)),
        "mean_q_delta": float(np.mean(delta)),
        "fraction_q_increased": float(np.mean(delta > 1e-5)),
        "mean_l2_action_change": float(np.mean(distances)),
        "max_l2_action_change": float(np.max(distances)),
        "critic_mean_deltas": [float(value) for value in np.mean(after - before, axis=1)],
    }


def run(
    config_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    *,
    episodes: int,
    seed: int,
    variants: list[str],
    diagnostics_episodes: int = 0,
) -> dict:
    experiment = load_experiment_config(config_path)
    env_config = experiment["environment"]
    env = make_ogbench_env(
        env_config["name"], sparse_reward=bool(env_config.get("sparse_reward", False))
    )
    try:
        agent_config = make_agent_config(experiment["agent"], env.action_space.shape[0])
        template = QCTrainState.create(
            seed=int(experiment.get("seed", 0)),
            observation_dim=env.observation_space.shape[0],
            config=agent_config,
        )
        state = load_checkpoint(checkpoint_path, template)
        result = {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "environment": env_config["name"],
            "sparse_reward": bool(env_config.get("sparse_reward", False)),
            "training_step": int(state.flow.step),
            "episodes": episodes,
            "seed": seed,
            "backend": jax.default_backend(),
            "versions": {
                "jax": jax.__version__,
                "ogbench": version("ogbench"),
                "gymnasium": version("gymnasium"),
                "mujoco": version("mujoco"),
            },
            "variants": {},
        }
        for name in variants:
            variant = VARIANTS[name]
            policy = make_policy(state, variant)
            # Compile outside the measured rollout time. All variants use the
            # same reset seeds and policy PRNG seed for paired evaluation.
            observation, _ = env.reset(seed=seed)
            dummy = jnp.asarray(observation, dtype=jnp.float32)[None, :]
            compile_start = time.perf_counter()
            warmup, _ = policy.select_action_chunks(dummy, jax.random.PRNGKey(seed))
            jax.block_until_ready(warmup)
            compile_seconds = time.perf_counter() - compile_start
            start = time.perf_counter()
            evaluation = evaluate(policy, env, EvaluationConfig(episodes=episodes, seed=seed))
            rollout_seconds = time.perf_counter() - start
            result["variants"][name] = {
                "settings": asdict(variant),
                "metrics": evaluation.as_metrics(),
                "successes": int(sum(episode.success for episode in evaluation.episodes)),
                "compile_seconds": compile_seconds,
                "rollout_seconds": rollout_seconds,
                "seconds_per_chunk": rollout_seconds
                / sum(episode.chunks_sampled for episode in evaluation.episodes),
                "episode_results": [asdict(episode) for episode in evaluation.episodes],
            }
            print(json.dumps({"variant": name, **result["variants"][name]["metrics"]}), flush=True)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if diagnostics_episodes:
            result["diagnostics"] = diagnose_refinement(
                state, env, episodes=diagnostics_episodes, seed=seed
            )
            output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, required=True)
    parser.add_argument("--diagnostics-episodes", type=int, default=0)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.diagnostics_episodes < 0:
        parser.error("--diagnostics-episodes must be nonnegative")
    run(
        args.config,
        args.checkpoint,
        args.output,
        episodes=args.episodes,
        seed=args.seed,
        variants=args.variants,
        diagnostics_episodes=args.diagnostics_episodes,
    )


if __name__ == "__main__":
    main()
