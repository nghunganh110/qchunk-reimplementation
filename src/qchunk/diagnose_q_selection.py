"""Trace selected-chunk Q estimates and realized returns for QC best-of-N."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from qchunk.agent import sample_and_score_action_chunks
from qchunk.checkpoint import load_checkpoint
from qchunk.evaluate_refinement import Variant, make_policy
from qchunk.ogbench import make_ogbench_env
from qchunk.train_ogbench import load_experiment_config, make_agent_config
from qchunk.training import QCTrainState


def make_diagnostic_selector(state: QCTrainState, candidates: int):
    config = state.config
    # Keep action selection in the *exact* JIT used by ordinary evaluation.
    # Extra Q computations run separately and cannot change selected actions.
    policy = make_policy(state, Variant(f"qc_n{candidates}", candidates))

    @jax.jit
    def score_selected(observations, selected):
        return state.critic.apply_fn({"params": state.critic.params}, observations, selected)

    @jax.jit
    def candidate_mean(observations, rng):
        _, q_values = sample_and_score_action_chunks(
            state.flow.apply_fn,
            state.flow.params,
            state.critic.apply_fn,
            state.critic.params,
            observations,
            rng,
            action_chunk_dim=config.action_chunk_dim,
            num_candidates=candidates,
            num_flow_steps=config.num_flow_steps,
        )
        return jnp.mean(jnp.mean(q_values, axis=0), axis=-1)

    return policy.select_action_chunks, score_selected, candidate_mean


def rollout_with_q_trace(state, env, select, score_selected, candidate_mean, rng, reset_seed):
    observation, _ = env.reset(seed=reset_seed)
    rewards = []
    trace = []
    success = 0.0
    done = False
    while not done:
        rng, action_rng = jax.random.split(rng)
        observations = jnp.asarray(observation, dtype=jnp.float32)[None, :]
        selected, _ = select(observations, action_rng)
        selected_q = score_selected(observations, selected)
        chunk = np.asarray(jax.device_get(selected[0])).reshape(
            state.config.horizon, state.config.action_dim
        )
        q_pair = np.asarray(jax.device_get(selected_q[:, 0]))
        q_mean = float(np.mean(q_pair))
        record = {
            "start_step": len(rewards),
            "q_values": [float(value) for value in q_pair],
            "q_mean": q_mean,
            "q_disagreement": float(abs(q_pair[0] - q_pair[1])),
        }
        if not trace:
            record["candidate_mean_q"] = float(
                jax.device_get(candidate_mean(observations, action_rng)[0])
            )
        trace.append(record)
        for action in chunk:
            observation, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            success = max(success, float(info.get("success", 0.0)))
            done = bool(terminated or truncated)
            if done:
                break

    discounted = np.empty(len(rewards) + 1, dtype=np.float64)
    discounted[-1] = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        discounted[index] = rewards[index] + state.config.discount * discounted[index + 1]
    for record in trace:
        record["discounted_return"] = float(discounted[record["start_step"]])
        record["optimism"] = record["q_mean"] - record["discounted_return"]
        if "candidate_mean_q" in record:
            record["winner_gap"] = record["q_mean"] - record["candidate_mean_q"]
    return {
        "success": success,
        "length": len(rewards),
        "episode_return": float(sum(rewards)),
        "chunks": trace,
    }, rng


def summarize(episodes):
    first = [episode["chunks"][0] for episode in episodes]
    all_chunks = [chunk for episode in episodes for chunk in episode["chunks"]]
    return {
        "success_rate": float(np.mean([episode["success"] for episode in episodes])),
        "successes": int(sum(episode["success"] for episode in episodes)),
        "mean_length": float(np.mean([episode["length"] for episode in episodes])),
        "mean_return": float(np.mean([episode["episode_return"] for episode in episodes])),
        "total_chunks": len(all_chunks),
        "first_q_mean": float(np.mean([chunk["q_mean"] for chunk in first])),
        "first_discounted_return_mean": float(
            np.mean([chunk["discounted_return"] for chunk in first])
        ),
        "first_optimism_mean": float(np.mean([chunk["optimism"] for chunk in first])),
        "first_optimism_median": float(np.median([chunk["optimism"] for chunk in first])),
        "first_optimistic_fraction": float(np.mean([chunk["optimism"] > 0 for chunk in first])),
        "first_q_disagreement_mean": float(np.mean([chunk["q_disagreement"] for chunk in first])),
        "first_winner_gap_mean": float(np.mean([chunk["winner_gap"] for chunk in first])),
        "chunk_optimism_mean": float(np.mean([chunk["optimism"] for chunk in all_chunks])),
        "chunk_q_disagreement_mean": float(
            np.mean([chunk["q_disagreement"] for chunk in all_chunks])
        ),
    }


def run(
    config_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    episodes: int,
    seed: int,
    candidate_counts: list[int],
):
    experiment = load_experiment_config(config_path)
    env_config = experiment["environment"]
    env = make_ogbench_env(
        env_config["name"], sparse_reward=bool(env_config.get("sparse_reward", False))
    )
    try:
        config = make_agent_config(experiment["agent"], env.action_space.shape[0])
        template = QCTrainState.create(
            seed=int(experiment.get("seed", 0)),
            observation_dim=env.observation_space.shape[0],
            config=config,
        )
        state = load_checkpoint(checkpoint_path, template)
        result = {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "training_step": int(state.flow.step),
            "seed": seed,
            "episodes": episodes,
            "discount": state.config.discount,
            "candidate_counts": candidate_counts,
            "rng_protocol": "independent policy PRNG key per episode via fold_in(seed, index)",
            "policies": {},
            "notes": (
                "Q is compared with the finite-episode discounted return under the evaluated "
                "policy. Time-limit truncation and policy mismatch to the N=64 training backup "
                "can affect the residual. This diagnostic uses a fresh environment per policy "
                "and per-episode policy RNG, unlike the earlier evaluation script."
            ),
        }
        for candidates in candidate_counts:
            select, score_selected, candidate_mean = make_diagnostic_selector(state, candidates)
            policy_env = make_ogbench_env(
                env_config["name"], sparse_reward=bool(env_config.get("sparse_reward", False))
            )
            try:
                observation, _ = policy_env.reset(seed=seed)
                warmup = select(
                    jnp.asarray(observation, dtype=jnp.float32)[None, :],
                    jax.random.PRNGKey(seed),
                )[0]
                jax.block_until_ready(warmup)
                records = []
                for episode_index in range(episodes):
                    episode_rng = jax.random.fold_in(jax.random.PRNGKey(seed), episode_index)
                    episode, _ = rollout_with_q_trace(
                        state,
                        policy_env,
                        select,
                        score_selected,
                        candidate_mean,
                        episode_rng,
                        reset_seed=seed + episode_index,
                    )
                    records.append(episode)
            finally:
                policy_env.close()
            result["policies"][f"n{candidates}"] = {
                "summary": summarize(records),
                "episodes": records,
            }
            print(
                json.dumps(
                    {"candidates": candidates, **result["policies"][f"n{candidates}"]["summary"]}
                ),
                flush=True,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if args.episodes <= 0 or any(count <= 0 for count in args.candidates):
        parser.error("episodes and all candidate counts must be positive")
    run(args.config, args.checkpoint, args.output, args.episodes, args.seed, args.candidates)


if __name__ == "__main__":
    main()
