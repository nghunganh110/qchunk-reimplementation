"""Cost-capped Modal launcher for the first QC OGBench experiment."""

import json
import os
import subprocess
import time
from pathlib import Path

import modal

from qchunk.modal_budget import (
    PRICE_SNAPSHOT_DATE,
    ModalResources,
    budget_timeout_seconds,
    estimate_cost,
)

APP_NAME = "qchunk-cube-triple"
VOLUME_NAME = "qchunk-experiments"
VOLUME_ROOT = Path("/vol")
REMOTE_CONFIG = Path("/root/configs/cube_triple.yaml")
CPU_CORES = 4.0
MEMORY_MIB = 32 * 1024
DEFAULT_BUDGET_USD = 30.0
DEFAULT_MAX_SPEND_USD = 8.0
DEFAULT_BENCHMARK_STEPS = 2_000
DEFAULT_TARGET_STEPS = 1_000_000


def _source_commit() -> str:
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return os.environ.get("QCHUNK_SOURCE_COMMIT", "unknown")
    return (
        result.stdout.strip()
        if result.returncode == 0
        else os.environ.get("QCHUNK_SOURCE_COMMIT", "unknown")
    )


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libegl1", "libgl1", "libglfw3")
    .uv_sync()
    .env(
        {
            "MUJOCO_GL": "egl",
            "PYTHONUNBUFFERED": "1",
            "QCHUNK_SOURCE_COMMIT": _source_commit(),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
        }
    )
    .workdir("/root")
    .add_local_python_source("qchunk")
    .add_local_dir("configs", remote_path="/root/configs")
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _runtime_config(
    run_name: str,
    target_steps: int,
    *,
    online_target_steps: int | None = None,
    evaluation_episodes: int | None = None,
) -> Path:
    import yaml

    from qchunk.train_ogbench import load_experiment_config

    experiment = load_experiment_config(REMOTE_CONFIG)
    experiment["environment"]["dataset_dir"] = str(VOLUME_ROOT / "data")
    experiment["offline"]["steps"] = target_steps
    if online_target_steps is not None:
        experiment["online"]["steps"] = online_target_steps
    if evaluation_episodes is not None:
        experiment["evaluation"]["episodes"] = evaluation_episodes
    run_dir = VOLUME_ROOT / "runs" / run_name
    experiment["output"] = {
        "run_dir": str(run_dir),
        "checkpoint": str(run_dir / "offline_final.msgpack"),
        "checkpoint_dir": str(run_dir / "checkpoints"),
        "online_checkpoint": str(run_dir / "online_final.msgpack"),
        "online_resume_checkpoint": str(run_dir / "online_resume"),
    }
    path = Path("/tmp") / f"{run_name.replace('/', '_')}.yaml"
    path.write_text(yaml.safe_dump(experiment, sort_keys=False), encoding="utf-8")
    return path


def _assert_gpu() -> dict[str, object]:
    import jax

    devices = jax.devices()
    details = {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
    }
    print(json.dumps({"jax": details}, sort_keys=True), flush=True)
    if not any(device.platform in {"cuda", "gpu", "rocm"} for device in devices):
        raise RuntimeError(f"Modal worker has no JAX GPU device: {details}")
    return details


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    cpu=2.0,
    memory=8 * 1024,
    timeout=2 * 60 * 60,
)
def prepare_data() -> dict[str, object]:
    """Download and CRC-check OGBench archives on cheap CPU resources."""
    from qchunk.ogbench import download_ogbench_dataset_archives

    started = time.monotonic()
    paths = download_ogbench_dataset_archives(
        "cube-triple-play-singletask-task2-v0",
        VOLUME_ROOT / "data",
    )
    volume.commit()
    result = {
        "dataset_paths": [str(path) for path in paths],
        "sizes_bytes": [path.stat().st_size for path in paths],
        "elapsed_seconds": time.monotonic() - started,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    cpu=CPU_CORES,
    memory=MEMORY_MIB,
    timeout=24 * 60 * 60,
)
def run_gpu(
    mode: str,
    steps: int,
    gpu: str,
    evaluation_episodes: int = 50,
    simulation_seed: int = 42,
    simulation_frame_stride: int = 5,
    simulation_fps: int = 10,
) -> dict[str, object]:
    """Benchmark, train, evaluate, or record the QC agent on a GPU worker."""
    jax_details = _assert_gpu()
    resources = ModalResources(gpu=gpu, cpu_cores=CPU_CORES, memory_gib=MEMORY_MIB / 1024)

    if mode == "benchmark":
        result = _benchmark_offline(steps, resources)
    elif mode == "offline":
        result = _train_offline(steps, resources)
    elif mode == "evaluation":
        result = _evaluate_offline(steps, resources)
    elif mode == "online":
        result = _train_online(steps, evaluation_episodes, resources)
    elif mode == "simulation":
        result = _simulate_online(
            simulation_seed,
            simulation_frame_stride,
            simulation_fps,
            resources,
        )
    else:
        raise ValueError(
            "mode must be 'benchmark', 'offline', 'online', 'evaluation', or 'simulation'"
        )
    result["jax"] = jax_details
    result["gpu"] = gpu
    result["price_snapshot_date"] = PRICE_SNAPSHOT_DATE
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _benchmark_offline(timed_steps: int, resources: ModalResources) -> dict[str, object]:
    import jax
    import numpy as np

    from qchunk.offline import OfflineTrainingConfig, train_offline
    from qchunk.ogbench import load_ogbench
    from qchunk.train_ogbench import load_experiment_config, make_agent_config
    from qchunk.training import QCTrainState

    if timed_steps <= 0:
        raise ValueError("benchmark steps must be positive")
    config_path = _runtime_config("modal_benchmark", timed_steps + 1)
    experiment = load_experiment_config(config_path)
    environment = experiment["environment"]
    bundle = load_ogbench(
        environment["name"],
        dataset_dir=environment["dataset_dir"],
        action_clip_eps=environment.get("action_clip_eps", 1e-5),
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

        offline_values["steps"] = timed_steps + 1
        offline_values["log_interval"] = timed_steps + 1
        started = time.monotonic()
        measured = train_offline(
            warmup.state,
            bundle.train_dataset,
            OfflineTrainingConfig(**offline_values),
            initial_step=1,
            dataset_rng_state=warmup.dataset_rng_state,
        )
        jax.block_until_ready(measured.state.critic.params)
        elapsed = time.monotonic() - started
        metrics = measured.records[-1].metrics
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"benchmark produced non-finite metrics: {metrics}")
    finally:
        bundle.env.close()
        bundle.eval_env.close()

    seconds_per_update = elapsed / timed_steps
    projected_seconds = seconds_per_update * DEFAULT_TARGET_STEPS
    return {
        "mode": "benchmark",
        "timed_steps": timed_steps,
        "elapsed_seconds": elapsed,
        "seconds_per_update": seconds_per_update,
        "updates_per_second": 1.0 / seconds_per_update,
        "projected_offline_hours": projected_seconds / 3600.0,
        "projected_offline_cost_usd": estimate_cost(resources, projected_seconds),
        "estimated_usd_per_hour": resources.usd_per_hour,
        "metrics": metrics,
        "projection_target_steps": DEFAULT_TARGET_STEPS,
        "projection_scope": "offline learner only; excludes online fine-tuning and image build",
    }


def _latest_checkpoint(run_dir: Path) -> Path | None:
    candidates = [run_dir / "offline_final.msgpack"]
    candidates.extend((run_dir / "checkpoints").glob("offline_step_*.msgpack"))
    existing = [path for path in candidates if path.is_file()]
    return max(existing, key=lambda path: path.stat().st_mtime_ns) if existing else None


def _train_offline(target_steps: int, resources: ModalResources) -> dict[str, object]:
    from qchunk.train_ogbench import run

    if target_steps <= 0:
        raise ValueError("target steps must be positive")
    run_name = "cube_triple"
    run_dir = VOLUME_ROOT / "runs" / run_name
    config_path = _runtime_config(run_name, target_steps)
    resume_path = _latest_checkpoint(run_dir)
    started = time.monotonic()
    try:
        checkpoint_path = run(
            config_path,
            steps_override=target_steps,
            resume_path=resume_path,
        )
    finally:
        volume.commit()
    elapsed = time.monotonic() - started
    return {
        "mode": "offline",
        "target_steps": target_steps,
        "resumed_from": None if resume_path is None else str(resume_path),
        "checkpoint": str(checkpoint_path),
        "elapsed_seconds": elapsed,
        "actual_cost_estimate_usd": estimate_cost(resources, elapsed),
        "estimated_usd_per_hour": resources.usd_per_hour,
    }


def _evaluate_offline(episodes: int, resources: ModalResources) -> dict[str, object]:
    from qchunk.evaluate_ogbench import run

    if episodes <= 0:
        raise ValueError("evaluation episodes must be positive")
    run_name = "cube_triple"
    run_dir = VOLUME_ROOT / "runs" / run_name
    checkpoint_path = run_dir / "offline_final.msgpack"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"offline checkpoint does not exist: {checkpoint_path}")

    config_path = _runtime_config(run_name, DEFAULT_TARGET_STEPS)
    started = time.monotonic()
    try:
        metrics = run(
            config_path,
            checkpoint_path=checkpoint_path,
            episodes_override=episodes,
        )
    finally:
        volume.commit()
    elapsed = time.monotonic() - started
    return {
        "mode": "evaluation",
        "episodes": episodes,
        "checkpoint": str(checkpoint_path),
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "actual_cost_estimate_usd": estimate_cost(resources, elapsed),
        "estimated_usd_per_hour": resources.usd_per_hour,
    }


def _simulate_online(
    seed: int,
    frame_stride: int,
    fps: int,
    resources: ModalResources,
) -> dict[str, object]:
    from qchunk.simulate_ogbench import run

    run_name = "cube_triple"
    run_dir = VOLUME_ROOT / "runs" / run_name
    checkpoint_path = run_dir / "online_final.msgpack"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"online checkpoint does not exist: {checkpoint_path}")

    output_path = run_dir / "simulations" / f"online_seed_{seed}.gif"
    config_path = _runtime_config(run_name, DEFAULT_TARGET_STEPS)
    started = time.monotonic()
    try:
        metrics = run(
            config_path,
            output_path,
            checkpoint_path,
            seed=seed,
            frame_stride=frame_stride,
            fps=fps,
        )
    finally:
        volume.commit()
    elapsed = time.monotonic() - started
    return {
        "mode": "simulation",
        "checkpoint": str(checkpoint_path),
        "output": str(output_path),
        "size_bytes": output_path.stat().st_size,
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "actual_cost_estimate_usd": estimate_cost(resources, elapsed),
        "estimated_usd_per_hour": resources.usd_per_hour,
    }


def _online_resume_step(path: Path) -> int | None:
    from flax import serialization

    metadata_path = path / "metadata.msgpack"
    if not metadata_path.is_file():
        return None
    payload = serialization.msgpack_restore(metadata_path.read_bytes())
    try:
        return int(payload["online"]["environment_steps"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid online resume checkpoint metadata: {metadata_path}") from error


def _phase_metrics(run_dir: Path, phase: str, step: int) -> dict[str, float] | None:
    from qchunk.tracking import MetricLogger

    records = MetricLogger(run_dir).read_records()
    for record in reversed(records):
        if record["phase"] == phase and int(record["step"]) == step:
            return {name: float(value) for name, value in record["metrics"].items()}
    return None


def _train_online(
    target_steps: int,
    evaluation_episodes: int,
    resources: ModalResources,
) -> dict[str, object]:
    from qchunk.train_online_ogbench import run

    if target_steps <= 0:
        raise ValueError("online target steps must be positive")
    if evaluation_episodes <= 0:
        raise ValueError("evaluation episodes must be positive")

    run_name = "cube_triple"
    run_dir = VOLUME_ROOT / "runs" / run_name
    offline_checkpoint = run_dir / "offline_final.msgpack"
    if not offline_checkpoint.is_file():
        raise FileNotFoundError(f"offline checkpoint does not exist: {offline_checkpoint}")

    resume_path = run_dir / "online_resume"
    resume_step = _online_resume_step(resume_path)
    if resume_step is not None and resume_step >= target_steps:
        raise ValueError(
            f"online checkpoint is already at step {resume_step}, which is not below "
            f"the requested target {target_steps}"
        )

    config_path = _runtime_config(
        run_name,
        DEFAULT_TARGET_STEPS,
        online_target_steps=target_steps,
        evaluation_episodes=evaluation_episodes,
    )
    started = time.monotonic()
    try:
        checkpoint_path = run(
            config_path,
            checkpoint_path=offline_checkpoint,
            steps_override=target_steps,
            evaluation_episodes_override=evaluation_episodes,
            resume_path=resume_path if resume_step is not None else None,
            on_checkpoint_saved=lambda _path: volume.commit(),
        )
    finally:
        volume.commit()
    elapsed = time.monotonic() - started
    initial_step = 0 if resume_step is None else resume_step
    completed_steps = target_steps - initial_step
    seconds_per_step = elapsed / completed_steps
    projected_seconds = seconds_per_step * DEFAULT_TARGET_STEPS
    return {
        "mode": "online",
        "target_steps": target_steps,
        "initial_step": initial_step,
        "completed_steps": completed_steps,
        "evaluation_episodes": evaluation_episodes,
        "resumed_from": None if resume_step is None else str(resume_path),
        "checkpoint": str(checkpoint_path),
        "resume_checkpoint": str(resume_path),
        "training_metrics": _phase_metrics(run_dir, "online", target_steps),
        "evaluation_metrics": _phase_metrics(run_dir, "online_evaluation", target_steps),
        "elapsed_seconds": elapsed,
        "seconds_per_environment_step": seconds_per_step,
        "actual_cost_estimate_usd": estimate_cost(resources, elapsed),
        "projected_million_step_hours_from_pilot": projected_seconds / 3600.0,
        "projected_million_step_cost_usd_from_pilot": estimate_cost(
            resources, projected_seconds
        ),
        "estimated_usd_per_hour": resources.usd_per_hour,
        "projection_scope": (
            "linear pilot extrapolation including pilot startup, final evaluation, and checkpoint I/O"
        ),
    }


@app.local_entrypoint()
def main(
    mode: str = "benchmark",
    gpu: str = "A10",
    benchmark_steps: int = DEFAULT_BENCHMARK_STEPS,
    target_steps: int = DEFAULT_TARGET_STEPS,
    online_target_steps: int = DEFAULT_TARGET_STEPS,
    evaluation_episodes: int = 50,
    simulation_seed: int = 42,
    simulation_frame_stride: int = 5,
    simulation_fps: int = 10,
    budget_usd: float = DEFAULT_BUDGET_USD,
    max_spend_usd: float = DEFAULT_MAX_SPEND_USD,
) -> None:
    """Prepare data, then benchmark, train, evaluate, or record with a cost cap."""
    if mode not in {"prepare", "benchmark", "offline", "online", "evaluation", "simulation"}:
        raise ValueError(
            "mode must be prepare, benchmark, offline, online, evaluation, or simulation"
        )
    if budget_usd <= 0:
        raise ValueError("budget_usd must be positive")
    if not 0 < max_spend_usd <= budget_usd:
        raise ValueError("max_spend_usd must lie in (0, budget_usd]")
    resources = ModalResources(gpu=gpu, cpu_cores=CPU_CORES, memory_gib=MEMORY_MIB / 1024)
    resources.validate()

    print(
        json.dumps(
            {
                "gpu": gpu,
                "estimated_usd_per_hour": resources.usd_per_hour,
                "budget_usd": budget_usd,
                "max_spend_usd": max_spend_usd,
                "price_snapshot_date": PRICE_SNAPSHOT_DATE,
            },
            sort_keys=True,
        )
    )
    if mode in {"prepare", "benchmark", "offline", "online"}:
        prepare_data.remote()
    if mode == "prepare":
        return

    if mode == "benchmark":
        timeout = min(2 * 60 * 60, budget_timeout_seconds(resources, max_spend_usd))
        result = run_gpu.with_options(gpu=gpu, timeout=timeout).remote(
            "benchmark", benchmark_steps, gpu
        )
    elif mode == "offline":
        timeout = budget_timeout_seconds(resources, max_spend_usd)
        print(
            f"Launching up to {timeout / 3600:.2f} hours; the configured resource ceiling is "
            f"${estimate_cost(resources, timeout):.2f}.",
            flush=True,
        )
        call = run_gpu.with_options(gpu=gpu, timeout=timeout).spawn("offline", target_steps, gpu)
        result = call.get()
    elif mode == "evaluation":
        timeout = budget_timeout_seconds(resources, max_spend_usd)
        print(
            f"Evaluating {evaluation_episodes} episodes for up to {timeout / 60:.1f} minutes; "
            f"the configured resource ceiling is ${estimate_cost(resources, timeout):.2f}.",
            flush=True,
        )
        result = run_gpu.with_options(gpu=gpu, timeout=timeout).remote(
            "evaluation", evaluation_episodes, gpu
        )
    elif mode == "simulation":
        timeout = min(2 * 60 * 60, budget_timeout_seconds(resources, max_spend_usd))
        print(
            f"Recording seed {simulation_seed} for up to {timeout / 60:.1f} minutes; "
            f"the configured resource ceiling is ${estimate_cost(resources, timeout):.2f}.",
            flush=True,
        )
        result = run_gpu.with_options(gpu=gpu, timeout=timeout).remote(
            "simulation",
            1,
            gpu,
            evaluation_episodes,
            simulation_seed,
            simulation_frame_stride,
            simulation_fps,
        )
    else:
        timeout = budget_timeout_seconds(resources, max_spend_usd)
        print(
            f"Fine-tuning online toward step {online_target_steps} with "
            f"{evaluation_episodes} evaluation episode(s) for up to {timeout / 3600:.2f} hours; "
            f"the configured resource ceiling is ${estimate_cost(resources, timeout):.2f}.",
            flush=True,
        )
        call = run_gpu.with_options(gpu=gpu, timeout=timeout).spawn(
            "online", online_target_steps, gpu, evaluation_episodes
        )
        result = call.get()
    print(json.dumps(result, indent=2, sort_keys=True))
