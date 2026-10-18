import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import ogbench

from qchunk.dataset import TransitionDataset

DEFAULT_DATASET_DIR = Path("~/.ogbench/data")


@dataclass(frozen=True)
class OGBenchBundle:
    env: Any
    eval_env: Any
    train_dataset: TransitionDataset
    validation_dataset: TransitionDataset


def convert_ogbench_dataset(
    data: Mapping[str, np.ndarray],
    action_clip_eps: float | None = 1e-5,
) -> TransitionDataset:
    """Convert a regular OGBench dataset into the QC transition format."""
    required = {
        "observations",
        "actions",
        "rewards",
        "next_observations",
        "masks",
        "terminals",
    }
    missing = required - data.keys()
    if missing:
        raise ValueError(f"OGBench dataset is missing fields: {sorted(missing)}")

    actions = np.asarray(data["actions"], dtype=np.float32)
    if action_clip_eps is not None:
        if not 0.0 <= action_clip_eps < 1.0:
            raise ValueError("action_clip_eps must lie in [0, 1)")
        actions = np.clip(actions, -1.0 + action_clip_eps, 1.0 - action_clip_eps)

    return TransitionDataset(
        {
            "observations": data["observations"],
            "actions": actions,
            "rewards": data["rewards"],
            "next_observations": data["next_observations"],
            "masks": data["masks"],
            "terminals": data["terminals"],
        }
    )


def make_ogbench_env(dataset_name: str):
    """Create an OGBench environment without downloading a dataset."""
    return ogbench.make_env_and_datasets(dataset_name, env_only=True)


def ogbench_dataset_paths(
    dataset_name: str,
    dataset_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Return the train/validation paths used internally by OGBench.

    OGBench environment names include the data collection policy and, for
    single-task data, the task number. Its downloaded archive names omit those
    components. Keeping this conversion here lets preflight inspect local data
    without invoking OGBench's downloader.
    """
    splits = dataset_name.split("-")
    if "singletask" in splits:
        position = splits.index("singletask")
        archive_name = "-".join(splits[:position] + splits[-1:])
    elif "oraclerep" in splits:
        archive_name = "-".join(splits[:-2] + splits[-1:])
    else:
        archive_name = "-".join(splits[:-2] + splits[-1:])

    root = Path(dataset_dir) if dataset_dir is not None else DEFAULT_DATASET_DIR
    root = root.expanduser()
    return root / f"{archive_name}.npz", root / f"{archive_name}-val.npz"


def capture_ogbench_env_state(env: Any) -> dict[str, Any]:
    """Capture MuJoCo and wrapper state required for mid-episode resume."""
    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "model") or not hasattr(unwrapped, "data"):
        raise TypeError("OGBench environment must expose MuJoCo model and data")

    specification = mujoco.mjtState.mjSTATE_FULLPHYSICS
    physics_state = np.empty(mujoco.mj_stateSize(unwrapped.model, specification), dtype=np.float64)
    mujoco.mj_getState(unwrapped.model, unwrapped.data, physics_state, specification)
    elapsed_steps = getattr(env, "_elapsed_steps", None)
    return {
        "physics_state": physics_state,
        "mocap_pos": unwrapped.data.mocap_pos.copy(),
        "mocap_quat": unwrapped.data.mocap_quat.copy(),
        "control": unwrapped.data.ctrl.copy(),
        # OGBench computes the next relative action from cached site kinematics.
        # mj_step may leave these at a different pipeline stage than mj_forward.
        "site_xpos": unwrapped.data.site_xpos.copy(),
        "site_xmat": unwrapped.data.site_xmat.copy(),
        "success": bool(getattr(unwrapped, "_success", False)),
        "elapsed_steps": -1 if elapsed_steps is None else int(elapsed_steps),
        "np_random_state_json": json.dumps(
            unwrapped.np_random.bit_generator.state,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def restore_ogbench_env_state(env: Any, state: Mapping[str, Any]) -> None:
    """Restore a snapshot produced by :func:`capture_ogbench_env_state`."""
    required = {
        "physics_state",
        "mocap_pos",
        "mocap_quat",
        "control",
        "site_xpos",
        "site_xmat",
        "success",
        "elapsed_steps",
        "np_random_state_json",
    }
    missing = required - state.keys()
    if missing:
        raise ValueError(f"OGBench environment state is missing fields: {sorted(missing)}")

    # Reset first so lazy MuJoCo model compilation and task setup are complete.
    env.reset()
    unwrapped = env.unwrapped
    specification = mujoco.mjtState.mjSTATE_FULLPHYSICS
    physics_state = np.asarray(state["physics_state"], dtype=np.float64)
    expected_size = mujoco.mj_stateSize(unwrapped.model, specification)
    if physics_state.shape != (expected_size,):
        raise ValueError("MuJoCo physics state has an incompatible shape")

    unwrapped.data.mocap_pos[:] = np.asarray(state["mocap_pos"])
    unwrapped.data.mocap_quat[:] = np.asarray(state["mocap_quat"])
    mujoco.mj_setState(unwrapped.model, unwrapped.data, physics_state, specification)
    unwrapped.data.ctrl[:] = np.asarray(state["control"])
    mujoco.mj_forward(unwrapped.model, unwrapped.data)
    # mj_forward refreshes derived kinematics but also recomputes warm-start
    # acceleration. Restore the full integration state once more so the next
    # mj_step starts from the exact saved solver state.
    mujoco.mj_setState(unwrapped.model, unwrapped.data, physics_state, specification)
    unwrapped.data.ctrl[:] = np.asarray(state["control"])
    unwrapped.data.site_xpos[:] = np.asarray(state["site_xpos"])
    unwrapped.data.site_xmat[:] = np.asarray(state["site_xmat"])
    unwrapped._success = bool(state["success"])
    unwrapped.np_random.bit_generator.state = json.loads(state["np_random_state_json"])

    elapsed_steps = int(state["elapsed_steps"])
    if hasattr(env, "_elapsed_steps"):
        env._elapsed_steps = None if elapsed_steps < 0 else elapsed_steps


def load_ogbench(
    dataset_name: str,
    dataset_dir: str | Path | None = None,
    action_clip_eps: float | None = 1e-5,
) -> OGBenchBundle:
    """Create independent train/eval environments and load regular datasets."""
    kwargs: dict[str, Any] = {"compact_dataset": False}
    if dataset_dir is not None:
        kwargs["dataset_dir"] = str(Path(dataset_dir).expanduser())

    env, train_data, validation_data = ogbench.make_env_and_datasets(dataset_name, **kwargs)
    eval_env = make_ogbench_env(dataset_name)
    try:
        if len(env.observation_space.shape) != 1:
            raise ValueError("Only state-based vector observations are currently supported")
        if len(env.action_space.shape) != 1:
            raise ValueError("Only continuous vector actions are currently supported")
        train_dataset = convert_ogbench_dataset(train_data, action_clip_eps)
        validation_dataset = convert_ogbench_dataset(validation_data, action_clip_eps)
    except Exception:
        env.close()
        eval_env.close()
        raise

    return OGBenchBundle(
        env=env,
        eval_env=eval_env,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
