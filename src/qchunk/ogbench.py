from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import ogbench

from qchunk.dataset import TransitionDataset


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
