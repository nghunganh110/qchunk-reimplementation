from types import SimpleNamespace

import numpy as np
import yaml

from qchunk.benchmark_ogbench import run
from qchunk.dataset import TransitionDataset


class FakeEnv:
    observation_space = SimpleNamespace(shape=(3,))
    action_space = SimpleNamespace(shape=(2,))

    def close(self):
        pass


def test_benchmark_excludes_warmup_and_projects_target(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    dataset = TransitionDataset(
        {
            "observations": rng.normal(size=(20, 3)),
            "actions": rng.normal(size=(20, 2)),
            "rewards": rng.normal(size=20),
            "next_observations": rng.normal(size=(20, 3)),
            "masks": np.ones(20),
            "terminals": np.array([0] * 9 + [1] + [0] * 9 + [1]),
        }
    )
    bundle = SimpleNamespace(
        env=FakeEnv(),
        eval_env=FakeEnv(),
        train_dataset=dataset,
    )
    monkeypatch.setattr("qchunk.benchmark_ogbench.load_ogbench", lambda *args, **kwargs: bundle)
    times = iter((10.0, 12.0))
    monkeypatch.setattr("qchunk.benchmark_ogbench.time.perf_counter", lambda: next(times))

    config = {
        "seed": 0,
        "environment": {"name": "fake-v0", "sparse_reward": True},
        "agent": {
            "horizon": 2,
            "hidden_dims": [8],
            "num_candidates": 2,
            "num_flow_steps": 2,
        },
        "offline": {
            "steps": 10,
            "batch_size": 4,
            "log_interval": 5,
            "dataset_seed": 0,
            "checkpoint_interval": 5,
        },
        "output": {"checkpoint": str(tmp_path / "checkpoint.msgpack")},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run(config_path, timed_steps=2)

    assert result["timed_steps"] == 2
    assert result["elapsed_seconds"] == 2.0
    assert result["updates_per_second"] == 1.0
    assert result["projection_target_steps"] == 10
    assert result["projected_offline_hours"] == 10 / 3600
