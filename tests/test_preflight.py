from pathlib import Path
from types import SimpleNamespace

import yaml

from qchunk.preflight import PreflightCheck, _check_devices, run_preflight


def write_config(tmp_path: Path, *, include_online: bool = True) -> Path:
    dataset_dir = tmp_path / "data"
    output_dir = tmp_path / "runs"
    config = {
        "seed": 0,
        "environment": {
            "name": "cube-triple-play-singletask-task2-v0",
            "dataset_dir": str(dataset_dir),
        },
        "agent": {
            "horizon": 2,
            "hidden_dims": [8],
            "num_candidates": 2,
            "num_flow_steps": 2,
        },
        "offline": {"steps": 2, "batch_size": 2},
        "evaluation": {"episodes": 1},
        "output": {
            "run_dir": str(output_dir),
            "checkpoint": str(output_dir / "offline.msgpack"),
        },
    }
    if include_online:
        config["online"] = {
            "steps": 2,
            "buffer_capacity": 10,
            "batch_size": 2,
            "start_training": 1,
        }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def patch_machine_checks(monkeypatch):
    monkeypatch.setattr(
        "qchunk.preflight._check_dependencies",
        lambda: PreflightCheck("dependencies", "pass", "ok"),
    )
    monkeypatch.setattr(
        "qchunk.preflight._check_devices",
        lambda require_gpu: PreflightCheck("jax_devices", "pass", "ok"),
    )
    monkeypatch.setattr(
        "qchunk.preflight._check_environment",
        lambda experiment: (PreflightCheck("environment", "pass", "ok"), 3, 2),
    )


def create_dataset_files(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    (dataset_dir / "cube-triple-play-v0.npz").write_bytes(b"train")
    (dataset_dir / "cube-triple-play-v0-val.npz").write_bytes(b"validation")


def check(report, name: str):
    return next(item for item in report.checks if item.name == name)


def test_offline_preflight_reports_missing_dataset(monkeypatch, tmp_path):
    patch_machine_checks(monkeypatch)
    report = run_preflight(write_config(tmp_path), minimum_free_gib=0)

    assert not report.ready
    assert check(report, "dataset").status == "fail"
    assert check(report, "checkpoint").status == "pass"
    assert report.as_dict()["summary"]["fail"] == 1


def test_offline_preflight_passes_with_local_archives(monkeypatch, tmp_path):
    patch_machine_checks(monkeypatch)
    create_dataset_files(tmp_path)

    report = run_preflight(write_config(tmp_path), minimum_free_gib=0)

    assert report.ready
    assert all(item.status == "pass" for item in report.checks)
    assert check(report, "hyperparameters").details["action_chunk_dim"] == 4


def test_online_preflight_requires_input_checkpoint(monkeypatch, tmp_path):
    patch_machine_checks(monkeypatch)
    create_dataset_files(tmp_path)

    report = run_preflight(write_config(tmp_path), phase="online", minimum_free_gib=0)

    assert not report.ready
    assert check(report, "checkpoint").status == "fail"
    assert "does not exist" in check(report, "checkpoint").message


def test_cpu_is_warning_unless_gpu_is_required(monkeypatch):
    cpu = SimpleNamespace(platform="cpu", __str__=lambda self: "CpuDevice(id=0)")
    monkeypatch.setattr("qchunk.preflight.jax.devices", lambda: [cpu])
    monkeypatch.setattr("qchunk.preflight.jax.default_backend", lambda: "cpu")

    assert _check_devices(require_gpu=False).status == "warning"
    assert _check_devices(require_gpu=True).status == "fail"
