import csv
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class MetricLogger:
    """Durable JSONL and long-form CSV metric writer for one run directory."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.csv_path = self.run_dir / "metrics.csv"

    def prepare_phase(self, phase: str, resume_step: int = 0) -> None:
        """Drop records after a resume point and rebuild both metric files."""
        if resume_step < 0:
            raise ValueError("resume_step must be non-negative")
        records = self.read_records()
        kept = [
            record
            for record in records
            if record.get("phase") != phase or int(record["step"]) <= resume_step
        ]
        self._rewrite(kept)

    def log(self, phase: str, step: int, metrics: dict[str, float]) -> None:
        if step < 0:
            raise ValueError("metric step must be non-negative")
        if not phase:
            raise ValueError("metric phase must not be empty")
        record = {
            "time": utc_now(),
            "phase": phase,
            "step": int(step),
            "metrics": {name: float(value) for name, value in metrics.items()},
        }
        with self.jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")
        csv_exists = self.csv_path.is_file() and self.csv_path.stat().st_size > 0
        with self.csv_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=("time", "phase", "step", "metric", "value"))
            if not csv_exists:
                writer.writeheader()
            for name, value in record["metrics"].items():
                writer.writerow(
                    {
                        "time": record["time"],
                        "phase": phase,
                        "step": step,
                        "metric": name,
                        "value": value,
                    }
                )

    def read_records(self) -> list[dict[str, Any]]:
        if not self.jsonl_path.is_file():
            return []
        records = []
        with self.jsonl_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSONL metric record at line {line_number}"
                    ) from error
                if (
                    not isinstance(record, dict)
                    or not {
                        "time",
                        "phase",
                        "step",
                        "metrics",
                    }
                    <= record.keys()
                ):
                    raise ValueError(f"Invalid metric record at line {line_number}")
                records.append(record)
        return records

    def _rewrite(self, records: list[dict[str, Any]]) -> None:
        jsonl_temporary = self.jsonl_path.with_suffix(".jsonl.tmp")
        csv_temporary = self.csv_path.with_suffix(".csv.tmp")
        with jsonl_temporary.open("w", encoding="utf-8") as jsonl_file:
            for record in records:
                jsonl_file.write(json.dumps(record, sort_keys=True) + "\n")
        with csv_temporary.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file, fieldnames=("time", "phase", "step", "metric", "value")
            )
            writer.writeheader()
            for record in records:
                for name, value in record["metrics"].items():
                    writer.writerow(
                        {
                            "time": record["time"],
                            "phase": record["phase"],
                            "step": record["step"],
                            "metric": name,
                            "value": value,
                        }
                    )
        jsonl_temporary.replace(self.jsonl_path)
        csv_temporary.replace(self.csv_path)


def record_manifest_event(
    run_dir: str | Path,
    experiment_config: dict[str, Any],
    phase: str,
    *,
    resume_checkpoint: str | Path | None = None,
    resume_step: int = 0,
) -> Path:
    """Create or update the reproducibility manifest for an experiment run."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise TypeError("manifest must contain a JSON object")
    else:
        manifest = {
            "schema_version": 1,
            "created_at": utc_now(),
            "runtime": collect_runtime_metadata(run_dir),
            "events": [],
        }

    event = {
        "time": utc_now(),
        "phase": phase,
        "resume_step": int(resume_step),
        "resume_checkpoint": (None if resume_checkpoint is None else str(Path(resume_checkpoint))),
        "argv": sys.argv.copy(),
        "config": deepcopy(experiment_config),
    }
    manifest.setdefault("events", []).append(event)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)
    return path


def collect_runtime_metadata(cwd: str | Path) -> dict[str, Any]:
    packages = {}
    for name in ("jax", "flax", "optax", "numpy", "ogbench", "gymnasium", "mujoco"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": packages,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "git": _git_metadata(cwd),
    }


def _git_metadata(cwd: str | Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        result = subprocess.run(
            ("git", *args),
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    commit = run("rev-parse", "HEAD")
    return {
        "commit": commit or os.environ.get("QCHUNK_SOURCE_COMMIT"),
        "branch": run("branch", "--show-current"),
        "dirty": None if status is None else bool(status),
    }


def run_directory(experiment_config: dict[str, Any]) -> Path:
    output = experiment_config["output"]
    return Path(output.get("run_dir", Path(output["checkpoint"]).parent))
