"""Build the self-contained Kaggle notebook shipped with this repository."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "notebooks" / "qchunk_kaggle.ipynb"


def _source_bundle() -> tuple[str, str]:
    paths = [ROOT / "pyproject.toml"]
    paths.extend(sorted((ROOT / "src").rglob("*.py")))
    paths.extend(sorted((ROOT / "configs").glob("*.yaml")))

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in paths:
            info = archive.gettarinfo(str(path), arcname=str(path.relative_to(ROOT)))
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as file:
                archive.addfile(info, file)
    payload = buffer.getvalue()
    return base64.b64encode(payload).decode("ascii"), hashlib.sha256(payload).hexdigest()


def _markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def _code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build() -> Path:
    payload, digest = _source_bundle()
    cells = [
        _markdown(
            """# Q-chunking trên Kaggle GPU

Notebook này tự chứa source code của dự án, mặc định chạy **Puzzle 3×3 Sparse – task 4**.

Trước khi chạy:

1. Mở **Settings → Accelerator → GPU**; ưu tiên P100 nếu Kaggle cho phép chọn.
2. Bật **Internet** để cài package và tải OGBench lần đầu.
3. Giữ `MODE = \"benchmark\"` cho lần chạy đầu. Khi benchmark ổn, đổi lần lượt sang
   `offline`, `evaluation`, `online`, `evaluation`, rồi `simulation`.
4. Mỗi phase dài nên chạy trong một Kaggle Version riêng. Kaggle lưu tối đa 20 GB từ
   `/kaggle/working`; gắn output của version trước làm Input và điền `PREVIOUS_RUN_DIR`
   để tiếp tục.

`benchmark` chỉ đo thử 2.000 update và **không** ghi đè checkpoint chính.
"""
        ),
        _code(
            """# Cấu hình duy nhất bạn thường cần chỉnh.
MODE = "benchmark"  # benchmark | offline | evaluation | online | simulation
EXPERIMENT = "puzzle_3x3"  # puzzle_3x3 | cube_triple

BENCHMARK_STEPS = 2_000
OFFLINE_STEPS = 1_000_000
ONLINE_STEPS = 1_000_000
EVALUATION_EPISODES = 50
SIMULATION_SEED = 42

# Khi tiếp tục từ output của một Kaggle Version trước, điền đường dẫn thư mục run.
# Ví dụ: /kaggle/input/<ten-output>/runs/puzzle_3x3_task4
PREVIOUS_RUN_DIR = ""

# Có thể trỏ tới thư mục ogbench_data của output trước để khỏi tải dataset lại.
# Ví dụ: /kaggle/input/<ten-output>/ogbench_data
PREVIOUS_DATASET_DIR = ""

assert MODE in {"benchmark", "offline", "evaluation", "online", "simulation"}
assert EXPERIMENT in {"puzzle_3x3", "cube_triple"}
"""
        ),
        _code(
            """import os
import platform
import subprocess
import sys

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["PYTHONUNBUFFERED"] = "1"

print("Python:", sys.version)
print("Platform:", platform.platform())
subprocess.run(["nvidia-smi"], check=False)
"""
        ),
        _code(
            f'''# Giải nén source code được nhúng trong chính notebook.
import base64
import io
import shutil
import tarfile
from pathlib import Path

SOURCE_SHA256 = "{digest}"
SOURCE_B64 = """{payload}"""
PROJECT_DIR = Path("/kaggle/working/qchunk-reimplementation")

if PROJECT_DIR.exists():
    shutil.rmtree(PROJECT_DIR)
PROJECT_DIR.mkdir(parents=True)
archive_bytes = base64.b64decode(SOURCE_B64)
assert __import__("hashlib").sha256(archive_bytes).hexdigest() == SOURCE_SHA256
with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
    archive.extractall(PROJECT_DIR, filter="data")
print("Đã giải nén source:", PROJECT_DIR)
print("Source bundle SHA-256:", SOURCE_SHA256)
'''
        ),
        _code(
            """# Dùng JAX GPU có sẵn của Kaggle; chỉ cài các dependency còn lại.
dependencies = [
    "flax==0.12.10",
    "optax==0.2.8",
    "ogbench==1.1.2",
    "gymnasium==1.1.1",
    "mujoco==3.3.1",
    "imageio==2.38.1",
    "PyYAML==6.0.3",
]
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", *dependencies],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--no-deps",
        "--no-build-isolation",
        "-e",
        str(PROJECT_DIR),
    ],
    check=True,
)
"""
        ),
        _code(
            """# Dừng ngay nếu notebook chưa thật sự có GPU JAX.
import jax
import jax.numpy as jnp

devices = jax.devices()
print("JAX:", jax.__version__)
print("Backend:", jax.default_backend())
print("Devices:", devices)
assert any(device.platform in {"cuda", "gpu", "rocm"} for device in devices), (
    "Không thấy GPU JAX. Hãy bật Settings > Accelerator > GPU rồi restart session."
)
print("GPU smoke test:", jnp.arange(4.0).devices())
"""
        ),
        _code(
            """# Tạo config dùng đường dẫn bền vững trong /kaggle/working.
import yaml

WORK_ROOT = Path("/kaggle/working")
CONFIG_NAME = {
    "puzzle_3x3": "puzzle_3x3.yaml",
    "cube_triple": "cube_triple.yaml",
}[EXPERIMENT]

with (PROJECT_DIR / "configs" / CONFIG_NAME).open(encoding="utf-8") as file:
    config = yaml.safe_load(file)

run_name = Path(config["output"]["run_dir"]).name
if MODE == "benchmark":
    run_name += "_benchmark"
RUN_DIR = WORK_ROOT / "runs" / run_name

if PREVIOUS_RUN_DIR and MODE != "benchmark":
    previous = Path(PREVIOUS_RUN_DIR)
    if not previous.is_dir():
        raise FileNotFoundError(f"PREVIOUS_RUN_DIR không tồn tại: {previous}")
    shutil.copytree(previous, RUN_DIR, dirs_exist_ok=True)
    print("Đã phục hồi run từ:", previous)

DATASET_DIR = Path(PREVIOUS_DATASET_DIR) if PREVIOUS_DATASET_DIR else WORK_ROOT / "ogbench_data"
if PREVIOUS_DATASET_DIR and not DATASET_DIR.is_dir():
    raise FileNotFoundError(f"PREVIOUS_DATASET_DIR không tồn tại: {DATASET_DIR}")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

config["environment"]["dataset_dir"] = str(DATASET_DIR)
config["offline"]["steps"] = OFFLINE_STEPS
config["online"]["steps"] = ONLINE_STEPS
config["evaluation"]["episodes"] = EVALUATION_EPISODES
config["output"] = {
    "run_dir": str(RUN_DIR),
    "checkpoint": str(RUN_DIR / "offline_final.msgpack"),
    "checkpoint_dir": str(RUN_DIR / "checkpoints"),
    "online_checkpoint": str(RUN_DIR / "online_final.msgpack"),
    "online_resume_checkpoint": str(RUN_DIR / "online_resume"),
}
if MODE == "benchmark":
    config["offline"]["checkpoint_interval"] = None
    config["offline"]["log_interval"] = BENCHMARK_STEPS

RUNTIME_CONFIG = WORK_ROOT / f"{run_name}.yaml"
RUNTIME_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(RUNTIME_CONFIG.read_text())
"""
        ),
        _code(
            """# Tải/kiểm tra hai archive OGBench rồi chạy preflight.
os.chdir(PROJECT_DIR)
from qchunk.ogbench import download_ogbench_dataset_archives

dataset_paths = download_ogbench_dataset_archives(
    config["environment"]["name"],
    DATASET_DIR,
)
print("Dataset:")
for path in dataset_paths:
    print(" ", path, f"({path.stat().st_size / 2**30:.2f} GiB)")
"""
        ),
        _code(
            """# Chạy phase đã chọn và tự động dùng checkpoint hiện có khi có thể.
import time


def run_command(arguments):
    command = [sys.executable, *map(str, arguments)]
    print("$", " ".join(command), flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=PROJECT_DIR, check=True)
    elapsed = time.perf_counter() - started
    print(f"Hoàn tất sau {elapsed / 60:.2f} phút")
    return elapsed


def latest_offline_checkpoint():
    candidates = [RUN_DIR / "offline_final.msgpack"]
    candidates.extend((RUN_DIR / "checkpoints").glob("offline_step_*.msgpack"))
    existing = [path for path in candidates if path.is_file()]
    return max(existing, key=lambda path: path.stat().st_mtime_ns) if existing else None


if MODE in {"benchmark", "offline"}:
    resume = None if MODE == "benchmark" else latest_offline_checkpoint()
    preflight = ["-m", "qchunk.preflight", "--config", RUNTIME_CONFIG, "--phase", "offline", "--require-gpu"]
    if resume is not None:
        preflight.extend(["--resume", resume])
    run_command(preflight)
    steps = BENCHMARK_STEPS if MODE == "benchmark" else OFFLINE_STEPS
    command = ["-m", "qchunk.train_ogbench", "--config", RUNTIME_CONFIG, "--steps", steps]
    if resume is not None:
        command.extend(["--resume", resume])
    elapsed = run_command(command)
    if MODE == "benchmark":
        projected_hours = elapsed / BENCHMARK_STEPS * OFFLINE_STEPS / 3600
        print(f"Ước lượng bảo thủ cho {OFFLINE_STEPS:,} update: {projected_hours:.2f} giờ")

elif MODE == "online":
    offline_checkpoint = RUN_DIR / "offline_final.msgpack"
    resume = RUN_DIR / "online_resume"
    if not offline_checkpoint.is_file():
        raise FileNotFoundError("Thiếu offline_final.msgpack; hãy chạy phase offline trước.")
    command = [
        "-m", "qchunk.train_online_ogbench",
        "--config", RUNTIME_CONFIG,
        "--checkpoint", offline_checkpoint,
        "--steps", ONLINE_STEPS,
        "--eval-episodes", EVALUATION_EPISODES,
    ]
    preflight = ["-m", "qchunk.preflight", "--config", RUNTIME_CONFIG, "--phase", "online", "--require-gpu"]
    if (resume / "metadata.msgpack").is_file():
        command.extend(["--resume", resume])
        preflight.extend(["--resume", resume])
    else:
        preflight.extend(["--checkpoint", offline_checkpoint])
    run_command(preflight)
    run_command(command)

elif MODE == "evaluation":
    online_checkpoint = RUN_DIR / "online_final.msgpack"
    checkpoint = online_checkpoint if online_checkpoint.is_file() else RUN_DIR / "offline_final.msgpack"
    if not checkpoint.is_file():
        raise FileNotFoundError("Không tìm thấy online hoặc offline checkpoint để evaluation.")
    run_command([
        "-m", "qchunk.preflight", "--config", RUNTIME_CONFIG,
        "--phase", "evaluation", "--checkpoint", checkpoint, "--require-gpu",
    ])
    run_command([
        "-m", "qchunk.evaluate_ogbench", "--config", RUNTIME_CONFIG,
        "--checkpoint", checkpoint, "--episodes", EVALUATION_EPISODES,
    ])

else:  # simulation
    online_checkpoint = RUN_DIR / "online_final.msgpack"
    checkpoint = online_checkpoint if online_checkpoint.is_file() else RUN_DIR / "offline_final.msgpack"
    if not checkpoint.is_file():
        raise FileNotFoundError("Không tìm thấy checkpoint để tạo simulation.")
    simulation_path = WORK_ROOT / "simulations" / f"{run_name}_seed_{SIMULATION_SEED}.gif"
    run_command([
        "-m", "qchunk.simulate_ogbench", "--config", RUNTIME_CONFIG,
        "--checkpoint", checkpoint, "--output", simulation_path,
        "--seed", SIMULATION_SEED,
    ])
    from IPython.display import Image, display
    display(Image(filename=str(simulation_path)))
"""
        ),
        _code(
            """# Tạo gói nhỏ để tải về: model cuối, config và metrics (không chứa replay resume lớn).
import zipfile

DELIVERABLE = WORK_ROOT / f"qchunk_{run_name}_{MODE}.zip"
with zipfile.ZipFile(DELIVERABLE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(RUNTIME_CONFIG, arcname=RUNTIME_CONFIG.name)
    for name in ("offline_final.msgpack", "online_final.msgpack", "metrics.jsonl", "metrics.csv", "manifest.json"):
        path = RUN_DIR / name
        if path.is_file():
            archive.write(path, arcname=f"{run_name}/{name}")
    simulation_dir = WORK_ROOT / "simulations"
    if simulation_dir.is_dir():
        for path in simulation_dir.glob("*.gif"):
            archive.write(path, arcname=f"simulations/{path.name}")

print("Gói tải về:", DELIVERABLE, f"({DELIVERABLE.stat().st_size / 2**20:.1f} MiB)")
print("Run directory:", RUN_DIR)
subprocess.run(["du", "-sh", str(RUN_DIR), str(DATASET_DIR)], check=False)
"""
        ),
        _markdown(
            """## Tiếp tục ở Kaggle Version kế tiếp

Sau khi phase kết thúc, chọn **Save Version** để Kaggle giữ `/kaggle/working`.
Trong notebook/version kế tiếp:

1. Thêm output của version trước bằng **Add Input**.
2. Tìm đường dẫn bằng `find /kaggle/input -name offline_final.msgpack -o -name metadata.msgpack`.
3. Điền thư mục cha phù hợp vào `PREVIOUS_RUN_DIR`.
4. Nếu output trước có `ogbench_data`, điền nó vào `PREVIOUS_DATASET_DIR`.
5. Đổi `MODE` sang phase tiếp theo và chạy lại từ đầu.

Để tải kết quả về máy, mở bảng **Output** và tải file `qchunk_*.zip`.
"""
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    print(build())
