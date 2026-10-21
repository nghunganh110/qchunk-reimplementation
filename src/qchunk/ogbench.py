import http.client
import json
import ssl
import urllib.error
import urllib.request
import warnings
import zipfile
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import ogbench

from qchunk.dataset import TransitionDataset

DEFAULT_DATASET_DIR = Path("~/.ogbench/data")
DATASET_URL = "https://rail.eecs.berkeley.edu/datasets/ogbench"
REQUIRED_ARCHIVE_MEMBERS = frozenset({"observations.npy", "actions.npy", "terminals.npy"})
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_REPORT_INTERVAL = 16 * 1024 * 1024
DOWNLOAD_MAX_ATTEMPTS = 256
DOWNLOAD_MAX_STALLED_ATTEMPTS = 3
DOWNLOAD_TIMEOUT_SECONDS = 30
RETRYABLE_DOWNLOAD_ERRORS = (
    urllib.error.URLError,
    TimeoutError,
    ConnectionError,
    http.client.HTTPException,
    ssl.SSLError,
)


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


def make_ogbench_env(dataset_name: str, **env_kwargs: Any):
    """Create an OGBench environment without downloading a dataset."""
    return _make_env_and_datasets(dataset_name, env_only=True, **env_kwargs)


def _make_env_and_datasets(dataset_name: str, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Box (low|high)'s precision lowered.*")
        return ogbench.make_env_and_datasets(dataset_name, **kwargs)


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


def validate_ogbench_dataset_archive(path: str | Path) -> None:
    """Validate an OGBench NPZ directory and the CRC of every compressed member."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"OGBench archive does not exist: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            members = set(archive.namelist())
            corrupt_member = archive.testzip()
    except (EOFError, zipfile.BadZipFile, zlib.error) as error:
        raise ValueError(
            f"OGBench archive is incomplete or corrupt: {path}. "
            "Move or remove it so OGBench can download it again."
        ) from error
    if corrupt_member is not None:
        raise ValueError(
            f"OGBench archive member failed its CRC check: {path}:{corrupt_member}. "
            "Move or remove the archive so OGBench can download it again."
        )
    missing = REQUIRED_ARCHIVE_MEMBERS - members
    if missing:
        raise ValueError(f"OGBench archive {path} is missing members: {sorted(missing)}")


def _response_total_size(response: Any, offset: int) -> int | None:
    content_range = response.headers.get("Content-Range")
    if content_range is not None and "/" in content_range:
        total = content_range.rsplit("/", 1)[-1]
        if total.isdigit():
            return int(total)
    content_length = response.headers.get("Content-Length")
    if content_length is not None and content_length.isdigit():
        return int(content_length) + (offset if response.status == 206 else 0)
    return None


def _validate_response_range(response: Any, offset: int) -> None:
    if offset == 0 or response.status != 206:
        return
    content_range = response.headers.get("Content-Range", "")
    prefix = content_range.split("/", 1)[0]
    fields = prefix.split()
    if len(fields) != 2 or fields[0].lower() != "bytes" or "-" not in fields[1]:
        raise OSError(f"Server returned an invalid Content-Range while resuming: {content_range!r}")
    start = fields[1].split("-", 1)[0]
    if not start.isdigit() or int(start) != offset:
        raise OSError(
            f"Server resumed at byte {start!r}, but the local archive requires byte {offset}"
        )


def _download_archive(url: str, path: Path) -> None:
    """Download one archive atomically, resuming an interrupted temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    expected_size = None
    stalled_attempts = 0

    for attempt in range(1, DOWNLOAD_MAX_ATTEMPTS + 1):
        initial_size = temporary_path.stat().st_size if temporary_path.exists() else 0
        offset = initial_size
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(url, headers=headers)
        network_error = None
        try:
            response_context = urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS)
        except urllib.error.HTTPError as error:
            if error.code == 416 and offset:
                validate_ogbench_dataset_archive(temporary_path)
                temporary_path.replace(path)
                return
            raise
        except RETRYABLE_DOWNLOAD_ERRORS as error:
            network_error = error
            response_context = None

        if response_context is not None:
            with response_context as response:
                status = getattr(response, "status", response.getcode())
                if offset and status != 206:
                    offset = 0
                else:
                    _validate_response_range(response, offset)
                mode = "ab" if offset else "wb"
                response_size = _response_total_size(response, offset)
                if response_size is not None:
                    expected_size = response_size
                downloaded = offset
                next_report = downloaded + DOWNLOAD_REPORT_INTERVAL
                action = "Resuming" if offset else "Downloading"
                print(f"{action} dataset: {url}")
                with temporary_path.open(mode) as file:
                    while True:
                        try:
                            chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                        except RETRYABLE_DOWNLOAD_ERRORS as error:
                            network_error = error
                            break
                        if not chunk:
                            break
                        file.write(chunk)
                        downloaded += len(chunk)
                        if downloaded >= next_report:
                            if expected_size is None:
                                print(f"  downloaded {downloaded / 1024**2:.1f} MiB")
                            else:
                                print(
                                    f"  downloaded {downloaded / 1024**2:.1f} / "
                                    f"{expected_size / 1024**2:.1f} MiB"
                                )
                            next_report += DOWNLOAD_REPORT_INTERVAL

        actual_size = temporary_path.stat().st_size if temporary_path.exists() else 0
        if expected_size is None and temporary_path.exists():
            try:
                validate_ogbench_dataset_archive(temporary_path)
            except (FileNotFoundError, ValueError):
                pass
            else:
                temporary_path.replace(path)
                return
        elif expected_size is not None and actual_size == expected_size:
            validate_ogbench_dataset_archive(temporary_path)
            temporary_path.replace(path)
            return
        elif expected_size is not None and actual_size > expected_size:
            raise OSError(
                f"Downloaded archive is larger than the server reported for {path}: "
                f"received {actual_size} of {expected_size} bytes"
            )

        if actual_size > initial_size:
            stalled_attempts = 0
        else:
            stalled_attempts += 1
        if attempt == DOWNLOAD_MAX_ATTEMPTS or stalled_attempts >= DOWNLOAD_MAX_STALLED_ATTEMPTS:
            break
        total = "unknown" if expected_size is None else str(expected_size)
        reason = (
            f"network error ({network_error})"
            if network_error is not None
            else "connection ended early"
        )
        print(
            f"  {reason} at {actual_size} / {total} bytes; automatically reconnecting "
            f"(attempt {attempt + 1}/{DOWNLOAD_MAX_ATTEMPTS})"
        )

    actual_size = temporary_path.stat().st_size if temporary_path.exists() else 0
    total = "unknown" if expected_size is None else str(expected_size)
    raise OSError(
        f"Incomplete download for {path}: received {actual_size} of {total} bytes after "
        f"{attempt} attempts. Run the command again to resume."
    )


def download_ogbench_dataset_archives(
    dataset_name: str,
    dataset_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Ensure both OGBench archives exist, using resumable atomic downloads."""
    paths = ogbench_dataset_paths(dataset_name, dataset_dir)
    for path in paths:
        if path.exists():
            validate_ogbench_dataset_archive(path)
            continue
        _download_archive(f"{DATASET_URL}/{path.name}", path)
    return paths


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
    # Flax msgpack may restore NumPy arrays backed by an immutable buffer.
    # MuJoCo's Python binding requires a writable, C-contiguous float64 array.
    physics_state = np.array(
        state["physics_state"],
        dtype=np.float64,
        order="C",
        copy=True,
    )
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
    archive_paths = download_ogbench_dataset_archives(dataset_name, dataset_dir)

    kwargs: dict[str, Any] = {"compact_dataset": False}
    if dataset_dir is not None:
        kwargs["dataset_dir"] = str(Path(dataset_dir).expanduser())

    try:
        env, train_data, validation_data = _make_env_and_datasets(dataset_name, **kwargs)
    except zipfile.BadZipFile as error:
        raise ValueError(
            "An OGBench dataset download produced an incomplete or corrupt NPZ archive. "
            f"Inspect these files and download them again: {archive_paths}"
        ) from error
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
