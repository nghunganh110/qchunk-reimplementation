import io
import urllib.error
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qchunk.config import QCConfig
from qchunk.ogbench import (
    _download_archive,
    capture_ogbench_env_state,
    convert_ogbench_dataset,
    load_ogbench,
    make_ogbench_env,
    ogbench_dataset_paths,
    restore_ogbench_env_state,
    validate_ogbench_dataset_archive,
)
from qchunk.train_ogbench import load_experiment_config, make_agent_config


def make_raw_dataset():
    return {
        "observations": np.arange(12).reshape(6, 2),
        "actions": np.array([[-2.0], [-0.5], [0.0], [0.5], [1.0], [2.0]], dtype=np.float32),
        "rewards": np.array([-1, -1, 0, -1, -1, -1]),
        "next_observations": np.arange(2, 14).reshape(6, 2),
        # Task completion and trajectory boundary are deliberately independent.
        "masks": np.array([1, 1, 0, 1, 1, 1]),
        "terminals": np.array([0, 0, 0, 1, 0, 1]),
    }


def test_conversion_preserves_ogbench_mask_and_terminal_semantics():
    dataset = convert_ogbench_dataset(make_raw_dataset(), action_clip_eps=0.1)
    np.testing.assert_array_equal(dataset.data["masks"], [1, 1, 0, 1, 1, 1])
    np.testing.assert_array_equal(dataset.data["terminals"], [0, 0, 0, 1, 0, 1])
    np.testing.assert_allclose(dataset.data["actions"][:, 0], [-0.9, -0.5, 0, 0.5, 0.9, 0.9])
    assert dataset.data["observations"].dtype == np.float32


def test_conversion_requires_regular_noncompact_dataset():
    raw = make_raw_dataset()
    del raw["next_observations"]
    with pytest.raises(ValueError, match="next_observations"):
        convert_ogbench_dataset(raw)


def test_cube_triple_config_matches_environment_dimensions():
    path = Path(__file__).parents[1] / "configs" / "cube_triple.yaml"
    experiment = load_experiment_config(path)
    agent = make_agent_config(experiment["agent"], action_dim=5)
    assert isinstance(agent, QCConfig)
    assert agent.horizon == 5
    assert agent.action_chunk_dim == 25
    assert agent.num_candidates == 32


def test_agent_config_rejects_wrong_explicit_action_dimension():
    with pytest.raises(ValueError, match="does not match"):
        make_agent_config({"horizon": 5, "action_dim": 4}, action_dim=5)


def test_dataset_paths_match_ogbench_single_task_archive_names(tmp_path):
    train_path, validation_path = ogbench_dataset_paths(
        "cube-triple-play-singletask-task2-v0", tmp_path
    )

    assert train_path == tmp_path / "cube-triple-play-v0.npz"
    assert validation_path == tmp_path / "cube-triple-play-v0-val.npz"


def test_dataset_archive_validation_checks_zip_directory_and_members(tmp_path):
    valid_path = tmp_path / "valid.npz"
    np.savez(
        valid_path,
        observations=np.zeros((2, 3)),
        actions=np.zeros((2, 1)),
        terminals=np.zeros(2),
    )
    validate_ogbench_dataset_archive(valid_path)

    truncated_path = tmp_path / "truncated.npz"
    truncated_path.write_bytes(valid_path.read_bytes()[:64])
    with pytest.raises(ValueError, match="incomplete or corrupt"):
        validate_ogbench_dataset_archive(truncated_path)


def test_dataset_archive_validation_checks_required_members(tmp_path):
    path = tmp_path / "missing.npz"
    np.savez(path, observations=np.zeros((2, 3)))

    with pytest.raises(ValueError, match="actions.npy"):
        validate_ogbench_dataset_archive(path)


def test_dataset_archive_validation_checks_compressed_member_crc(tmp_path):
    path = tmp_path / "corrupt.npz"
    np.savez_compressed(
        path,
        observations=np.zeros((100, 3)),
        actions=np.zeros((100, 1)),
        terminals=np.zeros(100),
    )
    content = bytearray(path.read_bytes())
    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo("observations.npy")
        with path.open("rb") as file:
            file.seek(member.header_offset + 26)
            filename_length = int.from_bytes(file.read(2), "little")
            extra_length = int.from_bytes(file.read(2), "little")
        data_offset = member.header_offset + 30 + filename_length + extra_length
    content[data_offset + member.compress_size // 2] ^= 0xFF
    path.write_bytes(content)

    with pytest.raises(ValueError, match="incomplete or corrupt|CRC"):
        validate_ogbench_dataset_archive(path)


def test_archive_download_resumes_temporary_file(monkeypatch, tmp_path):
    archive_buffer = io.BytesIO()
    np.savez(
        archive_buffer,
        observations=np.zeros((2, 3)),
        actions=np.zeros((2, 1)),
        terminals=np.zeros(2),
    )
    content = archive_buffer.getvalue()
    path = tmp_path / "dataset.npz"
    temporary_path = tmp_path / "dataset.npz.tmp"
    offset = len(content) // 2
    temporary_path.write_bytes(content[:offset])

    class FakeResponse(io.BytesIO):
        def __init__(self, value):
            super().__init__(value)
            self.status = 206
            self.headers = {
                "Content-Range": f"bytes {offset}-{len(content) - 1}/{len(content)}",
                "Content-Length": str(len(content) - offset),
            }

        def getcode(self):
            return self.status

    def fake_urlopen(request, timeout):
        assert timeout == 30
        assert request.get_header("Range") == f"bytes={offset}-"
        return FakeResponse(content[offset:])

    monkeypatch.setattr("qchunk.ogbench.urllib.request.urlopen", fake_urlopen)
    _download_archive("https://example.test/dataset.npz", path)

    assert path.read_bytes() == content
    assert not temporary_path.exists()


def test_archive_download_reconnects_after_early_eof(monkeypatch, tmp_path):
    archive_buffer = io.BytesIO()
    np.savez(
        archive_buffer,
        observations=np.zeros((2, 3)),
        actions=np.zeros((2, 1)),
        terminals=np.zeros(2),
    )
    content = archive_buffer.getvalue()
    split = len(content) // 3
    path = tmp_path / "dataset.npz"
    requests = []

    class FakeResponse(io.BytesIO):
        def __init__(self, value, status, headers):
            super().__init__(value)
            self.status = status
            self.headers = headers

        def getcode(self):
            return self.status

    def fake_urlopen(request, timeout):
        assert timeout == 30
        requests.append(request)
        if len(requests) == 1:
            assert request.get_header("Range") is None
            return FakeResponse(
                content[:split],
                200,
                {"Content-Length": str(len(content))},
            )
        assert request.get_header("Range") == f"bytes={split}-"
        return FakeResponse(
            content[split:],
            206,
            {
                "Content-Range": f"bytes {split}-{len(content) - 1}/{len(content)}",
                "Content-Length": str(len(content) - split),
            },
        )

    monkeypatch.setattr("qchunk.ogbench.urllib.request.urlopen", fake_urlopen)
    _download_archive("https://example.test/dataset.npz", path)

    assert path.read_bytes() == content
    assert len(requests) == 2


def test_archive_download_retries_network_error(monkeypatch, tmp_path):
    archive_buffer = io.BytesIO()
    np.savez(
        archive_buffer,
        observations=np.zeros((2, 3)),
        actions=np.zeros((2, 1)),
        terminals=np.zeros(2),
    )
    content = archive_buffer.getvalue()
    path = tmp_path / "dataset.npz"
    attempts = 0

    class FakeResponse(io.BytesIO):
        def __init__(self, value):
            super().__init__(value)
            self.status = 200
            self.headers = {"Content-Length": str(len(content))}

        def getcode(self):
            return self.status

    def fake_urlopen(request, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise urllib.error.URLError("temporary failure")
        return FakeResponse(content)

    monkeypatch.setattr("qchunk.ogbench.urllib.request.urlopen", fake_urlopen)
    _download_archive("https://example.test/dataset.npz", path)

    assert path.read_bytes() == content
    assert attempts == 2


def test_archive_download_rejects_misaligned_range(monkeypatch, tmp_path):
    path = tmp_path / "dataset.npz"
    temporary_path = tmp_path / "dataset.npz.tmp"
    temporary_path.write_bytes(b"partial")

    class FakeResponse(io.BytesIO):
        def __init__(self, value):
            super().__init__(value)
            self.status = 206
            self.headers = {
                "Content-Range": "bytes 0-9/10",
                "Content-Length": "10",
            }

        def getcode(self):
            return self.status

    monkeypatch.setattr(
        "qchunk.ogbench.urllib.request.urlopen",
        lambda request, timeout: FakeResponse(b"0123456789"),
    )

    with pytest.raises(OSError, match="local archive requires byte 7"):
        _download_archive("https://example.test/dataset.npz", path)
    assert temporary_path.read_bytes() == b"partial"


def test_loader_requests_regular_data_and_separate_eval_env(monkeypatch, tmp_path):
    class FakeEnv:
        observation_space = SimpleNamespace(shape=(2,))
        action_space = SimpleNamespace(shape=(1,))

        def close(self):
            pass

    train_env = FakeEnv()
    eval_env = FakeEnv()
    raw = make_raw_dataset()
    calls = []

    def fake_make(dataset_name, **kwargs):
        calls.append((dataset_name, kwargs))
        if kwargs.get("env_only"):
            return eval_env
        return train_env, raw, raw

    monkeypatch.setattr(
        "qchunk.ogbench.download_ogbench_dataset_archives",
        lambda dataset_name, dataset_dir: ogbench_dataset_paths(dataset_name, dataset_dir),
    )
    monkeypatch.setattr("qchunk.ogbench.ogbench.make_env_and_datasets", fake_make)
    bundle = load_ogbench("example-v0", dataset_dir=tmp_path)

    assert bundle.env is train_env
    assert bundle.eval_env is eval_env
    assert len(bundle.train_dataset) == 6
    assert calls == [
        (
            "example-v0",
            {"compact_dataset": False, "dataset_dir": str(tmp_path)},
        ),
        ("example-v0", {"env_only": True}),
    ]


@pytest.mark.filterwarnings("ignore:.*Wayland.*")
@pytest.mark.filterwarnings("ignore:.*Box low's precision lowered.*")
@pytest.mark.filterwarnings("ignore:.*Box high's precision lowered.*")
def test_cube_environment_state_round_trip():
    env = make_ogbench_env("cube-triple-play-singletask-task2-v0")
    try:
        env.reset(seed=123)
        env.step(np.zeros(5, dtype=np.float32))
        snapshot = capture_ogbench_env_state(env)
        actions = [
            np.array([0.1, -0.1, 0.05, 0.2, -0.2], dtype=np.float32),
            np.array([-0.05, 0.1, 0.0, -0.1, 0.2], dtype=np.float32),
        ]
        expected = [env.step(action) for action in actions]

        restore_ogbench_env_state(env, snapshot)
        actual = [env.step(action) for action in actions]

        for actual_step, expected_step in zip(actual, expected, strict=True):
            np.testing.assert_allclose(actual_step[0], expected_step[0], rtol=1e-6, atol=1e-6)
            assert actual_step[1:4] == expected_step[1:4]
            assert actual_step[4]["success"] == expected_step[4]["success"]
    finally:
        env.close()
