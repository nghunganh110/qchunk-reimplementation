from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from qchunk.simulation import record_episode


class FakeState:
    config = SimpleNamespace(horizon=2, action_dim=1)

    def select_action_chunks(self, observations, rng):
        del observations, rng
        return jnp.array([[0.1, 0.2]], dtype=jnp.float32), jnp.array([0])


class FakeEnv:
    def __init__(self):
        self.step_count = 0

    def reset(self, seed):
        assert seed == 7
        self.step_count = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        assert action.shape == (1,)
        self.step_count += 1
        done = self.step_count == 3
        return (
            np.array([self.step_count], dtype=np.float32),
            1.0,
            done,
            False,
            {"success": float(done)},
        )

    def render(self):
        return np.full((4, 5, 3), self.step_count, dtype=np.uint8)


def test_record_episode_saves_strided_frames(tmp_path, monkeypatch):
    saved = {}

    def fake_mimsave(path, frames, **kwargs):
        saved.update(path=path, frames=frames, kwargs=kwargs)

    monkeypatch.setattr("qchunk.simulation.imageio.mimsave", fake_mimsave)
    result = record_episode(
        FakeState(),
        FakeEnv(),
        tmp_path / "episode.gif",
        seed=7,
        frame_stride=2,
        fps=8,
    )

    assert result.episode_return == 3.0
    assert result.length == 3
    assert result.success == 1.0
    assert result.frames == 3
    assert saved["path"] == tmp_path / "episode.gif"
    assert [int(frame[0, 0, 0]) for frame in saved["frames"]] == [0, 2, 3]
    assert saved["kwargs"] == {"format": "GIF", "duration": 125.0, "loop": 0}


@pytest.mark.parametrize("field,value", [("frame_stride", 0), ("fps", 0)])
def test_record_episode_rejects_nonpositive_video_options(tmp_path, field, value):
    kwargs = {"frame_stride": 1, "fps": 1, field: value}
    with pytest.raises(ValueError, match=f"{field} must be positive"):
        record_episode(FakeState(), FakeEnv(), tmp_path / "episode.gif", **kwargs)
