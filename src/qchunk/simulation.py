from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np

from qchunk.training import QCTrainState


@dataclass(frozen=True)
class SimulationResult:
    output_path: Path
    episode_return: float
    length: int
    success: float
    frames: int

    def as_metrics(self) -> dict[str, float | int | str]:
        return {
            "output_path": str(self.output_path),
            "episode_return": self.episode_return,
            "length": self.length,
            "success": self.success,
            "frames": self.frames,
        }


def record_episode(
    state: QCTrainState,
    env: Any,
    output_path: str | Path,
    *,
    seed: int = 42,
    frame_stride: int = 5,
    fps: int = 10,
) -> SimulationResult:
    """Roll out one action-chunk policy episode and save an animated GIF."""
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")

    observation, _ = env.reset(seed=seed)
    frames = [_render_frame(env)]
    rng = jax.random.PRNGKey(seed)
    episode_return = 0.0
    episode_length = 0
    success = 0.0
    done = False

    while not done:
        rng, action_rng = jax.random.split(rng)
        observations = jnp.asarray(observation, dtype=jnp.float32)[None, :]
        flat_chunk, _ = state.select_action_chunks(observations, action_rng)
        action_chunk = np.asarray(jax.device_get(flat_chunk[0])).reshape(
            state.config.horizon, state.config.action_dim
        )

        for action in action_chunk:
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            episode_length += 1
            success = max(success, float(info.get("success", 0.0)))
            done = bool(terminated or truncated)
            if episode_length % frame_stride == 0 or done:
                frames.append(_render_frame(env))
            if done:
                break

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, format="GIF", duration=1000 / fps, loop=0)
    return SimulationResult(
        output_path=output_path,
        episode_return=episode_return,
        length=episode_length,
        success=success,
        frames=len(frames),
    )


def _render_frame(env: Any) -> np.ndarray:
    frame = np.asarray(env.render(), dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise ValueError(f"environment returned an invalid RGB frame shape: {frame.shape}")
    return frame.copy()
