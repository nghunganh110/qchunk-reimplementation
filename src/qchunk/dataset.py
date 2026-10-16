from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChunkBatch:
    observations: np.ndarray  # [B, S]
    actions: np.ndarray  # [B, H, A]
    rewards: np.ndarray  # [B, H], raw rewards
    next_observations: np.ndarray  # [B, H, S]
    masks: np.ndarray  # [B, H], raw bootstrap masks
    valid: np.ndarray  # [B, H]


class TransitionDataset:
    REQUIRED_FIELDS = (
        "observations",
        "actions",
        "rewards",
        "next_observations",
        "masks",
        "terminals",
    )

    def __init__(self, data: dict[str, np.ndarray]):
        missing = set(self.REQUIRED_FIELDS) - data.keys()
        if missing:
            raise ValueError(f"Missing fields: {sorted(missing)}")

        self.data = {key: np.asarray(data[key], dtype=np.float32) for key in self.REQUIRED_FIELDS}

        observations = self.data["observations"]
        actions = self.data["actions"]

        if observations.ndim != 2:
            raise ValueError("observations must have shape [M, S]")
        if actions.ndim != 2:
            raise ValueError("actions must have shape [M, A]")

        self.size = observations.shape[0]
        if self.size == 0:
            raise ValueError("Dataset must not be empty")

        for key, values in self.data.items():
            if values.ndim == 0 or values.shape[0] != self.size:
                raise ValueError(f"{key} must have {self.size} rows")

        if self.data["next_observations"].shape != observations.shape:
            raise ValueError("next_observations must match observations shape")

        for key in ("rewards", "masks", "terminals"):
            if self.data[key].shape != (self.size,):
                raise ValueError(f"{key} must have shape [M]")

        for key in ("masks", "terminals"):
            values = self.data[key]
            if not np.all((values == 0) | (values == 1)):
                raise ValueError(f"{key} must contain only 0 or 1")

    def __len__(self) -> int:
        return self.size

    def get_chunks(self, starts: np.ndarray, horizon: int) -> ChunkBatch:
        """Gather chunks without crossing episode or dataset boundaries.

        The transition ending an episode is valid. Later actions, rewards,
        and masks are padded with zero. Padded next observations repeat the
        last valid transition's next observation.
        """
        if not isinstance(horizon, (int, np.integer)) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        starts = np.asarray(starts)
        if starts.ndim != 1 or starts.size == 0:
            raise ValueError("starts must be a nonempty one-dimensional array")
        if not np.issubdtype(starts.dtype, np.integer):
            raise ValueError("starts must contain integer indices")
        if np.any(starts < 0) or np.any(starts >= self.size):
            raise ValueError("starts must index transitions within the dataset")
        starts = starts.astype(np.intp)

        # Shape [B, H]: each row contains consecutive transition indices.
        indices = starts[:, None] + np.arange(horizon)[None, :]
        in_bounds = indices < self.size
        safe_indices = np.minimum(indices, self.size - 1)

        # A terminal invalidates subsequent positions, not its own position.
        terminals = self.data["terminals"][safe_indices]
        before_boundary = np.ones_like(terminals)
        before_boundary[:, 1:] = np.cumprod(1.0 - terminals[:, :-1], axis=1)
        valid = (before_boundary * in_bounds).astype(np.float32)
        is_valid = valid.astype(bool)

        last_indices = starts + valid.sum(axis=1).astype(np.intp) - 1
        last_next = self.data["next_observations"][last_indices]

        return ChunkBatch(
            observations=self.data["observations"][starts].copy(),
            actions=np.where(is_valid[..., None], self.data["actions"][safe_indices], 0.0),
            rewards=np.where(is_valid, self.data["rewards"][safe_indices], 0.0),
            next_observations=np.where(
                is_valid[..., None],
                self.data["next_observations"][safe_indices],
                last_next[:, None, :],
            ),
            masks=np.where(is_valid, self.data["masks"][safe_indices], 0.0),
            valid=valid,
        )

    def sample_chunks(self, batch_size: int, horizon: int, rng: np.random.Generator) -> ChunkBatch:
        """Uniformly sample start transitions using the caller's RNG."""
        if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        starts = rng.integers(self.size, size=batch_size)
        return self.get_chunks(starts, horizon)
