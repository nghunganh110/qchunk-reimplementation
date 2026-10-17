from collections.abc import Mapping

import numpy as np

from qchunk.dataset import ChunkBatch, TransitionDataset


class ReplayBuffer:
    """Fixed-capacity circular buffer with episode-safe chunk sampling."""

    REQUIRED_FIELDS = TransitionDataset.REQUIRED_FIELDS

    def __init__(self, capacity: int, observation_dim: int, action_dim: int):
        for name, value in {
            "capacity": capacity,
            "observation_dim": observation_dim,
            "action_dim": action_dim,
        }.items():
            if not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.size = 0
        self.pointer = 0
        self._data = {
            "observations": np.zeros((self.capacity, self.observation_dim), dtype=np.float32),
            "actions": np.zeros((self.capacity, self.action_dim), dtype=np.float32),
            "rewards": np.zeros(self.capacity, dtype=np.float32),
            "next_observations": np.zeros((self.capacity, self.observation_dim), dtype=np.float32),
            "masks": np.zeros(self.capacity, dtype=np.float32),
            "terminals": np.zeros(self.capacity, dtype=np.float32),
        }

    @classmethod
    def from_dataset(cls, dataset: TransitionDataset, capacity: int) -> "ReplayBuffer":
        """Create a replay buffer initially containing an offline dataset."""
        if capacity < len(dataset):
            raise ValueError("capacity must hold the complete initial dataset")
        buffer = cls(
            capacity=capacity,
            observation_dim=dataset.data["observations"].shape[-1],
            action_dim=dataset.data["actions"].shape[-1],
        )
        for key in TransitionDataset.REQUIRED_FIELDS:
            buffer._data[key][: len(dataset)] = dataset.data[key]
        # The first online transition starts a new collection stream. Mark the
        # end of the seed dataset as a trajectory boundary while preserving its
        # Bellman mask (a boundary may be a time-limit truncation).
        buffer._data["terminals"][len(dataset) - 1] = 1.0
        buffer.size = len(dataset)
        buffer.pointer = buffer.size % buffer.capacity
        return buffer

    def __len__(self) -> int:
        return self.size

    def clear(self) -> None:
        """Logically remove all transitions without reallocating storage."""
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition: Mapping[str, object]) -> None:
        """Append one transition, overwriting the oldest when full."""
        missing = set(TransitionDataset.REQUIRED_FIELDS) - transition.keys()
        if missing:
            raise ValueError(f"Missing fields: {sorted(missing)}")

        observation = self._vector(transition["observations"], self.observation_dim, "observations")
        action = self._vector(transition["actions"], self.action_dim, "actions")
        next_observation = self._vector(
            transition["next_observations"],
            self.observation_dim,
            "next_observations",
        )
        reward = self._scalar(transition["rewards"], "rewards")
        mask = self._binary_scalar(transition["masks"], "masks")
        terminal = self._binary_scalar(transition["terminals"], "terminals")

        index = self.pointer
        self._data["observations"][index] = observation
        self._data["actions"][index] = action
        self._data["rewards"][index] = reward
        self._data["next_observations"][index] = next_observation
        self._data["masks"][index] = mask
        self._data["terminals"][index] = terminal
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_chunks(self, starts: np.ndarray, horizon: int) -> ChunkBatch:
        """Gather chronological chunks without crossing episode boundaries."""
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        if not isinstance(horizon, (int, np.integer)) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        starts = np.asarray(starts)
        if starts.ndim != 1 or starts.size == 0:
            raise ValueError("starts must be a nonempty one-dimensional array")
        if not np.issubdtype(starts.dtype, np.integer):
            raise ValueError("starts must contain integer indices")
        if np.any(starts < 0) or np.any(starts >= self.size):
            raise ValueError("starts must index transitions within the replay buffer")
        starts = starts.astype(np.intp)

        logical_indices = starts[:, None] + np.arange(horizon)[None, :]
        in_bounds = logical_indices < self.size
        safe_logical_indices = np.minimum(logical_indices, self.size - 1)
        physical_indices = self._logical_to_physical(safe_logical_indices)

        terminals = self._data["terminals"][physical_indices]
        before_boundary = np.ones_like(terminals)
        before_boundary[:, 1:] = np.cumprod(1.0 - terminals[:, :-1], axis=1)
        valid = (before_boundary * in_bounds).astype(np.float32)
        is_valid = valid.astype(bool)

        last_logical_indices = starts + valid.sum(axis=1).astype(np.intp) - 1
        last_physical_indices = self._logical_to_physical(last_logical_indices)
        last_next = self._data["next_observations"][last_physical_indices]
        start_physical_indices = self._logical_to_physical(starts)

        return ChunkBatch(
            observations=self._data["observations"][start_physical_indices].copy(),
            actions=np.where(is_valid[..., None], self._data["actions"][physical_indices], 0.0),
            rewards=np.where(is_valid, self._data["rewards"][physical_indices], 0.0),
            next_observations=np.where(
                is_valid[..., None],
                self._data["next_observations"][physical_indices],
                last_next[:, None, :],
            ),
            masks=np.where(is_valid, self._data["masks"][physical_indices], 0.0),
            valid=valid,
        )

    def sample_chunks(
        self,
        batch_size: int,
        horizon: int,
        rng: np.random.Generator,
    ) -> ChunkBatch:
        """Uniformly sample logical start transitions using the caller's RNG."""
        if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        starts = rng.integers(self.size, size=batch_size)
        return self.get_chunks(starts, horizon)

    def as_dataset(self) -> TransitionDataset:
        """Copy the current contents into chronological dataset order."""
        if self.size == 0:
            raise ValueError("Cannot convert an empty replay buffer")
        logical_indices = np.arange(self.size)
        physical_indices = self._logical_to_physical(logical_indices)
        return TransitionDataset(
            {key: values[physical_indices].copy() for key, values in self._data.items()}
        )

    def state_dict(self) -> dict[str, object]:
        """Return a chronological snapshot suitable for checkpointing."""
        dataset = self.as_dataset()
        return {
            "capacity": self.capacity,
            "data": {key: values.copy() for key, values in dataset.data.items()},
        }

    def storage_state(self) -> dict[str, object]:
        """Expose physical storage for streaming checkpoint writes without copies."""
        return {
            "capacity": self.capacity,
            "size": self.size,
            "pointer": self.pointer,
            "data": self._data,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "ReplayBuffer":
        """Restore a buffer snapshot produced by :meth:`state_dict`."""
        if "capacity" not in state or "data" not in state:
            raise ValueError("replay buffer state requires capacity and data")
        data = state["data"]
        if not isinstance(data, Mapping):
            raise TypeError("replay buffer data must be a mapping")
        dataset = TransitionDataset(dict(data))
        capacity = int(state["capacity"])
        if capacity < len(dataset):
            raise ValueError("replay buffer capacity is smaller than its data")

        buffer = cls(
            capacity=capacity,
            observation_dim=dataset.data["observations"].shape[-1],
            action_dim=dataset.data["actions"].shape[-1],
        )
        for key in TransitionDataset.REQUIRED_FIELDS:
            buffer._data[key][: len(dataset)] = dataset.data[key]
        buffer.size = len(dataset)
        buffer.pointer = buffer.size % buffer.capacity
        return buffer

    @classmethod
    def from_storage_state(cls, state: Mapping[str, object]) -> "ReplayBuffer":
        """Restore the physical storage representation used by large checkpoints."""
        required = {"capacity", "size", "pointer", "data"}
        missing = required - state.keys()
        if missing:
            raise ValueError(f"replay storage state is missing fields: {sorted(missing)}")
        data = state["data"]
        if not isinstance(data, Mapping):
            raise TypeError("replay storage data must be a mapping")
        missing_data = set(TransitionDataset.REQUIRED_FIELDS) - data.keys()
        if missing_data:
            raise ValueError(f"replay storage data is missing fields: {sorted(missing_data)}")

        observations = np.asarray(data["observations"])
        actions = np.asarray(data["actions"])
        if observations.ndim != 2 or actions.ndim != 2:
            raise ValueError("replay storage observations and actions must be matrices")
        capacity = int(state["capacity"])
        size = int(state["size"])
        pointer = int(state["pointer"])
        if observations.shape[0] != capacity or actions.shape[0] != capacity:
            raise ValueError("replay storage arrays must match capacity")
        if not 0 <= size <= capacity or not 0 <= pointer < capacity:
            raise ValueError("replay storage size or pointer is invalid")

        buffer = cls(capacity, observations.shape[1], actions.shape[1])
        expected_shapes = {
            "observations": (capacity, buffer.observation_dim),
            "actions": (capacity, buffer.action_dim),
            "rewards": (capacity,),
            "next_observations": (capacity, buffer.observation_dim),
            "masks": (capacity,),
            "terminals": (capacity,),
        }
        for key, expected_shape in expected_shapes.items():
            values = np.asarray(data[key], dtype=np.float32)
            if values.shape != expected_shape:
                raise ValueError(f"replay storage field {key} has an invalid shape")
            buffer._data[key][...] = values
        buffer.size = size
        buffer.pointer = pointer
        return buffer

    def _logical_to_physical(self, indices: np.ndarray) -> np.ndarray:
        oldest = (self.pointer - self.size) % self.capacity
        return (oldest + indices) % self.capacity

    @staticmethod
    def _vector(value: object, dimension: int, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (dimension,):
            raise ValueError(f"{name} must have shape [{dimension}]")
        return array

    @staticmethod
    def _scalar(value: object, name: str) -> np.float32:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != ():
            raise ValueError(f"{name} must be a scalar")
        return np.float32(array)

    @classmethod
    def _binary_scalar(cls, value: object, name: str) -> np.float32:
        scalar = cls._scalar(value, name)
        if scalar not in (0.0, 1.0):
            raise ValueError(f"{name} must be 0 or 1")
        return scalar
