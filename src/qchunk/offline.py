from collections.abc import Callable
from dataclasses import dataclass

import jax
import numpy as np

from qchunk.dataset import TransitionDataset
from qchunk.training import QCTrainState, batch_to_jax

Metrics = dict[str, float]


@dataclass(frozen=True)
class OfflineTrainingConfig:
    steps: int
    batch_size: int
    log_interval: int = 1_000
    dataset_seed: int = 0

    def validate(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")


@dataclass(frozen=True)
class MetricRecord:
    step: int
    metrics: Metrics


@dataclass(frozen=True)
class OfflineTrainingResult:
    state: QCTrainState
    records: tuple[MetricRecord, ...]


def _host_metrics(metrics) -> Metrics:
    host_values = jax.device_get(metrics)
    return {name: float(value) for name, value in host_values.items()}


def train_offline(
    state: QCTrainState,
    dataset: TransitionDataset,
    config: OfflineTrainingConfig,
    on_log: Callable[[MetricRecord], None] | None = None,
) -> OfflineTrainingResult:
    """Train QC on a fixed transition dataset."""
    config.validate()
    if dataset.data["actions"].shape[-1] != state.config.action_dim:
        raise ValueError("dataset action dimension does not match QCConfig")
    if dataset.data["observations"].shape[-1] <= 0:
        raise ValueError("dataset observations must have a feature dimension")

    dataset_rng = np.random.default_rng(config.dataset_seed)
    records = []
    for step in range(1, config.steps + 1):
        chunk_batch = dataset.sample_chunks(
            config.batch_size,
            state.config.horizon,
            dataset_rng,
        )
        state, device_metrics = state.update(batch_to_jax(chunk_batch))

        should_log = step == 1 or step % config.log_interval == 0 or step == config.steps
        if should_log:
            record = MetricRecord(step=step, metrics=_host_metrics(device_metrics))
            records.append(record)
            if on_log is not None:
                on_log(record)

    return OfflineTrainingResult(state=state, records=tuple(records))
