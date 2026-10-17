import jax
import numpy as np
import pytest
from flax import serialization

from qchunk.checkpoint import (
    load_checkpoint,
    load_training_checkpoint,
    save_checkpoint,
    save_training_checkpoint,
)
from qchunk.config import QCConfig
from qchunk.dataset import TransitionDataset
from qchunk.offline import OfflineTrainingConfig, train_offline
from qchunk.training import QCTrainState, batch_to_jax


def make_dataset(size=20, observation_dim=3, action_dim=2):
    rng = np.random.default_rng(40)
    terminals = np.zeros(size)
    terminals[9] = 1
    terminals[-1] = 1
    masks = 1.0 - terminals
    return TransitionDataset(
        {
            "observations": rng.normal(size=(size, observation_dim)),
            "actions": np.tanh(rng.normal(size=(size, action_dim))),
            "rewards": rng.normal(size=size),
            "next_observations": rng.normal(size=(size, observation_dim)),
            "masks": masks,
            "terminals": terminals,
        }
    )


def make_state(seed=0, **overrides):
    values = {
        "horizon": 2,
        "action_dim": 2,
        "hidden_dims": (8,),
        "num_candidates": 2,
        "num_flow_steps": 2,
    }
    values.update(overrides)
    return QCTrainState.create(seed, 3, QCConfig(**values))


def state_allclose(left, right):
    left = serialization.to_state_dict(left)
    right = serialization.to_state_dict(right)
    comparisons = jax.tree.map(lambda x, y: np.allclose(np.asarray(x), np.asarray(y)), left, right)
    return all(jax.tree.leaves(comparisons))


def test_offline_loop_updates_state_and_logs_requested_steps():
    callbacks = []
    result = train_offline(
        make_state(),
        make_dataset(),
        OfflineTrainingConfig(steps=3, batch_size=4, log_interval=2, dataset_seed=5),
        on_log=callbacks.append,
    )

    assert result.state.flow.step == 3
    assert result.state.critic.step == 3
    assert [record.step for record in result.records] == [1, 2, 3]
    assert list(result.records) == callbacks
    assert all(np.isfinite(value) for record in result.records for value in record.metrics.values())


def test_checkpoint_restores_exact_state_and_can_resume(tmp_path):
    state = make_state(seed=1)
    batch = batch_to_jax(make_dataset().get_chunks(np.array([0, 2, 4, 6]), horizon=2))
    state, _ = state.update(batch)

    checkpoint_path = save_checkpoint(tmp_path / "step_1.msgpack", state)
    template = make_state(seed=999)
    restored = load_checkpoint(checkpoint_path, template)

    assert restored.flow.step == state.flow.step
    assert restored.critic.step == state.critic.step
    assert state_allclose(restored, state)

    expected_next, _ = state.update(batch)
    restored_next, _ = restored.update(batch)
    assert state_allclose(restored_next, expected_next)


def test_interrupted_training_resumes_with_identical_state(tmp_path):
    dataset = make_dataset()
    uninterrupted = train_offline(
        make_state(seed=7),
        dataset,
        OfflineTrainingConfig(steps=6, batch_size=4, log_interval=10, dataset_seed=11),
    )
    interrupted = train_offline(
        make_state(seed=7),
        dataset,
        OfflineTrainingConfig(steps=3, batch_size=4, log_interval=10, dataset_seed=11),
    )
    path = save_training_checkpoint(
        tmp_path / "offline_step_3.msgpack",
        interrupted.state,
        interrupted.step,
        interrupted.dataset_rng_state,
    )
    restored = load_training_checkpoint(path, make_state(seed=999))
    resumed = train_offline(
        restored.state,
        dataset,
        OfflineTrainingConfig(steps=6, batch_size=4, log_interval=10, dataset_seed=999),
        initial_step=restored.step,
        dataset_rng_state=restored.dataset_rng_state,
    )

    assert resumed.step == 6
    assert state_allclose(resumed.state, uninterrupted.state)
    assert resumed.dataset_rng_state == uninterrupted.dataset_rng_state


def test_periodic_checkpoint_callback_receives_completed_steps():
    checkpoints = []
    result = train_offline(
        make_state(),
        make_dataset(),
        OfflineTrainingConfig(
            steps=5,
            batch_size=4,
            log_interval=10,
            checkpoint_interval=2,
        ),
        on_checkpoint=checkpoints.append,
    )

    assert [checkpoint.step for checkpoint in checkpoints] == [2, 4]
    assert [int(checkpoint.state.flow.step) for checkpoint in checkpoints] == [2, 4]
    assert result.step == 5


def test_plain_model_checkpoint_has_no_resume_progress(tmp_path):
    path = save_checkpoint(tmp_path / "state.msgpack", make_state())
    with pytest.raises(TypeError, match="offline-training progress"):
        load_training_checkpoint(path, make_state())


def test_training_checkpoint_rejects_step_mismatched_with_optimizer(tmp_path):
    state = make_state()
    dataset_rng_state = np.random.default_rng(0).bit_generator.state
    with pytest.raises(ValueError, match="optimizer step"):
        save_training_checkpoint(tmp_path / "state.msgpack", state, 1, dataset_rng_state)


def test_checkpoint_rejects_mismatched_config(tmp_path):
    path = save_checkpoint(tmp_path / "state.msgpack", make_state())
    mismatched_template = make_state(horizon=3)
    with pytest.raises(ValueError, match="configuration"):
        load_checkpoint(path, mismatched_template)


@pytest.mark.parametrize(
    "config",
    [
        OfflineTrainingConfig(steps=0, batch_size=4),
        OfflineTrainingConfig(steps=1, batch_size=0),
        OfflineTrainingConfig(steps=1, batch_size=4, log_interval=0),
        OfflineTrainingConfig(steps=1, batch_size=4, checkpoint_interval=0),
    ],
)
def test_invalid_offline_config_is_rejected(config):
    with pytest.raises(ValueError):
        train_offline(make_state(), make_dataset(), config)


@pytest.mark.parametrize("initial_step", [-1, 4])
def test_invalid_initial_step_is_rejected(initial_step):
    with pytest.raises(ValueError, match="initial_step"):
        train_offline(
            make_state(),
            make_dataset(),
            OfflineTrainingConfig(steps=3, batch_size=4),
            initial_step=initial_step,
        )
