# Q-chunking reimplementation

An implementation of the core QC algorithm from *Reinforcement Learning with
Action Chunking*. The codebase currently includes episode-safe chunk sampling,
flow matching, Euler action generation, chunk critics, best-of-N selection,
offline training, checkpoints, and an OGBench adapter.

## Set up

```bash
uv sync --python 3.12
uv run pytest -q
```

The benchmark environment versions match the released QC repository:
OGBench 1.1.2, Gymnasium 1.1.1, and MuJoCo 3.3.1.

Before starting a long run, inspect the local setup without downloading data or
changing a checkpoint:

```bash
uv run python -m qchunk.preflight \
  --config configs/cube_triple.yaml \
  --phase offline
```

The command validates package versions, JAX devices, environment creation,
hyperparameters, local dataset archives, writable disk space, and any checkpoint
required by the selected phase. A CPU-only setup is reported as a warning; add
`--require-gpu` when it should make the check fail. Use `--phase online` or
`--phase evaluation`, optionally with `--checkpoint` or `--resume`, before later
pipeline stages. `--json` emits a machine-readable report. The process exits
with status 1 whenever a required check fails.

## Offline OGBench training

The first configuration targets
`cube-triple-play-singletask-task2-v0` with action horizon 5 and best-of-32:

```bash
uv run python -m qchunk.train_ogbench \
  --config configs/cube_triple.yaml
```

OGBench downloads the training and validation datasets on first use. The
cube-triple training archive is about 1 GB, so ensure the configured dataset
directory has sufficient disk space. For a pipeline smoke run after the dataset
is available:

```bash
uv run python -m qchunk.train_ogbench \
  --config configs/cube_triple.yaml \
  --steps 1
```

The command prints JSON metrics and writes the final checkpoint path configured
under `output.checkpoint`. See `PAPER_SPEC.md` for the equation-to-code mapping
and implementation decisions.

Training also writes periodic checkpoints according to
`offline.checkpoint_interval`. Resume toward the total configured step count
with:

```bash
uv run python -m qchunk.train_ogbench \
  --config configs/cube_triple.yaml \
  --resume runs/cube_triple/checkpoints/offline_step_0050000.msgpack
```

Each training checkpoint includes the model, optimizer, target critic, JAX RNG,
completed step, and dataset-sampling RNG. Resumed training therefore preserves
the same sequence of minibatches as an uninterrupted run.

The online replay buffer can be initialized from the complete offline dataset.
It maintains chronological, episode-safe chunk sampling after circular storage
wraparound; online fine-tuning uses this buffer in the next pipeline stage.

After offline pretraining, fine-tune the checkpoint online with:

```bash
uv run python -m qchunk.train_online_ogbench \
  --config configs/cube_triple.yaml
```

Evaluate the resulting online checkpoint with:

```bash
uv run python -m qchunk.evaluate_ogbench \
  --config configs/cube_triple.yaml \
  --checkpoint runs/cube_triple/online_final.msgpack
```

At each action-chunk boundary the policy generates a best-of-N chunk and executes
it open-loop. Every environment transition is appended to the replay buffer;
after `online.start_training`, QC updates sample uniformly from the buffer seeded
with the complete offline dataset. A short end-to-end run that includes learner
updates and one evaluation episode is:

```bash
uv run python -m qchunk.train_online_ogbench \
  --config configs/cube_triple.yaml \
  --steps 10 \
  --start-training 1 \
  --eval-episodes 1
```

Online training overwrites one resumable checkpoint at
`output.online_resume_checkpoint` according to `online.checkpoint_interval`.
Resume toward the configured total environment-step count with:

```bash
uv run python -m qchunk.train_online_ogbench \
  --config configs/cube_triple.yaml \
  --resume runs/cube_triple/online_resume
```

This checkpoint includes the learner, complete replay buffer, policy and replay
RNGs, current observation, remaining queued actions, partial episode metrics,
MuJoCo physics state, task targets, environment RNG, and time-limit counter. It
can therefore continue in the middle of an episode and action chunk. Since the
complete replay buffer is stored, the checkpoint directory may grow to hundreds
of MB. Replay arrays are streamed directly to separate `.npy` files to avoid a
large in-memory serialization copy, and the same directory is atomically
replaced instead of accumulating step snapshots.

Each run directory also contains:

- `manifest.json`: effective configuration for every invocation, command-line
  arguments, Git commit and dirty state, Python/platform details, JAX devices,
  and dependency versions.
- `metrics.jsonl`: one structured record per offline log, online log, completed
  online episode, and evaluation.
- `metrics.csv`: the same data in long form with one metric per row.

On resume, metric rows after the checkpoint step are removed for the resumed
phase before new values are appended. This prevents duplicate or stale branches
in learning curves after an interrupted run.

Evaluate the saved checkpoint over the paper's default 50 episodes:

```bash
uv run python -m qchunk.evaluate_ogbench \
  --config configs/cube_triple.yaml
```

At each chunk boundary, evaluation generates one best-of-N chunk and executes
its actions open-loop until all $H$ actions are used or the episode terminates.
