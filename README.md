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
hyperparameters, NPZ archive directories, writable disk space, and any checkpoint
required by the selected phase. This catches interrupted OGBench downloads before
NumPy attempts to load their large arrays. A CPU-only setup is reported as a
warning; add `--require-gpu` when it should make the check fail. Use
`--phase online` or `--phase evaluation`, optionally with `--checkpoint` or
`--resume`, before later pipeline stages. `--json` emits a machine-readable
report. The process exits with status 1 whenever a required check fails.

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

Downloads are first written as `.npz.tmp` files. A short response automatically
reconnects with an HTTP range request; repeating the same command also resumes
the remaining bytes if all automatic attempts are exhausted. Resume responses
must start at exactly the requested byte. A file receives its final `.npz` name
only after its reported byte count, ZIP directory, and every compressed member's
CRC are valid. If preflight reports an already-finalized archive as corrupt,
move that `.npz` aside and repeat the training command.

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

## Modal GPU run

The repository includes a cost-capped Modal launcher. Install the locked
development dependencies and authenticate once:

```bash
uv sync --dev
uv run modal setup
uv run modal token info
```

Start with an A10 benchmark. It downloads and validates OGBench data on CPU,
stores it in the `qchunk-experiments` Modal Volume, warms up JAX compilation,
then times 2,000 learner updates:

```bash
uv run modal run modal_qchunk.py \
  --mode benchmark \
  --gpu A10 \
  --benchmark-steps 2000
```

Inspect `projected_offline_hours` and `projected_offline_cost_usd` before the
full run. The calibration on 2026-10-20 measured 88.63 updates/second on an A10,
projecting 3.13 hours and $4.85 for one million offline updates. The following
command allows roughly 65% headroom while limiting the recorded GPU, 4-core CPU,
and 32 GiB memory list prices to $8:

```bash
uv run modal run --detach modal_qchunk.py \
  --mode offline \
  --gpu A10 \
  --budget-usd 30 \
  --max-spend-usd 8
```

The full job resumes the newest checkpoint in
`/vol/runs/cube_triple`, writes a checkpoint every 50,000 updates, and persists
data, metrics, and checkpoints in the same Volume. Modal's dashboard continues
to show logs after the local terminal disconnects. The embedded cost table is a
snapshot dated 2026-10-20; compare it with Modal's pricing page before future
runs. The projection covers offline learner updates only. Benchmark online
fine-tuning separately before spending the remaining balance.

Evaluate the final offline checkpoint directly from the Modal Volume. Start
with one episode to include JAX compilation and verify the environment:

```bash
uv run modal run modal_qchunk.py \
  --mode evaluation \
  --gpu A10 \
  --evaluation-episodes 1 \
  --budget-usd 30 \
  --max-spend-usd 0.25
```

Then run the configured 50 episodes with a one-dollar resource ceiling:

```bash
uv run modal run modal_qchunk.py \
  --mode evaluation \
  --gpu A10 \
  --evaluation-episodes 50 \
  --budget-usd 30 \
  --max-spend-usd 1
```

The final JSON contains `success_rate`, `mean_return`, and `mean_length`, and the
same metrics are appended to `/vol/runs/cube_triple/metrics.jsonl`.

Before the million-step online run, measure the combined environment, learner,
evaluation, and checkpoint cost with a 10,000-step pilot. This runs one final
evaluation episode and limits the Modal resource estimate to one dollar:

```bash
uv run modal run modal_qchunk.py \
  --mode online \
  --gpu A10 \
  --online-target-steps 10000 \
  --evaluation-episodes 1 \
  --budget-usd 30 \
  --max-spend-usd 1
```

The pilot writes a resumable checkpoint directory to
`/vol/runs/cube_triple/online_resume` and a model-only checkpoint to
`/vol/runs/cube_triple/online_final.msgpack`. Inspect
`projected_million_step_cost_usd_from_pilot` before continuing. The target is a
total environment-step count, so a later run toward one million automatically
resumes from step 10,000 instead of repeating it:

```bash
uv run modal run --detach modal_qchunk.py \
  --mode online \
  --gpu A10 \
  --online-target-steps 1000000 \
  --evaluation-episodes 50 \
  --budget-usd 30 \
  --max-spend-usd 8
```

Online training commits the Volume after every 50,000-step resume checkpoint
and at normal completion. Evaluations run every 100,000 environment steps and
once at the final target step.

Record one rendered episode from the final online checkpoint. This reuses the
model already stored in the Modal Volume and writes a small animated GIF, so the
large online resume checkpoint does not need to be downloaded:

```bash
uv run modal run modal_qchunk.py \
  --mode simulation \
  --gpu A10 \
  --simulation-seed 42 \
  --simulation-frame-stride 5 \
  --simulation-fps 10 \
  --budget-usd 30 \
  --max-spend-usd 0.25
```

The final JSON reports the episode `success`, `episode_return`, `length`, and
the GIF path. Download the recording from the Volume:

```bash
mkdir -p results/simulations
uv run modal volume get \
  qchunk-experiments \
  runs/cube_triple/simulations/online_seed_42.gif \
  results/simulations/online_seed_42.gif
```

Change `--simulation-seed` to record another episode. A single seed can fail
even when the 50-episode success rate is high.
