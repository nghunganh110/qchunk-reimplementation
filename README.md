# Q-chunking reimplementation

An implementation of the core QC algorithm from *Reinforcement Learning with
Action Chunking*. The codebase currently includes episode-safe chunk sampling,
flow matching, Euler action generation, chunk critics, best-of-N selection,
offline training, checkpoints, and an OGBench adapter.

The small-scale comparison against the authors' released per-task results is in
[`results/reproduction/README.md`](results/reproduction/README.md). It combines
the existing cube-triple and puzzle-3x3 run logs with the published QC curves;
the saved puzzle checkpoint was also re-evaluated on 50 episodes without
retraining.

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

## Kaggle GPU notebook

[`notebooks/qchunk_kaggle.ipynb`](notebooks/qchunk_kaggle.ipynb) is a
self-contained notebook for Kaggle. It embeds the current package source, so it
does not require a public Git repository. In Kaggle, enable a GPU and Internet
before running it. The default experiment is the paper's sparse-reward
`puzzle-3x3-play-singletask-task4-v0` configuration, and the default mode is a
2,000-update benchmark measured after a one-update JAX warm-up.

Run long phases separately and save a Kaggle Version after each one. Attach the
previous Version's output to the next notebook and set `PREVIOUS_RUN_DIR` to
resume from its checkpoints. Kaggle keeps files under `/kaggle/working`, while
the notebook also produces a smaller `qchunk_*.zip` containing final model
checkpoints, metrics, configuration, and simulations.

The Puzzle 3×3 benchmark measured about 9.46 updates/second on one Kaggle T4,
which projects one million offline updates to roughly 29.4 hours. The notebook
therefore defaults to a cumulative offline target of 300,000 steps and an online
pilot target of 10,000 steps. Continue offline in separate saved Versions with
targets 600,000, 900,000, and 1,000,000; each resumed run restores optimizer and
dataset RNG state rather than restarting training.

After changing package source or experiment configs, regenerate the embedded
notebook with:

```bash
uv run python tools/build_kaggle_notebook.py
```

## Inference-only gradient refinement experiment

`src/qchunk/refinement.py` implements two projected Q-gradient steps around the
best-of-N action chunk, without retraining. The paired CPU evaluations on the
saved cube and puzzle checkpoints, commands, per-episode JSON, and limitations
are in [`results/refinement/README.md`](results/refinement/README.md). The
smaller trust radius shortened mean episodes relative to QC N=8 in these runs,
but did not improve success rate; a larger radius reduced cube success. A
follow-up projects the Q gradient onto directions formed by the three nearest
flow candidates. It retained QC N=8's 50-episode success counts on both tasks,
though its shorter mean episodes were not a reliable improvement in this sample.
On an earlier puzzle checkpoint (50,000 offline updates), a fresh set of 50
evaluation episodes gave 31/50 for QC N=8, 27/50 for small gradient refinement,
28/50 for geometric refinement, and 13/50 for QC N=64. The comparison changes
only inference and uses the same trained checkpoint for all four policies.

The follow-up Q trace in `src/qchunk/diagnose_q_selection.py` compares the two
critics' estimates for selected chunks with realized discounted returns. On
the 50,000-update puzzle checkpoint, N=64 reached 11/50 versus 30/50 for N=8
on a new evaluation seed, despite a higher predicted Q for its first chunk.
At 100,000 updates, N=64 reached 50/50 versus 47/50. A checkpoint-level rule
calibrated on those episodes selected N=8, then N=64; on separate evaluation
episodes it reached 49/60, versus 45/60 for fixed N=8 and 35/60 for fixed
N=64. Calibration itself required 200 environment episodes. The protocol,
per-episode data, and caveats are in the same results README.
