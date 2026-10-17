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

Evaluate the saved checkpoint over the paper's default 50 episodes:

```bash
uv run python -m qchunk.evaluate_ogbench \
  --config configs/cube_triple.yaml
```

At each chunk boundary, evaluation generates one best-of-N chunk and executes
its actions open-loop until all $H$ actions are used or the episode terminates.
