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

Evaluate the saved checkpoint over the paper's default 50 episodes:

```bash
uv run python -m qchunk.evaluate_ogbench \
  --config configs/cube_triple.yaml
```

At each chunk boundary, evaluation generates one best-of-N chunk and executes
its actions open-loop until all $H$ actions are used or the episode terminates.
