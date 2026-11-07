# Research results and reproducibility notes

## Scope and attribution

This repository independently implements Q-chunking (QC) from Q. Li, Z. Zhou,
and S. Levine, [*Reinforcement Learning with Action Chunking*](https://proceedings.neurips.cc/paper_files/paper/2025/hash/50348e8f9aef984abe0ea1ec2a326f78-Abstract-Conference.html),
NeurIPS 2025. The authors' [repository](https://github.com/ColinQiyangLi/qc)
provides the released per-task curves used for comparison. This project is a
reimplementation and follow-up analysis, not an official release or a new
training algorithm.

## Reproduction of two OGBench tasks

The implementation used the paper's tasks and key settings: action horizon
`H=5`, batch size 256, two critics, 10 flow-integration steps, four hidden
layers of width 512, and one million offline updates. Best-of-32 was used for
cube-triple task2 and best-of-64 for puzzle-3x3 task4. The puzzle task used
sparse rewards. Cube training continued for one million online environment
steps; puzzle had only a 10,000-step online pilot.

| Task, evaluation point | Paper QC, mean of 4 training seeds | This run, 1 training seed and 50 episodes |
| --- | ---: | ---: |
| Cube-triple task2, after 1M offline updates | 0% | 0/50 (0%) |
| Cube-triple task2, after 1M online steps | 89% | 48/50 (96%) |
| Puzzle-3x3 task4, after 1M offline updates | 100% | 50/50 (100%) |
| Puzzle-3x3 task4, after 10k online steps | No matched endpoint | 50/50 (100%) |

The cube learning curve also had a close intermediate point: at 300,000 online
steps the released QC curve was 36.5%, and this run was 36%. These observations
support reproduction of the main trend and endpoints on these two tasks. They
do not establish statistical equivalence across training seeds, nor superiority
over the paper. Fifty evaluation episodes estimate performance of one trained
checkpoint; they are not 50 independent training runs.

The detailed [comparison note](../results/reproduction/README.md) identifies
the exact released plot-data artifact and hash, run logs, configuration
manifests, plot-building command, and CPU checkpoint re-evaluations. The
[comparison figure](../results/reproduction/paper_vs_reimplementation.png)
is built from committed data rather than manually redrawn.

## Inference-only follow-up

The extension in `src/qchunk/refinement.py` takes the best chunk from QC's
candidate set and performs two projected steps along the mean critics' Q
gradient. A geometric variant projects that gradient onto directions between
nearby flow candidates. Neither variant retrains the policy or critic.

On 50 paired evaluation episodes using the same trained checkpoint, the small
gradient radius kept cube success at 47/50 versus QC N=8's 47/50 and changed
puzzle success from 50/50 to 49/50. Mean episode lengths fell, but paired
bootstrap intervals for the length differences included zero. The geometric
variant kept N=8's 47/50 cube and 50/50 puzzle success counts, again without a
reliable length improvement. A larger gradient radius reduced cube success
from 47/50 to 44/50. These were exploratory choices made after inspecting
earlier runs.

For a less saturated puzzle checkpoint at 50,000 offline updates, 20 screening
episodes selected the checkpoint and a fresh 50-episode set compared policies:

| Policy at the same 50k-update checkpoint | Successes / 50 |
| --- | ---: |
| QC, N=8 | 31 |
| N=8 plus small Q-gradient refinement | 27 |
| N=8 plus geometric refinement | 28 |
| QC, N=64 | 13 |

The poor N=64 result is specific to this early checkpoint. In a separate Q
diagnostic, N=64 succeeded on 11/50 versus N=8's 30/50 at 50,000 updates,
but succeeded on 50/50 versus N=8's 47/50 at 100,000 updates. Higher predicted
Q did not always correspond to better realized returns at the early checkpoint.
The diagnostic is consistent with optimistic selection among many candidates,
but it does not isolate a causal mechanism or prove a generally better choice
of N.

A rule calibrated on 200 environment episodes selected N=8 at 50,000 updates
and N=64 at 100,000. On separate evaluation episodes it reached 49/60 versus
45/60 for fixed N=8 and 35/60 for fixed N=64. Because calibration used more
episodes than the holdout evaluation, this is a checkpoint-level diagnostic,
not a cheap inference-time improvement. Full commands, per-episode JSON, paired
comparisons, and limitations are in the
[extension note](../results/refinement/README.md).

## What is and is not archived here

Committed artifacts include configurations, run manifests, metrics, plots,
comparison data, and per-episode diagnostics. OGBench datasets and model
checkpoints are omitted because of size. Some checkpoint analyses therefore
require the original local or cloud checkpoint to rerun; the archived JSON
records checkpoint hashes for provenance. The code and small tests can be run
without those checkpoints. See the root README for setup, preflight, training,
evaluation, and plot commands.
