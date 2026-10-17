# QC implementation specification

## Scope

Implement QC:

- Flow-matching behavior policy.
- Critic over action chunks.
- Best-of-N action selection.
- Offline pretraining and online fine-tuning.

First benchmark: `cube-triple-play-singletask-task2-v0`.

## Notation

| Symbol | Meaning |
| --- | --- |
| $B$ | Batch size |
| $S$ | Observation dimension |
| $A$ | Action dimension |
| $H$ | Action chunk length |
| $N$ | Number of candidate chunks |
| $K$ | Number of critics in the ensemble |
| $T$ | Number of Euler integration steps |
| $\gamma$ | Discount factor |
| $\xi$ | Flow model parameters |
| $\theta_k$ | Parameters of critic $k$ |
| $\bar{\theta}_k$ | Parameters of target critic $k$ |

Represent the action chunk starting at time $t$ as a flattened vector:

$$
\mathbf{a}_t
= \operatorname{vec}(a_t, a_{t+1}, \ldots, a_{t+H-1})
\in \mathbb{R}^{HA}.
$$

The batched action tensor has shape $B \times H \times A$ and is flattened
to $B \times (HA)$ before entering the flow model or critic.

## Flow matching

Sample Gaussian noise and an interpolation time:

$$
z \sim \mathcal{N}(0, I_{HA}),
\qquad
u \sim \mathcal{U}([0, 1]).
$$

Construct the interpolated action and target velocity:

$$
x_u = (1-u)z + u\mathbf{a}_t,
\qquad
v^{\star} = \mathbf{a}_t - z.
$$

Train the vector field $f_{\xi}$ using the mean squared error:

$$
\mathcal{L}_{\mathrm{flow}}(\xi)
= \mathbb{E}_{(s_t,\mathbf{a}_t),z,u}
\left[
\frac{1}{HA}
\left\|f_{\xi}(s_t, x_u, u) - (\mathbf{a}_t-z)\right\|_2^2
\right].
$$

The factor $1/(HA)$ corresponds to averaging over action coordinates.

## Action generation

Generate a candidate chunk using Euler integration:

$$
\begin{aligned}
x^{(0)} &= z, \\
x^{(\ell+1)} &= x^{(\ell)}
+ \frac{1}{T}f_{\xi}\left(s_t, x^{(\ell)}, \frac{\ell}{T}\right).
\end{aligned}
$$

Apply the update for $\ell = 0, \ldots, T-1$.

Clip $x^{(T)}$ coordinatewise to $[-1, 1]$ to obtain a candidate action chunk.
Generate $N$ independent candidates $\mathbf{a}_t^{(1)}, \ldots, \mathbf{a}_t^{(N)}$.
Define the mean Q-value across the critic ensemble:

$$
Q_{\mathrm{mean}}(s,\mathbf{a})
= \frac{1}{K}\sum_{k=1}^{K}Q_{\theta_k}(s,\mathbf{a}).
$$

Select the best candidate:

$$
\begin{aligned}
i^{\star}
&= \operatorname*{arg\,max}_{i\in\{1,\ldots,N\}}
Q_{\mathrm{mean}}(s_t,\mathbf{a}_t^{(i)}), \\
\mathbf{a}_t^{\star} &= \mathbf{a}_t^{(i^{\star})}.
\end{aligned}
$$

Execute its actions sequentially until $H$ steps have
elapsed or the episode ends.

## Critic target — complete chunk

For a complete chunk, compute the discounted return:

$$
R_t^{(H)} = \sum_{j=0}^{H-1}\gamma^j r_{t+j}.
$$

Let $m_t^{(H)}$ be the chunk's bootstrap mask. Use best-of-$N$ selection at
$s_{t+H}$ to obtain $\mathbf{a}_{t+H}^{\star}$, then compute the TD target:

$$
y_t = R_t^{(H)}
+ \gamma^H m_t^{(H)}
\left[
\frac{1}{K}\sum_{k=1}^{K}
Q_{\bar{\theta}_k}(s_{t+H},\mathbf{a}_{t+H}^{\star})
\right].
$$

For a batch of $B$ complete chunks, minimize the mean squared TD error:

$$
\mathcal{L}_{\mathrm{critic}}(\theta)
= \frac{1}{BK}\sum_{b=1}^{B}\sum_{k=1}^{K}
\left(Q_{\theta_k}(s_b,\mathbf{a}_b)-y_b\right)^2.
$$

## Episode boundaries

Episode boundary and bootstrap eligibility are separate:

- `terminals`: indicates an episode boundary.
- `masks`: controls whether the critic target bootstraps.
- `valid`: indicates which positions belong to the starting episode.

For OGBench, these fields are intentionally independent. `terminals = 1` marks
the end of a trajectory stored in the dataset, regardless of task completion.
`masks = 0` marks task completion and disables the Bellman backup, even when the
stored trajectory continues. Therefore both `terminal = 1, mask = 1` and
`terminal = 0, mask = 0` are valid combinations. The chunk sampler uses
`terminals` for sequence boundaries and the TD target uses `masks` for bootstrap.

The transition ending the episode remains valid. Positions after that transition
or beyond the stored dataset are invalid. Never join transitions from different
episodes. Pad invalid actions, rewards, and masks with zero; padded next
observations repeat the last valid next observation.

Flow loss uses the valid action coordinates only. Critic loss uses complete chunks
only, identified by `valid[:, -1] = 1`. A complete chunk ending at an episode
boundary is eligible for critic training; its bootstrap mask decides whether to
use the target critic. An incomplete chunk is not a shorter-horizon critic update.

## Chunk batch contract

| Field | Shape | Meaning |
| --- | --- | --- |
| `observations` | $B \times S$ | Starting states |
| `actions` | $B \times H \times A$ | Consecutive actions, padded if necessary |
| `rewards` | $B \times H$ | Raw rewards, not cumulative returns |
| `next_observations` | $B \times H \times S$ | Next state for each transition |
| `masks` | $B \times H$ | Raw per-transition bootstrap masks |
| `valid` | $B \times H$ | Valid positions in the starting episode |

For sample $b$, compute the discounted reward from the raw batch rewards:

$$
R_b^{(H)} = \sum_{j=0}^{H-1}\gamma^j
\,\mathrm{valid}_{b,j}\,\mathrm{rewards}_{b,j}.
$$

For complete chunks, the bootstrap mask is

$$
m_b^{(H)} = \prod_{j=0}^{H-1}\mathrm{masks}_{b,j}.
$$

For a complete chunk, `next_observations[:, -1, :]` supplies $s_{t+H}$.
Implementation: `src/qchunk/dataset.py`. Verification: `tests/test_dataset.py`.

## Network architecture

Both networks operate on flattened action chunks of dimension $HA$:

$$
f_\xi : \mathbb{R}^{S+HA+1} \rightarrow \mathbb{R}^{HA},
\qquad
Q_{\theta_k} : \mathbb{R}^{S+HA} \rightarrow \mathbb{R}.
$$

For the benchmark configuration, each network uses four hidden layers of width
512 with GELU activations. The critic applies layer normalization after each
hidden activation; the flow vector field does not. The critic ensemble contains
$K=2$ independently initialized networks and returns values with shape
$K \times B$. The output layers are linear.

Implementation: `src/qchunk/networks.py`. Verification: `tests/test_networks.py`.

## Flow implementation contract

The training function receives action chunks with shape $B \times H \times A$
and flattens only their final two axes. One interpolation time $u_b$ is sampled
per chunk and broadcast over its $HA$ action coordinates. Gaussian noise has
the same shape and dtype as the flattened action chunks.

Invalid action positions are zeroed before averaging the squared velocity error.
The denominator remains $BHA$, matching the reference implementation, so an
incomplete chunk contributes less total weight than a complete chunk.

Euler sampling starts from Gaussian noise, evaluates the vector field at
$0, 1/T, \ldots, (T-1)/T$, and applies updates of size $1/T$. Clipping to the
action range $[-1,1]$ occurs after the final Euler update. The same functions
support extra leading axes, including the $N$ candidates needed by best-of-$N$.

Implementation: `src/qchunk/flow.py`. Verification: `tests/test_flow.py`.

## QC target and best-of-N contract

The sampler supplies raw rewards, masks, and validity indicators. The agent
computes $R^{(H)}$ and $m^{(H)}$ from these arrays, then forms the target with
$\gamma^H$. Incomplete chunks have `valid[:, -1] = 0` and contribute zero to
the critic loss while remaining in its fixed-size denominator, matching the
reference implementation.

Best-of-$N$ creates candidates with shape $B \times N \times HA$. The critic
ensemble evaluates them in one call and returns $K \times B \times N$. QC uses
the configured ensemble aggregation (`mean` by default), applies `argmax` over
the $N$ axis, and returns one flattened chunk per observation. The same procedure
selects environment actions and next chunks for TD targets.

Target critic parameters are updated after optimization using

$$
\bar\theta \leftarrow (1-\tau)\bar\theta + \tau\theta.
$$

Implementation: `src/qchunk/agent.py`. Verification: `tests/test_agent.py`.

## Training update contract

The mutable training state contains flow parameters and Adam state, online
critic parameters and Adam state, target critic parameters, and a JAX random
key. One update performs the following operations:

1. Compute flow-matching gradients on the dataset action chunks and update the
   flow network.
2. Use the pre-update flow and online critic to select best-of-$N$ chunks at
   $s_{t+H}$.
3. Evaluate those chunks with the target critic and construct a stopped-gradient
   TD target.
4. Update the online critic from the chunk TD loss.
5. Polyak-update the target critic from the newly updated online critic.

The whole update is JIT-compiled. Default benchmark values are Adam with learning
rate $3\times10^{-4}$, $\gamma=0.99$, $\tau=0.005$, $K=2$, $N=32$, and $T=10$.

Implementation: `src/qchunk/config.py` and `src/qchunk/training.py`.
Verification: `tests/test_training.py`.

## Offline training and checkpoints

Offline pretraining samples start transitions uniformly from the fixed dataset,
builds episode-safe chunks of length $H$, and performs one QC update per sampled
batch. Dataset sampling uses a NumPy generator with its own recorded seed; model
sampling uses the JAX key stored in the training state.

Metrics are transferred to the host only at logging steps to avoid synchronizing
the accelerator after every update. A checkpoint contains flow and critic
parameters, both Adam states, target critic parameters, the current JAX random
key, and a serialized copy of the static QC configuration. Loading requires a
template state with the same configuration and restores enough state to resume
the exact random and optimizer sequence.

Implementation: `src/qchunk/offline.py` and `src/qchunk/checkpoint.py`.
Verification: `tests/test_offline.py`.

## OGBench integration

The first benchmark is `cube-triple-play-singletask-task2-v0` using OGBench
1.1.2, Gymnasium 1.1.1, and MuJoCo 3.3.1, matching the released QC repository.
OGBench is asked for a regular
(`compact_dataset=False`) dataset containing explicit next observations. Its
reward, mask, and terminal arrays are preserved; observations and actions are
converted to float32 and actions are clipped to $[-1+10^{-5},1-10^{-5}]$.

Training and evaluation use separate environment instances. The current adapter
supports state-based vector observations and continuous vector actions. The
benchmark configuration uses $S=46$, $A=5$, and $H=5$, so flow and critic
networks operate on flattened chunks of dimension $HA=25$.

Implementation: `src/qchunk/ogbench.py` and `src/qchunk/train_ogbench.py`.
Configuration: `configs/cube_triple.yaml`. Verification: `tests/test_ogbench.py`.
