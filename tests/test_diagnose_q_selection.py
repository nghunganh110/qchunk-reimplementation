from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from qchunk.diagnose_q_selection import rollout_with_q_trace, summarize


class ThreeStepEnv:
    def reset(self, *, seed):
        self.step_index = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, _action):
        rewards = [-1.0, -1.0, 0.0]
        reward = rewards[self.step_index]
        self.step_index += 1
        done = self.step_index == len(rewards)
        return np.zeros(2, dtype=np.float32), reward, done, False, {"success": float(done)}


def test_q_trace_aligns_discounted_return_with_chunk_start():
    state = SimpleNamespace(config=SimpleNamespace(horizon=2, action_dim=1, discount=0.9))

    def select(_observations, _rng):
        return jnp.zeros((1, 2)), jnp.array([0])

    def score_selected(_observations, _selected):
        return jnp.array([[-2.0], [-2.0]])

    def candidate_mean(_observations, _rng):
        return jnp.array([-3.0])

    episode, _ = rollout_with_q_trace(
        state, ThreeStepEnv(), select, score_selected, candidate_mean, jax.random.PRNGKey(0), 1
    )
    assert episode["length"] == 3 and episode["success"] == 1.0
    assert [chunk["start_step"] for chunk in episode["chunks"]] == [0, 2]
    np.testing.assert_allclose(
        [chunk["discounted_return"] for chunk in episode["chunks"]], [-1.9, 0.0]
    )
    np.testing.assert_allclose(episode["chunks"][0]["optimism"], -0.1)
    np.testing.assert_allclose(episode["chunks"][0]["winner_gap"], 1.0)
    summary = summarize([episode])
    assert summary["successes"] == 1 and summary["total_chunks"] == 2
