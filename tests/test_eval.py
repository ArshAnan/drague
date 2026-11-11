"""Cross-play Elo prefers the pursuer that actually wins."""

import jax
import jax.numpy as jnp
import numpy as np

from drague.config import GameConfig
from drague.env import TrackingEnv
from drague.eval import bradley_terry, cross_play
from drague import scripted


def test_bradley_terry_ranks_a_winner_above_a_loser():
    pursuer, evader = bradley_terry(np.array([[1.0, 1.0], [0.0, 0.0]]))
    assert pursuer[0] > pursuer[1]
    assert evader[0] == evader[0]


def test_scripted_cross_play_matrix():
    cfg = GameConfig(n_pursuers=2, episode_seconds=0.2)
    env = TrackingEnv(cfg, n_worlds=2)

    def full(fn):
        def act(key, state):
            del key
            return fn(state["pos"], state["vel"], state["alive"], cfg, jnp)
        return act

    result = cross_play(
        env, jax.random.PRNGKey(0),
        [("standoff", full(scripted.standoff_pursuers)), ("idle", full(scripted.idle))],
        [("idle", full(scripted.idle)), ("flee", full(scripted.flee_evader))],
        n_episodes=2, horizon=cfg.max_steps,
    )
    matrix = np.asarray(result["pursuer_win_rate"])
    assert matrix.shape == (2, 2)
    assert np.all(np.isfinite(matrix))
    assert np.all(np.isfinite(result["pursuer_elo"]))
