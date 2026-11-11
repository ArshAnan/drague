"""TrackingEnv: shapes, a scripted match, auto-reset, and jit."""

import jax
import jax.numpy as jnp
import numpy as np

from drague.config import GameConfig
from drague import scripted
from drague.env import TrackingEnv, finished_records
from drague.logbook import EPISODE_FIELDS, LOSS_OUTCOMES, WIN_OUTCOMES


def _actions(state, cfg, pursuer, evader):
    pos, vel, alive = state["pos"], state["vel"], state["alive"]
    a_p = pursuer(pos, vel, alive, cfg, jnp)
    a_e = evader(pos, vel, alive, cfg, jnp)
    is_pursuer = (np.arange(cfg.n_drones) < cfg.n_pursuers)[None, :, None]
    return jnp.where(is_pursuer, a_p, a_e)


def test_reset_shapes_and_safe_spawn():
    cfg = GameConfig(n_pursuers=3)
    env = TrackingEnv(cfg, n_worlds=16)
    state, obs = env.reset(jax.random.PRNGKey(0))
    assert state["pos"].shape == (16, 4, 3)
    assert obs["self"].shape == (16, 4, 10)
    assert obs["others"].shape == (16, 4, 3, 10)
    assert not bool(jnp.any(rules_tracked(state, cfg)))
    assert bool(jnp.all(state["alive"]))
    assert not bool(jnp.any(state["steps"]))


def rules_tracked(state, cfg):
    from drague import rules
    return rules.is_tracked(state["pos"], state["alive"], cfg, jnp)


def test_step_jits_and_autoresets():
    cfg = GameConfig(n_pursuers=2, episode_seconds=0.1)
    env = TrackingEnv(cfg, n_worlds=4)
    step = jax.jit(env.step)
    key = jax.random.PRNGKey(1)
    state, _ = env.reset(key)
    actions = jnp.zeros((4, cfg.n_drones, 3))
    done = jnp.zeros((4,), dtype=bool)
    info = None
    for _ in range(cfg.max_steps):
        key, k = jax.random.split(key)
        state, obs, rewards, done, info = step(k, state, actions)
        assert rewards.shape == (4, cfg.n_drones)
        assert obs["self"].shape == (4, cfg.n_drones, 10)
    assert bool(jnp.all(done))
    assert bool(jnp.all(state["steps"] == 0))
    assert bool(jnp.all(info["length_steps"] == cfg.max_steps))
    records = finished_records(info, done)
    assert len(records) == 4
    for record in records:
        assert set(EPISODE_FIELDS) - {"episode_id", "env_steps_at_end"} <= set(record)
        parts = (record["return_tracking"] + record["return_shaping"]
                 + record["return_crash"] + record["return_terminal"])
        assert abs(parts - record["return_total"]) < 1e-4
        expected = WIN_OUTCOMES if record["pursuers_win"] else LOSS_OUTCOMES
        assert record["outcome"] in expected


def test_standoff_beats_idle_through_the_env():
    cfg = GameConfig(n_pursuers=3)
    env = TrackingEnv(cfg, n_worlds=32)
    step = jax.jit(env.step)
    key = jax.random.PRNGKey(0)
    state, _ = env.reset(key)
    wins = []
    tracked = []
    crashes = 0
    for _ in range(cfg.max_steps):
        key, k = jax.random.split(key)
        actions = _actions(state, cfg, scripted.standoff_pursuers, scripted.idle)
        state, _, _, done, info = step(k, state, actions)
        crashes += int(np.asarray(info["crashes"]).sum())
        if bool(np.asarray(done).any()):
            for record in finished_records(info, done):
                wins.append(record["pursuers_win"])
                tracked.append(record["tracked_fraction"])
    assert len(wins) >= 32
    assert np.mean(wins[:32]) > 0.9
    assert np.mean(tracked[:32]) > 0.5
    assert crashes / 32 < 2.0
