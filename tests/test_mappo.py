"""The shared network acts for every team size, and one PPO update is finite."""

import jax
import numpy as np

from drague.algos.mappo import collect_rollout, deterministic_role_actions, init_role, make_optimizer, update_pursuer
from drague.algos.policy import ActorCritic
from drague.config import GameConfig
from drague.env import TrackingEnv


def test_one_network_acts_for_two_team_sizes():
    net = ActorCritic(hidden=32)
    key = jax.random.PRNGKey(0)
    small = TrackingEnv(GameConfig(n_pursuers=2, episode_seconds=0.1), n_worlds=2)
    large = TrackingEnv(GameConfig(n_pursuers=4, episode_seconds=0.1), n_worlds=2)
    _, obs_s = small.reset(key)
    _, obs_l = large.reset(key)
    params, _ = init_role(key, net, obs_s["self"][:, :2], obs_s["others"][:, :2])
    small_actions = deterministic_role_actions(params, net, obs_s["self"][:, :2], obs_s["others"][:, :2])
    large_actions = deterministic_role_actions(params, net, obs_l["self"][:, :4], obs_l["others"][:, :4])
    assert small_actions.shape == (2, 2, 3)
    assert large_actions.shape == (2, 4, 3)
    assert np.all(np.isfinite(np.asarray(small_actions)))
    assert np.all(np.isfinite(np.asarray(large_actions)))


def test_one_update_returns_finite_losses():
    cfg = GameConfig(n_pursuers=2, episode_seconds=0.1)
    env = TrackingEnv(cfg, n_worlds=2)
    net = ActorCritic(hidden=32)
    key = jax.random.PRNGKey(1)
    state, obs = env.reset(key)
    p_params, p_opt = init_role(key, net, obs["self"][:, :2], obs["others"][:, :2])
    e_params, _ = init_role(key, net, obs["self"][:, 2:3], obs["others"][:, 2:3])
    key, state, frames, p_boot, _ = collect_rollout(
        key, state, p_params, e_params, cfg.max_steps, env, net)
    _, _, metrics = update_pursuer(p_params, p_opt, make_optimizer(), frames, p_boot, net, epochs=1)
    for name in ("policy_loss", "value_loss", "entropy"):
        assert np.isfinite(float(metrics[name]))
