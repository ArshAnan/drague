"""End-to-end sanity checks: scripted policies + point-mass physics + game rules together."""

import numpy as np

from drague.config import GameConfig
from drague import rules, scripted
from drague.physics import point_mass


def _play(cfg, pursuer_policy, evader_policy, xp, n_worlds=64, seed=0):
    """Minimal rollout loop. Returns (pursuer win rate, mean tracked fraction, crashes per episode)."""
    rng = np.random.default_rng(seed)
    P = cfg.n_pursuers
    u = [xp.asarray(a, dtype=xp.float32) for a in
         (rng.random((n_worlds, 1)), rng.random((n_worlds, P, 3)), rng.random((n_worlds, 2)))]
    pos = rules.spawn_positions(*u, cfg, xp)
    vel = xp.zeros_like(pos)
    alive = xp.ones(pos.shape[:2], dtype=bool)
    is_pursuer = (np.arange(cfg.n_drones) < P)[None, :, None]
    t = np.zeros(n_worlds, int); tracked = np.zeros(n_worlds, int)
    done = np.zeros(n_worlds, bool); win = np.zeros(n_worlds, bool); crashes = 0
    for _ in range(cfg.max_steps):
        a = xp.where(is_pursuer, pursuer_policy(pos, vel, alive, cfg, xp),
                     evader_policy(pos, vel, alive, cfg, xp))
        pos, vel = point_mass.step(pos, vel, a, alive, cfg, xp)
        alive, crashed = rules.update_alive(alive, pos, cfg, xp)
        trk = np.asarray(rules.is_tracked(pos, alive, cfg, xp))
        live = ~done
        crashes += int(np.sum(np.asarray(crashed) & live[:, None]))
        t += live
        tracked += trk & live
        d, w = (np.asarray(x) for x in rules.resolve_outcome(alive, xp.asarray(tracked), xp.asarray(t), cfg, xp))
        win = np.where(live & d, w, win)
        done |= d
        if done.all():
            break
    return win.mean(), (tracked / cfg.max_steps).mean(), crashes / n_worlds


def test_standoff_beats_idle_evader_without_crashing(xp):
    cfg = GameConfig(n_pursuers=3)
    win, tracked, crashes = _play(cfg, scripted.standoff_pursuers, scripted.idle, xp)
    assert win > 0.95
    assert tracked > 0.8
    assert crashes < 0.1


def test_idle_pursuers_never_track_a_fleeing_evader(xp):
    cfg = GameConfig(n_pursuers=3)
    win, _, crashes = _play(cfg, scripted.idle, scripted.flee_evader, xp)
    assert win < 0.05
    assert crashes < 0.1
