"""Tests for the game rules. Each test runs on NumPy and (if installed) JAX via the `xp` fixture."""

import numpy as np

from drague.config import GameConfig
from drague import rules
from drague.rules import OTHER_DIM, SELF_DIM

CFG = GameConfig(n_pursuers=3)
W = 64


def _uniforms(cfg, n_worlds, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((n_worlds, 1)),
            rng.random((n_worlds, cfg.n_pursuers, 3)),
            rng.random((n_worlds, 2)))


def _spawn(xp, cfg=CFG, n_worlds=W, seed=0):
    u = [xp.asarray(a, dtype=xp.float32) for a in _uniforms(cfg, n_worlds, seed)]
    return rules.spawn_positions(*u, cfg, xp)


def _alive(xp, cfg=CFG, n_worlds=1):
    return xp.ones((n_worlds, cfg.n_drones), dtype=bool)


def _pos(xp, rows):
    """Build a single-world position array from a list of [x, y, z] rows."""
    return xp.asarray([rows], dtype=xp.float32)


# ----------------------------------------------------------------------------- spawning

def test_spawn_shape_and_layout(xp):
    for P in (1, 2, 3, 4, 8):
        cfg = GameConfig(n_pursuers=P)
        pos = np.asarray(_spawn(xp, cfg))
        assert pos.shape == (W, P + 1, 3)
        r = np.linalg.norm(pos[:, :P, :2], axis=-1)
        slack = cfg.spawn_jitter * np.sqrt(2) + 1e-5
        assert np.all(np.abs(r - cfg.spawn_ring_radius) <= slack)
        ev = pos[:, P]
        assert np.all(np.abs(ev[:, :2]) <= cfg.evader_spawn_halfwidth + 1e-6)
        assert np.allclose(ev[:, 2], cfg.spawn_height)


def test_spawn_is_safe(xp):
    """Nobody starts crashed, out of bounds, or already tracking the evader."""
    for P in (2, 3, 4, 8):
        cfg = GameConfig(n_pursuers=P)
        pos = _spawn(xp, cfg)
        alive = _alive(xp, cfg, W)
        assert not np.any(np.asarray(rules.out_of_bounds(pos, cfg, xp)))
        assert not np.any(np.asarray(rules.pursuer_collisions(pos, alive, cfg, xp)))
        assert not np.any(np.asarray(rules.is_tracked(pos, alive, cfg, xp)))
        d = np.asarray(rules.pairwise_dist(pos, xp))
        d = d + np.eye(P + 1) * 1e9
        assert d.min() > 4 * cfg.collision_radius


# ----------------------------------------------------------------------------- events

def test_tracking(xp):
    near = _pos(xp, [[0.5, 0, 1], [3, 0, 1], [-3, 0, 1], [0, 0, 1]])
    far = _pos(xp, [[1.0, 0, 1], [3, 0, 1], [-3, 0, 1], [0, 0, 1]])
    alive = _alive(xp)
    assert bool(rules.is_tracked(near, alive, CFG, xp)[0])
    assert not bool(rules.is_tracked(far, alive, CFG, xp)[0])
    # a crashed pursuer next to the evader doesn't count
    dead0 = xp.asarray([[False, True, True, True]])
    assert not bool(rules.is_tracked(near, dead0, CFG, xp)[0])


def test_out_of_bounds(xp):
    pos = _pos(xp, [[4.5, 0, 1], [0, 0, 0.1], [0, 0, 3.0], [1, 1, 1]])
    oob = np.asarray(rules.out_of_bounds(pos, CFG, xp))[0]
    assert oob.tolist() == [True, True, True, False]


def test_collisions_only_between_pursuers(xp):
    # pursuers 0 and 1 touch; pursuer 2 touches the evader (not a crash)
    pos = _pos(xp, [[1, 0, 1], [1.1, 0, 1], [-1, 0, 1], [-1.05, 0, 1]])
    hit = np.asarray(rules.pursuer_collisions(pos, _alive(xp), CFG, xp))[0]
    assert hit.tolist() == [True, True, False, False]
    # an already-crashed pursuer can't cause a new crash
    alive = xp.asarray([[True, False, True, True]])
    hit = np.asarray(rules.pursuer_collisions(pos, alive, CFG, xp))[0]
    assert hit.tolist() == [False, False, False, False]


def test_update_alive(xp):
    pos = _pos(xp, [[1, 0, 1], [1.1, 0, 1], [0, 0, 0.05], [2, 2, 1]])
    alive, crashed = rules.update_alive(_alive(xp), pos, CFG, xp)
    assert np.asarray(crashed)[0].tolist() == [True, True, True, False]
    assert np.asarray(alive)[0].tolist() == [False, False, False, True]


# ----------------------------------------------------------------------------- outcome

def _outcome(xp, alive_row, tracked, t):
    alive = xp.asarray([alive_row])
    return [bool(np.asarray(x)[0]) for x in rules.resolve_outcome(
        alive, xp.asarray([tracked]), xp.asarray([t]), CFG, xp)]


def test_outcomes(xp):
    T = CFG.max_steps
    half = int(np.ceil(CFG.win_tracked_fraction * T))
    assert _outcome(xp, [True, True, True, True], 0, 10) == [False, False]          # still playing
    assert _outcome(xp, [True, True, True, False], 0, 10) == [True, True]           # evader crashed
    assert _outcome(xp, [False, False, False, True], 5, 10) == [True, False]        # pursuers all down
    assert _outcome(xp, [False, False, False, False], 0, 10) == [True, True]        # both: evader crash wins
    assert _outcome(xp, [True, False, True, True], half, T) == [True, True]         # timeout, tracked enough
    assert _outcome(xp, [True, True, True, True], half - 1, T) == [True, False]     # timeout, not enough


def test_rewards_zero_sum_and_terminal(xp):
    pos = _pos(xp, [[0.5, 0, 1], [3, 0, 1], [-3, 0, 1], [0, 0, 1]])
    alive = _alive(xp)
    no_crash = xp.zeros((1, CFG.n_drones), dtype=bool)
    tracked = rules.is_tracked(pos, alive, CFG, xp)
    for done, win, sign in ((False, False, 0), (True, True, 1), (True, False, -1)):
        rp, re = rules.team_rewards(pos, alive, tracked, no_crash, xp.asarray([done]),
                                    xp.asarray([win]), CFG, xp)
        rp, re = float(np.asarray(rp)[0]), float(np.asarray(re)[0])
        assert abs(rp + re) < 1e-6
        step_part = 1.0 / CFG.max_steps - CFG.shaping_coef * (0.5 / 8.0) / CFG.max_steps
        assert abs(rp - (step_part + sign)) < 1e-5


def test_crash_penalty(xp):
    pos = _pos(xp, [[3, 0, 1], [-3, 0, 1], [0, 3, 1], [0, 0, 1]])
    alive = _alive(xp)
    crashed = xp.asarray([[True, True, False, False]])
    f = xp.asarray([False])
    base, _ = rules.team_rewards(pos, alive, f, xp.zeros_like(crashed), f, f, CFG, xp)
    hit, _ = rules.team_rewards(pos, alive, f, crashed, f, f, CFG, xp)
    assert abs(float(np.asarray(base - hit)[0]) - 2 * CFG.crash_penalty) < 1e-5


# ----------------------------------------------------------------------------- observations

def _obs(xp, pos, vel, alive):
    return [np.asarray(a) for a in rules.observations(pos, vel, alive, CFG, xp)]


def test_observation_shapes(xp):
    pos = _spawn(xp)
    vel = xp.zeros_like(pos)
    s, o = _obs(xp, pos, vel, _alive(xp, n_worlds=W))
    D = CFG.n_drones
    assert s.shape == (W, D, SELF_DIM)
    assert o.shape == (W, D, D - 1, OTHER_DIM)
    flat = np.asarray(rules.flatten_obs(xp.asarray(s), xp.asarray(o), xp))
    assert flat.shape == (W, D, SELF_DIM + (D - 1) * OTHER_DIM)
    assert np.all(np.isfinite(s)) and np.all(np.isfinite(o))


def test_observation_content(xp):
    pos = _pos(xp, [[1, 0, 1], [-1, 0, 1], [0, 2, 1], [0, 0, 1]])
    vel = xp.zeros_like(pos)
    s, o = _obs(xp, pos, vel, _alive(xp))
    R = CFG.arena_radius
    # pursuer 0 sees: pursuer 1, pursuer 2, then the evader last
    ev_row = o[0, 0, 2]
    assert np.allclose(ev_row[:3], np.array([-1, 0, 0]) / R)    # evader is 1 m in -x
    assert np.isclose(ev_row[6], 1 / R)                          # distance
    assert ev_row[7] == 1 and ev_row[8] == 0                     # is_evader, not same team
    assert o[0, 0, 0, 7] == 0 and o[0, 0, 0, 8] == 1             # teammate row
    assert s[0, 3, 8] == 1 and s[0, 0, 8] == 0                   # is_evader flag on self


def test_observations_translation_invariant(xp):
    """Shifting everyone by the same offset leaves the relative ('others') features unchanged."""
    pos = _spawn(xp)
    vel = xp.asarray(np.random.default_rng(1).normal(size=pos.shape), dtype=xp.float32)
    alive = _alive(xp, n_worlds=W)
    _, o1 = _obs(xp, pos, vel, alive)
    _, o2 = _obs(xp, pos + xp.asarray([0.3, -0.2, 0.1], dtype=xp.float32), vel, alive)
    assert np.allclose(o1, o2, atol=1e-5)


def test_observations_permutation_equivariant(xp):
    """Swapping two pursuers swaps their observations; the evader's view just reorders rows."""
    pos = np.asarray(_spawn(xp))
    vel = np.random.default_rng(2).normal(size=pos.shape).astype(np.float32)
    perm = [1, 0, 2, 3]
    alive = _alive(xp, n_worlds=W)
    s1, o1 = _obs(xp, xp.asarray(pos), xp.asarray(vel), alive)
    s2, o2 = _obs(xp, xp.asarray(pos[:, perm]), xp.asarray(vel[:, perm]), alive)
    assert np.allclose(s2, s1[:, perm], atol=1e-6)
    assert np.allclose(o2[:, 0], o1[:, 1], atol=1e-6)
    assert np.allclose(o2[:, 1], o1[:, 0], atol=1e-6)
    assert np.allclose(o2[:, 3], o1[:, 3][:, [1, 0, 2]], atol=1e-6)
