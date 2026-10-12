"""Tests for the point-mass physics backend."""

import numpy as np

from drague.config import GameConfig
from drague.physics import point_mass

CFG = GameConfig(n_pursuers=3)


def _zeros(xp, W=4):
    return xp.zeros((W, CFG.n_drones, 3), dtype=xp.float32)


def test_zero_action_keeps_velocity(xp):
    pos = _zeros(xp)
    vel = xp.zeros_like(pos) + xp.asarray([0.5, 0.0, 0.0], dtype=xp.float32)
    alive = xp.ones(pos.shape[:2], dtype=bool)
    p, v = point_mass.step(pos, vel, xp.zeros_like(pos), alive, CFG, xp)
    assert np.allclose(np.asarray(v), np.asarray(vel))
    assert np.allclose(np.asarray(p)[..., 0], 0.5 * CFG.control_dt)


def test_speed_limits_per_role(xp):
    """Full throttle for a long time: pursuers cap at their speed, the evader at its higher speed."""
    pos = _zeros(xp)
    vel = xp.zeros_like(pos)
    alive = xp.ones(pos.shape[:2], dtype=bool)
    action = xp.zeros_like(pos) + xp.asarray([1.0, 1.0, 0.0], dtype=xp.float32)  # gets norm-capped
    for _ in range(200):
        pos, vel = point_mass.step(pos, vel, action, alive, CFG, xp)
    speed = np.linalg.norm(np.asarray(vel), axis=-1)
    assert np.allclose(speed[:, :CFG.n_pursuers], CFG.pursuer_max_speed, atol=1e-4)
    assert np.allclose(speed[:, CFG.evader_idx], CFG.evader_max_speed, atol=1e-4)


def test_accel_is_norm_capped(xp):
    pos = _zeros(xp)
    alive = xp.ones(pos.shape[:2], dtype=bool)
    action = xp.zeros_like(pos) + xp.asarray([10.0, 0.0, 0.0], dtype=xp.float32)
    _, v = point_mass.step(pos, xp.zeros_like(pos), action, alive, CFG, xp)
    v = np.asarray(v)
    assert np.allclose(v[:, 0, 0], CFG.pursuer_max_accel * CFG.control_dt, atol=1e-6)
    assert np.allclose(v[:, CFG.evader_idx, 0], CFG.evader_max_accel * CFG.control_dt, atol=1e-6)


def test_dead_drones_freeze(xp):
    pos = _zeros(xp) + 1.0
    vel = xp.zeros_like(pos) + 0.5
    alive = xp.asarray(np.array([[False, True, True, True]] * 4))
    action = xp.zeros_like(pos) + 1.0
    p, v = point_mass.step(pos, vel, action, alive, CFG, xp)
    p, v = np.asarray(p), np.asarray(v)
    assert np.allclose(v[:, 0], 0.0) and np.allclose(p[:, 0], 1.0)
    assert not np.allclose(p[:, 1], 1.0)
