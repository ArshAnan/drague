"""Rules of the tracking game, written as pure array functions.

Why this shape:
- Every function takes `xp`, an array namespace (`numpy` or `jax.numpy`). The same code runs in
  NumPy for quick tests and in JAX inside `jax.jit` for training. Nothing mutates arrays in place,
  and nothing draws random numbers internally (callers pass uniform samples in), which keeps the
  functions jit-safe and deterministic.
- Everything is batched over a leading "world" axis W (parallel arenas), then a drone axis D.
  Drones 0..P-1 are pursuers; drone P (the last one) is the evader.

Shapes used below:
    pos, vel   (W, D, 3)   positions [m] and velocities [m/s]
    alive      (W, D)      bool, False once a drone has crashed
"""

from math import pi

import numpy as np

from drague.config import GameConfig

SELF_DIM = 10
OTHER_DIM = 10


# ----------------------------------------------------------------------------- helpers

def drone_limits(cfg: GameConfig, xp):
    """Per-drone max speed and max acceleration, shape (D,). The evader is the last drone."""
    is_evader = xp.arange(cfg.n_drones) == cfg.evader_idx
    max_speed = xp.where(is_evader, cfg.evader_max_speed, cfg.pursuer_max_speed)
    max_accel = xp.where(is_evader, cfg.evader_max_accel, cfg.pursuer_max_accel)
    return max_speed, max_accel


def _norm(x, xp, axis=-1, keepdims=False):
    return xp.sqrt(xp.sum(x * x, axis=axis, keepdims=keepdims))


def pairwise_dist(pos, xp):
    """Distance between every pair of drones in each world. (W, D, D), zero on the diagonal."""
    diff = pos[:, :, None, :] - pos[:, None, :, :]
    return _norm(diff, xp)


def pursuer_evader_dist(pos, cfg: GameConfig, xp):
    """Distance from each pursuer to the evader. (W, P)."""
    P = cfg.n_pursuers
    return _norm(pos[:, :P, :] - pos[:, P:P + 1, :], xp)


# ----------------------------------------------------------------------------- spawning

def spawn_positions(u_phase, u_jitter, u_evader, cfg: GameConfig, xp):
    """Start positions: pursuers evenly spaced on a ring, evader near the center.

    The randomness comes in as uniform [0, 1) samples so this stays a pure function:
        u_phase   (W, 1)     rotates the whole ring per world
        u_jitter  (W, P, 3)  small per-pursuer offset
        u_evader  (W, 2)     evader xy inside a small box at the center
    Returns (W, D, 3).
    """
    P = cfg.n_pursuers
    ang = 2 * pi * u_phase + xp.arange(P) * (2 * pi / P)            # (W, P)
    ring = xp.stack(
        [cfg.spawn_ring_radius * xp.cos(ang),
         cfg.spawn_ring_radius * xp.sin(ang),
         xp.zeros_like(ang) + cfg.spawn_height],
        axis=-1,
    )                                                                 # (W, P, 3)
    pursuers = ring + (2 * u_jitter - 1) * cfg.spawn_jitter
    exy = (2 * u_evader - 1) * cfg.evader_spawn_halfwidth             # (W, 2)
    ez = xp.zeros_like(exy[:, :1]) + cfg.spawn_height                 # (W, 1)
    evader = xp.concatenate([exy, ez], axis=-1)[:, None, :]           # (W, 1, 3)
    return xp.concatenate([pursuers, evader], axis=1)


# ----------------------------------------------------------------------------- events

def is_tracked(pos, alive, cfg: GameConfig, xp):
    """True where at least one live pursuer is within the tracking radius of the evader. (W,)"""
    close = pursuer_evader_dist(pos, cfg, xp) < cfg.tracking_radius
    return xp.any(close & alive[:, :cfg.n_pursuers], axis=1)


def out_of_bounds(pos, cfg: GameConfig, xp):
    """True where a drone has left the cylinder, hit the floor, or gone above the ceiling. (W, D)"""
    r_xy = _norm(pos[..., :2], xp)
    z = pos[..., 2]
    return (r_xy > cfg.arena_radius) | (z < cfg.z_min) | (z > cfg.z_max)


def pursuer_collisions(pos, alive, cfg: GameConfig, xp):
    """True for each live pursuer touching another live pursuer. (W, D); evader column is False.

    Pursuer-evader contact is not a crash: it just means the evader is very well tracked.
    """
    P = cfg.n_pursuers
    d = pairwise_dist(pos[:, :P, :], xp)                              # (W, P, P)
    off_diag = ~np.eye(P, dtype=bool)
    alive_p = alive[:, :P]
    both_alive = alive_p[:, :, None] & alive_p[:, None, :]
    hit = xp.any((d < cfg.collision_radius) & off_diag & both_alive, axis=2)   # (W, P)
    return xp.concatenate([hit, xp.zeros_like(hit[:, :1])], axis=1)


def update_alive(alive, pos, cfg: GameConfig, xp):
    """Apply crashes for this step. Returns (new_alive, crashed_now), both (W, D)."""
    crashed_now = alive & (out_of_bounds(pos, cfg, xp) | pursuer_collisions(pos, alive, cfg, xp))
    return alive & ~crashed_now, crashed_now


def resolve_outcome(alive, tracked_steps, t, cfg: GameConfig, xp):
    """Decide whether each world's episode is over and who won.

    Call after `t` and `tracked_steps` have been incremented for the current step.
    Priority when several things happen on the same step:
        1. evader crashed            -> pursuers win
        2. every pursuer crashed     -> evader wins
        3. time is up                -> pursuers win iff tracked fraction >= win_tracked_fraction
    Returns (done, pursuers_win), both (W,) bool. `pursuers_win` is only meaningful where done.
    """
    evader_down = ~alive[:, cfg.evader_idx]
    pursuers_down = ~xp.any(alive[:, :cfg.n_pursuers], axis=1)
    timeout = t >= cfg.max_steps
    won_on_time = (tracked_steps / cfg.max_steps) >= cfg.win_tracked_fraction
    done = evader_down | pursuers_down | timeout
    pursuers_win = evader_down | (~pursuers_down & timeout & won_on_time)
    return done, pursuers_win


# ----------------------------------------------------------------------------- reward

def team_rewards(pos, alive, tracked_now, crashed_now, done, pursuers_win, cfg: GameConfig, xp):
    """Zero-sum rewards. Returns (pursuer_team_reward, evader_reward), each (W,).

    Pursuer team reward per step:
        + 1/T                          if the evader is tracked this step (sums to tracked fraction)
        - shaping_coef * d_min / (2R) / T   dense pull toward the evader (d_min over live pursuers)
        - crash_penalty * (#pursuers that crashed this step)
        +1 / -1                        on the final step, for a win / loss
    The evader gets exactly the negative, so the game is zero-sum.
    """
    T = cfg.max_steps
    P = cfg.n_pursuers
    diameter = 2.0 * cfg.arena_radius
    d = xp.where(alive[:, :P], pursuer_evader_dist(pos, cfg, xp), diameter)
    d_min = xp.min(d, axis=1)

    r = tracked_now * (1.0 / T)
    r = r - cfg.shaping_coef * (d_min / diameter) / T
    r = r - cfg.crash_penalty * xp.sum(crashed_now[:, :P] * 1.0, axis=1)
    r = r + xp.where(done, xp.where(pursuers_win, 1.0, -1.0), 0.0)
    return r, -r


# ----------------------------------------------------------------------------- observations

def _others_index(D):
    """For each drone i, the indices of every other drone in ascending order. (D, D-1), static."""
    return np.array([[j for j in range(D) if j != i] for i in range(D)], dtype=np.int32)


def observations(pos, vel, alive, cfg: GameConfig, xp):
    """Egocentric, entity-based observations for every drone.

    Returns
        self_feat  (W, D, SELF_DIM)          what a drone knows about itself
        others     (W, D, D-1, OTHER_DIM)    one row per other drone, relative to the observer

    self_feat  = [pos/R (3), vel/vmax (3), wall_clearance (1), height_frac (1), is_evader (1), alive (1)]
    others row = [rel_pos/R (3), rel_vel/vmax (3), dist/R (1), is_evader (1), same_team (1), alive (1)]

    Entity rows (rather than one flat vector) let the policy use attention or deep-sets pooling,
    so one network can handle any team size. Use `flatten_obs` for a plain MLP baseline.
    For pursuer i, rows are the other pursuers in index order, then the evader last.
    """
    W, D, _ = pos.shape
    R = cfg.arena_radius
    vmax = max(cfg.pursuer_max_speed, cfg.evader_max_speed)

    is_ev = (xp.arange(D) == cfg.evader_idx)                          # (D,)
    is_ev_wd = xp.zeros_like(alive) | is_ev                           # (W, D) bool
    wall = (R - _norm(pos[..., :2], xp)) / R
    height = (pos[..., 2] - cfg.z_min) / (cfg.z_max - cfg.z_min)
    self_feat = xp.concatenate(
        [pos / R, vel / vmax, wall[..., None], height[..., None],
         is_ev_wd[..., None] * 1.0, alive[..., None] * 1.0],
        axis=-1,
    )

    rel_pos = pos[:, None, :, :] - pos[:, :, None, :]                 # [w, i, j] = pos_j - pos_i
    rel_vel = vel[:, None, :, :] - vel[:, :, None, :]
    dist = _norm(rel_pos, xp, keepdims=True)
    same_team = (is_ev[:, None] == is_ev[None, :])                    # (D, D)
    full = xp.concatenate(
        [rel_pos / R, rel_vel / vmax, dist / R,
         (xp.zeros_like(dist) + is_ev[None, None, :, None]),
         (xp.zeros_like(dist) + same_team[None, :, :, None]),
         (xp.zeros_like(dist) + alive[:, None, :, None])],
        axis=-1,
    )                                                                 # (W, D, D, OTHER_DIM)
    idx = _others_index(D)
    rows = np.arange(D)[:, None]
    others = full[:, rows, idx]                                       # (W, D, D-1, OTHER_DIM)
    return self_feat, others


def flatten_obs(self_feat, others, xp):
    """Concatenate entity observations into one vector per drone: (W, D, SELF_DIM + (D-1)*OTHER_DIM)."""
    W, D = self_feat.shape[:2]
    return xp.concatenate([self_feat, others.reshape(W, D, -1)], axis=-1)
