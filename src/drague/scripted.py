"""Hand-written policies: sanity checks now, held-out opponents later.

Each returns an action array (W, D, 3) in the same format the physics backends take.
`standoff_pursuers` and `flee_evader` return zeros for drones they don't control, so they can
simply be added together. `idle` returns actions for every drone; pick its rows with a mask.
The physics backend caps action magnitude, so these can return unnormalized vectors.
"""

from drague.config import GameConfig


def _unit(v, xp):
    return v / (xp.sqrt(xp.sum(v * v, axis=-1, keepdims=True)) + 1e-8)


def _hold_altitude(pos, cfg: GameConfig, gain=2.0):
    return gain * (cfg.spawn_height - pos[..., 2:3])


def standoff_pursuers(pos, vel, alive, cfg: GameConfig, xp, standoff=0.35, gain=2.0, damping=1.0,
                      separation=0.6, repel=6.0):
    """Each pursuer flies to a point `standoff` meters from the evader, on its own side,
    while pushing away from teammates closer than `separation` meters.

    Aiming at the evader itself makes pursuers converge on one point and crash into each other,
    so each one approaches along its current bearing and keeps its distance from teammates.
    """
    P = cfg.n_pursuers
    pp = pos[:, :P, :]
    ev = pos[:, P:P + 1, :]
    bearing = _unit(pp - ev, xp)
    target = ev + standoff * bearing
    a = gain * (target - pp) - damping * vel[:, :P, :]

    diff = pp[:, :, None, :] - pp[:, None, :, :]                   # (W, P, P, 3): i minus j
    d = xp.sqrt(xp.sum(diff * diff, axis=-1, keepdims=True))
    push = xp.maximum(0.0, separation - d) / separation            # 0 when far apart
    other_alive = alive[:, None, :P, None]
    a = a + repel * xp.sum(push * other_alive * diff / (d + 1e-8), axis=2)
    a = xp.concatenate([a[..., :2], _hold_altitude(pos[:, :P, :], cfg)], axis=-1)
    return xp.concatenate([a, xp.zeros_like(pos[:, P:, :])], axis=1)


def flee_evader(pos, vel, alive, cfg: GameConfig, xp, wall_margin=0.7):
    """Evader runs directly away from the closest live pursuer and steers inward near the wall."""
    P = cfg.n_pursuers
    ev = pos[:, P, :]                                              # (W, 3)
    diff = ev[:, None, :] - pos[:, :P, :]                          # (W, P, 3)
    d = xp.sqrt(xp.sum(diff * diff, axis=-1))
    d = xp.where(alive[:, :P], d, 1e6)
    nearest = xp.argmin(d, axis=1)                                 # (W,)
    away = xp.take_along_axis(diff, nearest[:, None, None], axis=1)[:, 0, :]
    away_xy = _unit(away[:, :2], xp)
    r = xp.sqrt(xp.sum(ev[:, :2] ** 2, axis=-1, keepdims=True))
    near_wall = r > (cfg.arena_radius - wall_margin)
    inward = xp.where(near_wall, -ev[:, :2] / (r + 1e-8), 0.0)
    a_xy = away_xy + 1.5 * inward
    a = xp.concatenate([a_xy, _hold_altitude(ev, cfg)], axis=-1)[:, None, :]
    return xp.concatenate([xp.zeros_like(pos[:, :P, :]), a], axis=1)


def idle(pos, vel, alive, cfg: GameConfig, xp):
    """Brake and hold altitude. Useful as the weakest possible opponent."""
    a = -vel
    return xp.concatenate([a[..., :2], _hold_altitude(pos, cfg)], axis=-1)
