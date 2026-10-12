"""Point-mass (double integrator) drones.

Each drone is a point with a velocity. The action is a desired acceleration; we cap its
magnitude, integrate with semi-implicit Euler, then cap the speed. Pursuers and the evader
can have different limits (see GameConfig), which is how the game is balanced.
Crashed drones stop where they are.
"""

from drague.config import GameConfig
from drague.rules import drone_limits


def step(pos, vel, action, alive, cfg: GameConfig, xp):
    """Advance one control step (cfg.control_dt). All arrays (W, D, 3) except alive (W, D)."""
    max_speed, max_accel = drone_limits(cfg, xp)
    dt = cfg.control_dt

    # Cap the action's magnitude at 1 (direction is preserved), then scale per drone.
    a_norm = xp.sqrt(xp.sum(action * action, axis=-1, keepdims=True))
    action = action * xp.minimum(1.0, 1.0 / xp.maximum(a_norm, 1e-8))
    accel = action * max_accel[None, :, None]

    v = vel + accel * dt
    speed = xp.sqrt(xp.sum(v * v, axis=-1, keepdims=True))
    v = v * xp.minimum(1.0, max_speed[None, :, None] / xp.maximum(speed, 1e-8))
    v = xp.where(alive[..., None], v, 0.0)
    return pos + v * dt, v
