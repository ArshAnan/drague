"""Physics backends.

Every backend exposes the same step signature so the env doesn't care which one it uses:

    pos, vel = step(pos, vel, action, alive, cfg, xp)

with `action` of shape (W, D, 3) in [-1, 1], interpreted as a desired acceleration direction
and magnitude (scaled by each drone's max acceleration).

- `point_mass`: double-integrator drones. Fast, simple, and the fallback if quadrotor
  dynamics turn out too hard to learn on. Lets training code be developed right away.
- `crazyflow` (coming next, after the week-1 spike): real Crazyflie dynamics via Crazyflow.
"""
