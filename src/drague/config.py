"""Game configuration.

Every number that defines the tracking game lives here, so experiments differ only by config.
The dataclass is frozen (hashable), so it can be passed to `jax.jit` as a static argument.

Units: meters, seconds, m/s, m/s^2. Drone index convention: drones 0..n_pursuers-1 are pursuers,
the last drone (index n_pursuers) is the evader.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    # Teams
    n_pursuers: int = 3

    # Arena: a vertical cylinder centered at the origin
    arena_radius: float = 4.0
    z_min: float = 0.2          # below this counts as hitting the floor
    z_max: float = 2.5          # above this counts as leaving the arena

    # Spawning: pursuers on a ring, evader near the center
    spawn_ring_radius: float = 2.5
    spawn_height: float = 1.0
    spawn_jitter: float = 0.2           # max per-axis offset added to each pursuer's ring slot
    evader_spawn_halfwidth: float = 0.5  # evader xy is uniform in [-h, h]^2

    # Interaction distances
    tracking_radius: float = 0.6   # evader counts as "tracked" if any live pursuer is this close
    collision_radius: float = 0.15  # two pursuers closer than this both crash

    # Movement limits (used by the point-mass backend; mirrored as action limits for Crazyflow)
    pursuer_max_speed: float = 1.0
    evader_max_speed: float = 1.3
    pursuer_max_accel: float = 2.0
    evader_max_accel: float = 2.5

    # Timing
    control_dt: float = 0.05        # 20 Hz policy rate
    episode_seconds: float = 20.0

    # Outcome and reward
    win_tracked_fraction: float = 0.5  # pursuers win at timeout if tracked >= this fraction of steps
    shaping_coef: float = 0.1          # weight of the dense distance term (0 disables it)
    crash_penalty: float = 0.2         # team penalty when a pursuer crashes

    @property
    def n_drones(self) -> int:
        return self.n_pursuers + 1

    @property
    def evader_idx(self) -> int:
        return self.n_pursuers

    @property
    def max_steps(self) -> int:
        return int(round(self.episode_seconds / self.control_dt))

    def __post_init__(self):
        if self.n_pursuers < 1:
            raise ValueError("need at least one pursuer")
        if self.spawn_ring_radius + self.spawn_jitter >= self.arena_radius:
            raise ValueError("spawn ring must fit inside the arena")
        if not (self.z_min < self.spawn_height < self.z_max):
            raise ValueError("spawn height must be inside the arena's vertical bounds")
