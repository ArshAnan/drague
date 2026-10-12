# Tracking game spec (v0)

Status: draft for review by both of us. Code in `src/drague/` implements everything marked **implemented**.
Numbers below are the defaults in `src/drague/config.py`; change them there, not here.

## 1. The game

- **Teams.** `P` pursuers (the "team" we study) vs 1 evader. Drone indices `0..P-1` are pursuers, index `P` is the evader.
- **Arena.** A vertical cylinder, radius 4.0 m, valid height 0.2–2.5 m. **implemented**
- **Spawn.** Pursuers evenly spaced on a 2.5 m ring at 1.0 m height, with the ring randomly rotated per episode and up to ±0.2 m jitter per axis. Evader uniformly inside a 1 m × 1 m box at the center. Nobody starts crashed or already tracking. **implemented, tested for P = 1–8**
- **Tracking.** The evader is *tracked* on a step if any live pursuer is within 0.6 m of it. **implemented**
- **Crashes.**
  - Any drone leaving the cylinder, going below 0.2 m or above 2.5 m crashes.
  - Two pursuers closer than 0.15 m both crash.
  - Pursuer-evader contact is *not* a crash.
  - Crashed drones freeze in place and stop counting.
  - **implemented**
- **Episode length.** 20 s at a 20 Hz policy rate = 400 steps.
- **Outcome** (binary, zero-sum). If several things happen on the same step, the first matching rule wins:
  1. Evader crashed → pursuers win (episode ends).
  2. All pursuers crashed → evader wins (episode ends).
  3. Time up → pursuers win if the evader was tracked on at least 50% of steps, otherwise the evader wins.

  **implemented**
- **Movement limits** (point-mass backend):
  - Pursuers: 1.0 m/s, 2.0 m/s².
  - Evader: 1.3 m/s, 2.5 m/s².

  The evader is faster, so pursuers need teamwork to win. These are the main knobs for balancing.

## 2. Rewards (zero-sum)

Per step, the pursuer team gets:

| Term | Value |
|---|---|
| Tracking | `+1/T` if tracked this step (sums to the tracked fraction over an episode) |
| Shaping | `-0.1 · d_min / (2R) / T`, where `d_min` is the closest live pursuer's distance to the evader |
| Crash | `-0.2` per pursuer that crashed this step |
| Terminal | `+1` win / `-1` loss on the final step |

The evader gets exactly the negative. Every pursuer receives the same team reward. **implemented**

Note: zero-sum means the evader is rewarded when pursuers crash, so it may learn to bait collisions. That's legitimate adversarial play, and exactly the kind of weakness exploiters should find.

## 3. Observations

Every drone gets an egocentric, entity-based observation. **implemented, tested for translation invariance and permutation equivariance**

- `self` (10 values): `pos/R (3)`, `vel/vmax (3)`, `wall clearance (1)`, `height fraction (1)`, `is_evader (1)`, `alive (1)`
- `others` (one 10-value row per other drone): `rel_pos/R (3)`, `rel_vel/vmax (3)`, `dist/R (1)`, `is_evader (1)`, `same_team (1)`, `alive (1)`
- For pursuer `i`, the rows are the other pursuers in index order, then the evader last.
- `flatten_obs` concatenates both into one vector for an MLP baseline. Use attention or deep-sets pooling over `others` for the main experiments, so one network works for any team size.

v0 is fully observable. A limited sensing radius is an open question (section 6).

## 4. Actions

`(W, D, 3)` floats. Each drone's action is a desired acceleration direction and magnitude:

- Magnitude above 1 is capped to 1, then scaled by that drone's max acceleration.
- Speed is capped after integration.
- Actions for crashed drones are ignored.

For the Crazyflow backend we'll map the same 3-vector onto Crazyflie commands (most likely a velocity/acceleration setpoint through state control), so policies and training code don't change.

## 5. Environment API (the contract between env and training code)

The env is **batched over W worlds** natively, because Crazyflow is. All functions are pure and jit-compatible.

```python
env = TrackingEnv(cfg: GameConfig, backend="point_mass" | "crazyflow", n_worlds=W)

state, obs = env.reset(key)
state, obs, rewards, dones, info = env.step(key, state, actions)
```

| Name | Shape / type | Meaning |
|---|---|---|
| `actions` | `(W, D, 3)` float | rows `0..P-1` from the pursuer policy, row `P` from the evader policy |
| `obs["self"]` | `(W, D, 10)` | see section 3 |
| `obs["others"]` | `(W, D, D-1, 10)` | see section 3 |
| `rewards` | `(W, D)` | team reward in pursuer columns, evader reward in the last column |
| `dones` | `(W,)` bool | episode ended this step; that world auto-resets and `obs` is from the new episode |
| `info["pursuers_win"]` | `(W,)` bool | final outcome, valid where `dones` |
| `info["tracked_fraction"]` | `(W,)` float | fraction of steps tracked so far (final value where `dones`) |
| `info["crashes"]` | `(W, D)` bool | crashed this step |

The env knows nothing about which policy controls which side. The league code picks the pursuer and evader policies and stacks their actions. Training code should count **env steps = W × number of `step` calls** for the equal-compute budget.

`TrackingEnv` itself is the next piece to write: it's a thin wrapper around the functions in `rules.py` plus a physics backend.

## 6. Open questions (decide together)

1. **Early termination.** Should an episode end once the outcome is mathematically decided (e.g. tracked ≥ 50% already)? It saves compute, but changes the learning signal late in episodes.
2. **Observability.** Full observability (v0) vs a sensing radius for teammates/evader. Partial observability makes teamwork matter more but makes learning harder.
3. **Balance per team size.** Scripted pursuers (`scripted.standoff_pursuers`) vs a fleeing evader win 2% / 5% / 28% / 51% for P = 2 / 3 / 4 / 8. We should tune evader speed per team size so self-play sits near 50/50 at the middle budget.
4. **Crazyflow action mapping.** State control (position/velocity setpoints, easiest to learn) vs attitude control (more agile, harder). Decide after the spike.
5. **Shaping coefficient.** 0.1 is a guess. Check that turning it off doesn't change who wins in self-play, only how fast learning goes.
