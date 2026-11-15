# Experiment log

Newest entries at the top. Each entry: date, who, commit, what was run, result, decision.

---

## 2026-11-12 · Arsh · pursuer collapse: cause found, partly fixed

All diagnostics train only the pursuers (P = 3 unless noted) against a frozen evader that never moves, at W = 128 and 20–30M env steps. Scratch script; not in the repo.

**How it collapses.** At the peak (~4–7M steps, win 0.8–0.9) all three pursuers crowd the evader (closest pursuer–pursuer distance ~0.6 m) and ~2 of 3 crash into each other every episode. Then the policy flips to "spread out": ~2.7 m apart, no crashes, no tracking, every episode lost. It stays there because once every episode is lost, the −1 terminal reward is the same everywhere. Approx KL stays at 0.002–0.004 and clip fraction at 1–4%, so it is not PPO step size. Pre-tanh |mean| ~0.15, so it is not tanh saturation.

**Ruled out (all still collapse):** crash penalty 0.05; shaping 4.0; `gamma` 0.995; masking crashed pursuers out of the policy loss (kept anyway as standard MAPPO practice). The learning-rate annealing variant crashed in the scratch setup and was not retested.

**Control.** P = 1 is perfectly stable with both networks: win 1.00 from 4M to 20M, tracked 0.88, zero crashes. So PPO is fine; the problem is multi-pursuer.

**Cause: the network.** `ActorCritic` encoded every other drone with one layer and mean-pooled teammates and the evader together, so "teammate close" and "evader close" looked alike. Learning to avoid teammates also taught pursuers to back off the evader: at the flip, distance to the evader jumps from 0.58 m to 2.2 m.

**Change** (uncommitted on top of `3e3d97a`):
- `algos/policy.py`: separate two-layer encoders for teammates and opponents, split by the `same_team` flag; masked mean and max pooling per group; two-layer trunk; named `mean_head` and `value_head`. Still independent of team size. **Old checkpoints don't load into it.**
- `algos/mappo.py`: active-mask policy loss; logs `approx_kl`, `clip_frac`, `active_frac`, `mean_abs`; `make_optimizer(learning_rate)` accepts a schedule; `gamma` is now an argument.
- `config.py`: `shaping_coef` default 0.1 → **1.0**; spec section 2 and open question 5 updated.

**Result with the split network:**

| Run | 4M | 12M | 20M | 30M |
|---|---|---|---|---|
| shaping 0.1, seed 0 | win 0.00 | 0.00 | 0.00 (track 0.08) | |
| shaping 1.0, seed 0 | 0.00 | 0.82 | 1.00 (track 0.80) | |
| shaping 1.0, seed 1 | 0.03 at 3M, 0.99 at 6M | 1.00 | 0.98 | 0.99 (track 0.81) |
| shaping 1.0, seed 2 | 0.00 | 0.00 | 0.00 | 0.00 (track 0.01) |

Two of three seeds learn and **hold** (seed 1 above 0.9 from 6M to 30M), with fewer crashes (0.2–1.2 per episode vs ~2). Seed 2 never takes off. Its entropy falls from 2.3 to −1.7 while the pursuers sit ~2 m apart, so the remaining failure is premature convergence: with `ENTROPY = 0` nothing keeps exploration alive. Entropy also goes negative in the successful seeds.

**Self-play with the new defaults** (`2026-11-12_18-20-46_selfplay_p3_s0`, 20M, W = 128; the manifest records 9,281 s wall, run alongside two diagnostics):
- First real tracking in self-play. From 18M on, pursuers win 0.58–0.60 and track 0.52–0.56; out-of-bounds crashes near 0.
- **They don't generalize.** Final checkpoints, 64 episodes each:

| Pursuers (mean action) vs | win | tracked |
|---|---|---|
| their own trained evader | 0.47 | 0.47 |
| scripted `idle` | 0.00 | 0.03 |
| scripted `flee` | 0.00 (wiped 38/64) | 0.00 |

  Sampling actions instead gives the same picture, so it is not a deterministic-eval artifact. The pursuers learned to beat their one partner, not to track. The fresh exploiter (2M) also wipes them 16/16.

**Next.**
1. Exploration: a small entropy bonus now that updates are minibatched (e.g. 1e-3), or a floor on `log_std`. Rerun the 3 seeds against the motionless evader. The gate stays: every seed ≥ 0.9.
2. Then self-play again and check the held-out evals. The idle/flee gap is the self-play brittleness the project is about, but at 0.00 vs `idle` it is too extreme to compare methods on yet. PFSP or league training, or mixing scripted opponents into training, may close it; that is the actual experiment.
3. Speed: the new network is ~3× slower per step, and `train.py` adds logging overhead on top.

---

## 2026-11-12 · Arsh · PPO update fixed; pursuers learn, then collapse

**Change** (`algos/mappo.py`, uncommitted on top of `3e3d97a`, so these runs are marked `git_dirty`):
- One jitted update per rollout: `EPOCHS = 4` shuffled passes × `MINIBATCHES = 8`, so 32 gradient steps instead of 2.
- Gradient clipping at global norm 0.5.
- `ENTROPY = 0.0`: the pre-tanh entropy bonus was what pushed std up.
- `train.py` passes a fresh key to every update and logs `minibatches`, `entropy_coef` and `max_grad_norm` in `hparams`.

**Self-play rerun, P = 3, seed 0** (`2026-11-12_16-55-31_selfplay_p3_s0` 2M/W = 32, `2026-11-12_16-55-34_selfplay_p3_s0` 10M/W = 128):

| | 2M before → after | 10M before → after |
|---|---|---|
| Entropy, end | 2.99 → 2.23 | 3.08 → 1.67 |
| Episode length, end | 155 → 400 | 342 → 383 |
| Pursuer OOB / episode, end | 2.28 → 0.00 | 1.89 → 0.85 |
| Pursuers vs `idle`, win | 0/16 → 0/16 | 0/16 → 0/16 |
| Fresh exploiter vs frozen pursuers, pursuer win | 0.69 → 0.00 | 1.00 → 0.00 |

Training is now stable in the basic sense: entropy falls, drones mostly stay in the arena, and a fresh exploiter beats the frozen team 16/16 by timeout, so the exploitability measurement now works. Pursuers still don't track. In the 10M run the evader learned to stay away (mean `d_min` 4.4 m at the end), and pursuers collide with each other 0.5–1 times per episode.

**Diagnostic: pursuers alone vs a frozen, motionless evader** (scratch script; MAPPO pursuer update only; evader net with zeroed action head):

| Shaping | 2M steps (W = 32) | peak (~4M, W = 128) | 20M steps (W = 128) |
|---|---|---|---|
| 0.1 (default) | track 0.21, win 0.00 | track 0.36, win 0.24 | track 0.07, win 0.00, entropy 0.52 |
| 1.0 | track 0.49, win 0.44 | track 0.55, win 0.62 | track 0.04, win 0.00, entropy 0.30 |

So the pursuer side can learn the task, and stronger shaping learns it about twice as fast. But after ~4M steps it **collapses**: tracking goes to ~0 and stays there while entropy keeps falling. Self-play can't work until a single-agent run holds on to this.

**Likely causes, not yet tested.**
- Reward scale. Tracking is worth `1/T` = 0.0025 per step, while a crash costs 0.2 at once, as much as 80 tracked steps. With `gamma = 0.99` (horizon ~100 steps) the terminal ±1 at step 400 barely reaches early decisions. After advantage normalization, the rare large crash penalties dominate, so "stay apart, don't approach" is a safe local optimum. Crowding a target also causes pursuer–pursuer collisions (0.15 m radius).
- No learning-rate decay, value normalization or KL monitoring, so nothing stops a late collapse.

**Timing.** The standalone loop trains 2M steps in 22 s on CPU. `train.py` takes 168 s for the same cell, so most of its wall time is per-episode logging and Python overhead, not the network or physics.

**Next.**
1. Make "pursuers vs motionless evader reach and hold win ≥ 0.9" a gate before any self-play. Candidate fixes, one at a time: per-step tracking reward on an O(1) scale (or a smaller crash penalty), `gamma` 0.995, learning-rate annealing, return/value normalization, logging approx-KL and clip fraction.
2. Then pursuers vs `flee`, then self-play again.
3. Profile `train.py`'s logging path. A 7× speedup makes the full grid a laptop job.

---

## 2026-11-12 · Arsh · first real self-play runs, P = 3 (laptop CPU, point-mass)

**Setup.** Commit `3e3d97a`, clean tree. MAPPO self-play, default `GameConfig`, seed 0, two cells:
- `--budget paper`: 2M env steps, W = 32, about 2 min wall. Run `2026-11-12_16-49-12_selfplay_p3_s0`.
- `--budget 10000000 --worlds 128 --exploitability`: 10M env steps, about 3 min wall. Run `2026-11-12_16-49-11_selfplay_p3_s0`.

**Throughput.** Training runs at 70k env steps/s with W = 32 and 120k with W = 128 on CPU, including policy and updates. The ~1k from the smoke run was compile time. The 2M paper budget is therefore cheap; budgets can go up 10–100×.

**Result. Neither side learned the task.**

| | 2M (W = 32) | 10M (W = 128) |
|---|---|---|
| Self-play tracked fraction (last interval) | 0.014 | 0.10 |
| Pursuer OOB crashes / episode | 2.28 | 1.89 |
| Held-out: pursuers vs `idle` evader, win | 0/16 | 0/16 |
| Held-out: pursuers vs `flee` evader, win | 1/16 | 4/16 (all evader crashes) |
| Fresh exploiter vs frozen pursuers, pursuer win | 0.69 (200k steps) | 1.00 (1M steps; exploiter crashed 16/16) |
| Policy entropy, start → end | 2.76 → 2.99 | 2.76 → 3.08 |

The scripted `standoff` pursuer beats `idle` 100%. Self-play win rates swing between 0 and 0.8 and are decided by out-of-bounds crashes, not tracking. The exploitability number is meaningless: the fresh exploiter loses by crashing, so the pursuers look unexploitable only because nothing can fly.

**Suspected causes (in `algos/mappo.py`).**
1. Very few updates. `EPOCHS = 2` and each epoch is one full-batch gradient step, so a run gets about 2 steps per 400-step rollout: ~310 optimizer steps for 2M, ~390 for 10M. PPO normally takes many minibatch steps per rollout.
2. The entropy bonus rewards unbounded std. It is the pre-tanh Gaussian entropy, so increasing `log_std` always raises the bonus, but after `tanh` the actions saturate at ±1 (max acceleration in random directions). Entropy rises through both runs, consistent with this and with the OOB crashes.

**Decision.** Don't run the method comparison until a single self-play run beats `idle` and, ideally, `flee`. Fix the PPO update first (minibatches, more epochs, an entropy term that can't push actions to saturation, e.g. a smaller coefficient or a learned-std cap), then rerun these two cells.

**Next.** PPO fixes plus a rerun at P = 3. Then budget sizing (CPU is fast enough for laptop-scale grids) and balance (spec Q3).

---

## 2026-11-05 · Arsh · Crazyflow spike, corrected (laptop CPU)

**Setup.** `scripts/spike_crazyflow.py --sweep` on an M-series MacBook Pro, CPU JAX. Crazyflow 0.3.2, JAX 0.11.2, `so_rpy` dynamics, now with **state control**. One env step = one 50 ms policy step = 25 physics ticks at 500 Hz. Physics only: no policy, rules or observations.

**Why rerun.** The 2026-10-14 run (not logged before now) had two problems:
- **Throughput was overstated ~25–40×.** Attitude control runs at 500 Hz, the same as the physics, so `sim.freq // sim.control_freq` was 1 tick (2 ms), not one 20 Hz policy step. It reported 1.6–1.8M "env steps/s"; the real figure is tens of thousands.
- **Hover failed (0.62 m drift in 2 s)** because open-loop thrust `m·g` is ~3% short of hover in `so_rpy`, a steady 0.31 m/s² sink. Not a simulator bug.

**Result.** 7/7 checks pass. Hover under state control: max drift 0.086 m over 2 s. Env steps/s:

| Pursuers | W = 1024 | W = 4096 | W = 16384 |
|---|---|---|---|
| 2 | 12k* | 85k | 138k |
| 3 | 43k | 80k | 110k |
| 4 | 32k | 70k | 92k |
| 8 | 22k | 44k | 50k |

\* first cell of the sweep; probably includes leftover compile/warm-up time.

**Decisions.**
- Use state control for the Crazyflow backend (spec open question 4). Crazyflow doesn't enforce our per-role speed/accel limits, so `TrackingEnv` must clamp setpoints.
- Pinned `crazyflow==0.3.2`.
- Don't call `sim.step()` from a Python loop in training. Run the step as a pure function inside `jax.lax.scan`.

**Next.** Run the same sweep on an Empire AI GPU (fill in `PARTITION`/`ACCOUNT` in `cluster/spike.sbatch`), then size the budgets from those numbers.

---

## 2026-10-11 · Arsh · scripted baselines on the point-mass backend

**Setup.** Default `GameConfig`: evader 1.3 m/s vs pursuers 1.0 m/s, tracking radius 0.6 m, 20 s episodes, 1024 episodes per cell.

**Result.** Pursuer win rate:

| Pursuers | `standoff_pursuers` vs `idle` evader | `standoff_pursuers` vs `flee_evader` | `idle` pursuers vs `flee_evader` |
|---|---|---|---|
| 2 | 1.00 | 0.02 | 0.00 |
| 3 | 1.00 | 0.05 | 0.00 |
| 4 | 1.00 | 0.28 | 0.00 |
| 8 | 1.00 | 0.51 | 0.00 |

**Notes.**
- The first version of the scripted pursuer aimed straight at the evader. Pursuers collided with each other in almost every episode (about 2–8 crashes per episode). Adding a separation term fixed it. Learned policies will need to learn the same thing.
- Naive chasing loses to a faster evader in small teams. That leaves room for learned teamwork, but we'll need per-team-size balancing (see `docs/game_spec.md`, open question 3).

**Next.** Run the Crazyflow spike locally and on Empire AI, then record throughput here.
