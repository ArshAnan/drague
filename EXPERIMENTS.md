# Experiment log

Newest entries at the top. Each entry: date, who, commit, what was run, result, decision.

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
