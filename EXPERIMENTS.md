# Experiment log

Newest entries at the top. Each entry: date, who, commit, what was run, result, decision.

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
