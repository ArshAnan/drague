# drague

League training vs self-play vs prioritized fictitious self-play (PFSP) for multi-drone tracking teams,
compared at equal compute. Class research project.

**Research question.** Does league training with dedicated exploiter agents make multi-drone tracking
teams less exploitable, and better at generalizing, than plain self-play or PFSP at the same compute?
Exploitability is measured as the win rate of a fresh exploiter trained for a fixed budget against a
frozen team, alongside Elo and cross-play.

## Layout

```
src/drague/
  config.py            all game numbers (GameConfig)
  rules.py             spawn, tracking, crashes, outcome, rewards, observations (pure NumPy/JAX functions)
  physics/point_mass.py  simple drone model, for fast development and as a fallback
  scripted.py          hand-written pursuers/evaders: sanity checks and held-out opponents
tests/                 run on NumPy and JAX
scripts/spike_crazyflow.py   week-1 check of Crazyflow + throughput measurement
cluster/spike.sbatch         Empire AI (Slurm) job for the spike
docs/game_spec.md            the game rules and the env API both of us code against
EXPERIMENTS.md               running log of what we ran and what we learned
```

Coming next:
- Env side: `TrackingEnv` (the batched reset/step wrapper) and the Crazyflow physics backend.
- Training side: `algos/`, `league/` and `eval/`.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev                # laptop (CPU JAX; works on macOS)
uv sync --extra gpu --extra dev    # Linux + CUDA 12 (Empire AI)
uv run pytest -q                   # all tests should pass
```

## Week-1 spike

```bash
uv run python scripts/spike_crazyflow.py            # laptop, quick
sbatch cluster/spike.sbatch                         # Empire AI, full sweep (fill in partition/account first)
```

The spike writes `results/spike_<host>_<time>.json`. Copy the key numbers into `EXPERIMENTS.md`.

## Conventions

- Drone index `0..P-1` = pursuers, index `P` = evader.
- Everything is batched over a leading world axis `W`.
- Game numbers only live in `GameConfig`. Experiments differ by config, not by code edits.
- Count compute as **env steps = worlds × policy steps**, including exploiter training.
