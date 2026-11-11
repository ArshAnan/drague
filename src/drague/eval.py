"""Held-out matches, a fresh exploiter's win rate, and cross-play Elo."""

from __future__ import annotations

from collections import Counter

import numpy as np

from drague.env import finished_records


def bradley_terry(win_rate: np.ndarray, steps: int = 200, lr: float = 0.05):
    """Pursuer and evader Elo from a win-rate matrix. `win_rate[i, j]` is pursuer i vs evader j."""
    win_rate = np.asarray(win_rate, dtype=np.float64)
    n_p, n_e = win_rate.shape
    pursuer = np.zeros(n_p)
    evader = np.zeros(n_e)
    for _ in range(steps):
        for i in range(n_p):
            for j in range(n_e):
                expected = 1.0 / (1.0 + 10 ** ((evader[j] - pursuer[i]) / 400.0))
                pursuer[i] += lr * (win_rate[i, j] - expected)
                evader[j] += lr * ((1.0 - win_rate[i, j]) - (1.0 - expected))
    return pursuer, evader


def play_matches(env, key, action_fn, n_episodes: int, horizon: int, step_fn=None):
    """Roll until `n_episodes` worlds have finished. `action_fn(key, state)` returns actions."""
    import jax

    step_fn = step_fn or env.step
    wins: list[bool] = []
    tracked: list[float] = []
    outcomes: Counter = Counter()
    while len(wins) < n_episodes:
        key, k_reset = jax.random.split(key)
        state, _ = env.reset(k_reset)
        for _ in range(horizon):
            key, k_act, k_step = jax.random.split(key, 3)
            actions = action_fn(k_act, state)
            state, _, _, dones, info = step_fn(k_step, state, actions)
            if bool(np.asarray(dones).any()):
                for record in finished_records(info, dones):
                    if len(wins) >= n_episodes:
                        break
                    wins.append(record["pursuers_win"])
                    tracked.append(record["tracked_fraction"])
                    outcomes[record["outcome"]] += 1
            if len(wins) >= n_episodes:
                break
    return {
        "n_episodes": len(wins),
        "pursuer_win_rate": float(np.mean(wins)) if wins else 0.0,
        "tracked_fraction": float(np.mean(tracked)) if tracked else 0.0,
        "outcome_counts": dict(outcomes),
    }


def cross_play(env, key, pursuers: list, evaders: list, n_episodes: int, horizon: int):
    """`pursuers` and `evaders` are `(name, action_fn)` pairs.

    Each action_fn returns a full `(W, D, 3)` action with the other role's rows ignored
    by the caller. `combine(pursuer_fn, evader_fn)` is applied here: pursuer rows from the
    pursuer function, the evader row from the evader function.
    """
    import jax

    P = env.cfg.n_pursuers
    matrix = np.zeros((len(pursuers), len(evaders)))
    tracked = np.zeros_like(matrix)
    for i, (pname, p_fn) in enumerate(pursuers):
        for j, (ename, e_fn) in enumerate(evaders):
            def act(key, state, p_fn=p_fn, e_fn=e_fn):
                k1, k2 = jax.random.split(key)
                pursuer = p_fn(k1, state)
                evader = e_fn(k2, state)
                return pursuer.at[:, P:].set(evader[:, P:])

            result = play_matches(env, key, act, n_episodes, horizon)
            key, _ = jax.random.split(key)
            matrix[i, j] = result["pursuer_win_rate"]
            tracked[i, j] = result["tracked_fraction"]
    p_elo, e_elo = bradley_terry(matrix)
    return {
        "pursuers": [name for name, _ in pursuers],
        "evaders": [name for name, _ in evaders],
        "pursuer_win_rate": matrix.tolist(),
        "tracked_fraction": tracked.tolist(),
        "pursuer_elo": p_elo.tolist(),
        "evader_elo": e_elo.tolist(),
    }
