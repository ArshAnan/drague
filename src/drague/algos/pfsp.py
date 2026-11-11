"""Prioritized fictitious self-play: sample opponents the learner does not yet beat."""

from __future__ import annotations

import numpy as np


def prioritized_weights(win_rates, alpha: float = 1.0, uniform: float = 0.1) -> np.ndarray:
    """Probability of each opponent. `win_rates` is the learner's win rate against them.

    A win rate of 0 (never beaten) gets more mass than a win rate of 1. `uniform` keeps
    every checkpoint in the mix.
    """
    win_rates = np.asarray(win_rates, dtype=np.float64)
    if win_rates.size == 0:
        return win_rates
    weights = np.clip(1.0 - win_rates, 0.0, None) ** alpha
    if float(weights.sum()) <= 0.0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    weights = (1.0 - uniform) * weights + uniform / weights.size
    return weights


class CheckpointPool:
    """A capped list of past networks and the learner's win rate against each."""

    def __init__(self, cap: int = 32):
        self.cap = cap
        self.entries: list[dict] = []
        self._next = 0

    def add(self, params, win_rate: float = 0.5) -> dict:
        entry = {"id": self._next, "params": params, "win_rate": float(win_rate)}
        self._next += 1
        self.entries.append(entry)
        if len(self.entries) > self.cap:
            self.entries.pop(0)
        return entry

    def sample(self, rng: np.random.Generator, alpha: float = 1.0, uniform: float = 0.1):
        if not self.entries:
            return None
        weights = prioritized_weights([e["win_rate"] for e in self.entries], alpha, uniform)
        index = int(rng.choice(len(self.entries), p=weights))
        return self.entries[index]

    def update_win(self, entry_id: int, win_rate: float, beta: float = 0.2) -> None:
        for entry in self.entries:
            if entry["id"] == entry_id:
                entry["win_rate"] = (1.0 - beta) * entry["win_rate"] + beta * float(win_rate)
                return
