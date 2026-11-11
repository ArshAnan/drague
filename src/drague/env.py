"""Batched tracking environment.

`reset` and `step` are pure functions of a JAX key and a state pytree, so a training
loop can wrap `step` in `jax.lax.scan`. A world that finishes is reported in `info`
and then respawned; the returned observation is the new episode.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from drague.config import GameConfig
from drague import rules
from drague.physics import point_mass

OUTCOME_NAMES = {
    1: "evader_crashed",
    2: "pursuers_wiped",
    3: "timeout_win",
    4: "timeout_loss",
}
_SEP_NONE = 1.0e6


class TrackingEnv:
    """Point-mass tracking game, batched over `n_worlds` arenas."""

    def __init__(self, cfg: GameConfig, backend: str = "point_mass", n_worlds: int = 1):
        if backend != "point_mass":
            raise NotImplementedError(
                f"backend {backend!r} is not implemented; use 'point_mass'"
            )
        if n_worlds < 1:
            raise ValueError("n_worlds must be positive")
        self.cfg = cfg
        self.backend = backend
        self.n_worlds = int(n_worlds)

    def reset(self, key):
        state = self._spawn(key)
        return state, self.observe(state)

    def observe(self, state):
        self_feat, others = rules.observations(
            state["pos"], state["vel"], state["alive"], self.cfg, jnp)
        return {"self": self_feat, "others": others}

    def step(self, key, state, actions):
        cfg = self.cfg
        P = cfg.n_pursuers
        pos, vel = point_mass.step(state["pos"], state["vel"], actions, state["alive"], cfg, jnp)
        alive_before = state["alive"]
        oob = rules.out_of_bounds(pos, cfg, jnp)
        col = rules.pursuer_collisions(pos, alive_before, cfg, jnp)
        alive, crashed = rules.update_alive(alive_before, pos, cfg, jnp)
        just_oob = alive_before & oob
        just_col = alive_before & col
        tracked_now = rules.is_tracked(pos, alive, cfg, jnp)

        steps = state["steps"] + 1
        tracked_steps = state["tracked_steps"] + tracked_now.astype(jnp.int32)
        first_track = jnp.where(
            (state["first_track"] < 0) & tracked_now, steps, state["first_track"])
        pursuer_oob = state["pursuer_oob"] + just_oob[:, :P].sum(axis=1).astype(jnp.int32)
        pursuer_col = state["pursuer_collision"] + just_col[:, :P].sum(axis=1).astype(jnp.int32)
        evader_oob = state["evader_oob"] + just_oob[:, cfg.evader_idx].astype(jnp.int32)

        d = rules.pursuer_evader_dist(pos, cfg, jnp)
        d_min = jnp.where(alive[:, :P], d, 2.0 * cfg.arena_radius).min(axis=1)
        d_min_sum = state["d_min_sum"] + d_min
        sep = _min_separation(pos, alive_before, cfg)
        min_sep = jnp.minimum(state["min_sep"], sep)

        speed = jnp.linalg.norm(vel, axis=-1)
        sat_p = state["sat_p"] + (
            (speed[:, :P] >= 0.99 * cfg.pursuer_max_speed) & alive_before[:, :P]
        ).sum(axis=1).astype(jnp.int32)
        live_p = state["live_p"] + alive_before[:, :P].sum(axis=1).astype(jnp.int32)
        sat_e = state["sat_e"] + (
            (speed[:, P] >= 0.99 * cfg.evader_max_speed) & alive_before[:, P]
        ).astype(jnp.int32)
        live_e = state["live_e"] + alive_before[:, P].astype(jnp.int32)

        done, pursuers_win = rules.resolve_outcome(alive, tracked_steps, steps, cfg, jnp)
        terms = rules.reward_terms(pos, alive, tracked_now, crashed, done, pursuers_win, cfg, jnp)
        ret_t = state["return_tracking"] + terms[0]
        ret_s = state["return_shaping"] + terms[1]
        ret_c = state["return_crash"] + terms[2]
        ret_term = state["return_terminal"] + terms[3]
        team = terms[0] + terms[1] + terms[2] + terms[3]
        is_pursuer = (jnp.arange(cfg.n_drones) < P)[None, :]
        rewards = jnp.where(is_pursuer, team[:, None], -team[:, None])

        evader_down = ~alive[:, cfg.evader_idx]
        pursuers_down = ~jnp.any(alive[:, :P], axis=1)
        timeout = steps >= cfg.max_steps
        won_on_time = (tracked_steps / cfg.max_steps) >= cfg.win_tracked_fraction
        outcome = jnp.zeros(self.n_worlds, dtype=jnp.int32)
        outcome = jnp.where(timeout, 4, outcome)
        outcome = jnp.where(timeout & won_on_time, 3, outcome)
        outcome = jnp.where(pursuers_down, 2, outcome)
        outcome = jnp.where(evader_down, 1, outcome)
        outcome = jnp.where(done, outcome, 0)

        length = jnp.maximum(steps, 1)
        info = {
            "pursuers_win": pursuers_win,
            "tracked_fraction": tracked_steps / cfg.max_steps,
            "crashes": crashed,
            "crashed_oob": just_oob,
            "crashed_collision": just_col,
            "outcome_code": outcome,
            "length_steps": steps,
            "tracked_steps": tracked_steps,
            "time_to_first_track_steps": first_track,
            "pursuer_oob": pursuer_oob,
            "pursuer_collision": pursuer_col,
            "evader_oob": evader_oob,
            "return_tracking": ret_t,
            "return_shaping": ret_s,
            "return_crash": ret_c,
            "return_terminal": ret_term,
            "return_total": ret_t + ret_s + ret_c + ret_term,
            "mean_d_min_m": d_min_sum / length,
            "final_d_min_m": d_min,
            "min_pursuer_separation_m": min_sep,
            "pursuer_speed_saturated": jnp.where(live_p > 0, sat_p / live_p, 0.0),
            "evader_speed_saturated": jnp.where(live_e > 0, sat_e / live_e, 0.0),
            "tracked": tracked_now,
            "pos": pos,
            "vel": vel,
            "alive": alive,
        }

        spawned = self._spawn(key)
        cont = {
            "pos": pos, "vel": vel, "alive": alive, "steps": steps,
            "tracked_steps": tracked_steps, "first_track": first_track,
            "pursuer_oob": pursuer_oob, "pursuer_collision": pursuer_col,
            "evader_oob": evader_oob, "return_tracking": ret_t, "return_shaping": ret_s,
            "return_crash": ret_c, "return_terminal": ret_term, "d_min_sum": d_min_sum,
            "min_sep": min_sep, "sat_p": sat_p, "live_p": live_p, "sat_e": sat_e, "live_e": live_e,
        }
        new_state = {
            name: jnp.where(done.reshape(done.shape + (1,) * (value.ndim - 1)), spawned[name], value)
            for name, value in cont.items()
        }
        return new_state, self.observe(new_state), rewards, done, info

    def _spawn(self, key):
        cfg = self.cfg
        W, P = self.n_worlds, cfg.n_pursuers
        k1, k2, k3 = jax.random.split(key, 3)
        pos = rules.spawn_positions(
            jax.random.uniform(k1, (W, 1)),
            jax.random.uniform(k2, (W, P, 3)),
            jax.random.uniform(k3, (W, 2)),
            cfg, jnp).astype(jnp.float32)
        vel = jnp.zeros_like(pos)
        alive = jnp.ones((W, cfg.n_drones), dtype=bool)
        zeros = jnp.zeros((W,), dtype=jnp.int32)
        return {
            "pos": pos,
            "vel": vel,
            "alive": alive,
            "steps": zeros,
            "tracked_steps": zeros,
            "first_track": jnp.full((W,), -1, dtype=jnp.int32),
            "pursuer_oob": zeros,
            "pursuer_collision": zeros,
            "evader_oob": zeros,
            "return_tracking": jnp.zeros((W,), dtype=jnp.float32),
            "return_shaping": jnp.zeros((W,), dtype=jnp.float32),
            "return_crash": jnp.zeros((W,), dtype=jnp.float32),
            "return_terminal": jnp.zeros((W,), dtype=jnp.float32),
            "d_min_sum": jnp.zeros((W,), dtype=jnp.float32),
            "min_sep": jnp.full((W,), _SEP_NONE, dtype=jnp.float32),
            "sat_p": zeros,
            "live_p": zeros,
            "sat_e": zeros,
            "live_e": zeros,
        }


def _min_separation(pos, alive, cfg: GameConfig):
    P = cfg.n_pursuers
    W = pos.shape[0]
    if P < 2:
        return jnp.full((W,), _SEP_NONE)
    dist = rules.pairwise_dist(pos[:, :P], jnp)
    dist = dist + jnp.eye(P) * _SEP_NONE
    dead = ~alive[:, :P]
    dist = jnp.where(dead[:, :, None] | dead[:, None, :], _SEP_NONE, dist)
    return dist.min(axis=(1, 2))


def finished_records(info, dones, steps_before: int = 0) -> list[dict]:
    """Host-side episode rows for worlds that finished on this step or rollout.

    `dones` is `(W,)` or `(T, W)`. Array fields of `info` gain a leading time axis
    when `dones` is two-dimensional. The caller still sets `episode_id` and `learner`.
    """
    dones = np.asarray(dones)
    if dones.ndim == 1:
        info = {key: np.asarray(value)[None] for key, value in info.items()}
        dones = dones[None]
    else:
        info = {key: np.asarray(value) for key, value in info.items()}
    n_worlds = dones.shape[-1]
    records = []
    for t, w in zip(*np.nonzero(dones)):
        record = _record_at(info, int(t), int(w))
        record["env_steps_at_end"] = steps_before + (int(t) + 1) * n_worlds
        records.append(record)
    return records


def _record_at(info, t: int, w: int) -> dict:
    first = int(info["time_to_first_track_steps"][t, w])
    sep = float(info["min_pursuer_separation_m"][t, w])
    return {
        "world": w,
        "outcome": OUTCOME_NAMES[int(info["outcome_code"][t, w])],
        "pursuers_win": bool(info["pursuers_win"][t, w]),
        "length_steps": int(info["length_steps"][t, w]),
        "tracked_steps": int(info["tracked_steps"][t, w]),
        "tracked_fraction": float(info["tracked_fraction"][t, w]),
        "time_to_first_track_steps": None if first < 0 else first,
        "pursuer_oob": int(info["pursuer_oob"][t, w]),
        "pursuer_collision": int(info["pursuer_collision"][t, w]),
        "evader_oob": int(info["evader_oob"][t, w]),
        "return_tracking": float(info["return_tracking"][t, w]),
        "return_shaping": float(info["return_shaping"][t, w]),
        "return_crash": float(info["return_crash"][t, w]),
        "return_terminal": float(info["return_terminal"][t, w]),
        "return_total": float(info["return_total"][t, w]),
        "mean_d_min_m": float(info["mean_d_min_m"][t, w]),
        "final_d_min_m": float(info["final_d_min_m"][t, w]),
        "min_pursuer_separation_m": None if sep >= 1.0e5 else sep,
        "pursuer_speed_saturated": float(info["pursuer_speed_saturated"][t, w]),
        "evader_speed_saturated": float(info["evader_speed_saturated"][t, w]),
    }
