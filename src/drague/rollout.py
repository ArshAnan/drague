"""Scripted matches on the point-mass backend, written through a RunLog.

Step order matches the sanity-check loop in tests/test_scripted.py: act, integrate, apply
crashes, then score. Finished worlds freeze, so later batched steps do not add reward or frames.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from drague.config import GameConfig
from drague import rules, scripted
from drague.logbook import LOSS_OUTCOMES, WIN_OUTCOMES, RunLog
from drague.physics import point_mass

# CLI name -> (name stored in the manifest, policy function).
SCRIPTED = {
    "standoff": ("standoff_pursuers", scripted.standoff_pursuers),
    "flee": ("flee_evader", scripted.flee_evader),
    "idle": ("idle", scripted.idle),
}


def outcome_labels(evader_down, pursuers_down, timeout, won_on_time) -> np.ndarray:
    """Outcome string per world, using the same priority as `rules.resolve_outcome`.

    evader crashed, then every pursuer crashed, then timeout. Empty where the episode
    is still going.
    """
    evader_down = np.asarray(evader_down, dtype=bool)
    pursuers_down = np.asarray(pursuers_down, dtype=bool)
    timeout = np.asarray(timeout, dtype=bool)
    won_on_time = np.asarray(won_on_time, dtype=bool)
    names = np.full(evader_down.shape, "", dtype=object)
    names[timeout] = "timeout_loss"
    names[timeout & won_on_time] = "timeout_win"
    names[pursuers_down] = "pursuers_wiped"
    names[evader_down] = "evader_crashed"
    return names


def _scripted(name: str):
    try:
        return SCRIPTED[name]
    except KeyError:
        known = ", ".join(SCRIPTED)
        raise KeyError(f"unknown scripted policy {name!r}; choose from {known}") from None


def _separation(pos, alive, n_pursuers: int) -> np.ndarray:
    """Min distance between two live pursuers. Inf where fewer than two are alive. (W,)"""
    if n_pursuers < 2:
        return np.full(pos.shape[0], np.inf)
    dist = np.asarray(rules.pairwise_dist(pos[:, :n_pursuers], np))
    dist = dist + np.eye(n_pursuers) * 1e6
    dead = ~alive[:, :n_pursuers]
    dist = np.where(dead[:, :, None] | dead[:, None, :], 1e6, dist)
    return dist.min(axis=(1, 2))


def _play_batch(cfg: GameConfig, pursuer_fn, evader_fn, n_worlds: int, rng: np.random.Generator,
                sample_worlds: list[int]):
    """Play `n_worlds` episodes. Returns episode dicts (without ids) and trajectory arrays."""
    P = cfg.n_pursuers
    D = cfg.n_drones
    W = n_worlds
    max_steps = cfg.max_steps
    u_phase = rng.random((W, 1))
    u_jitter = rng.random((W, P, 3))
    u_evader = rng.random((W, 2))
    pos = np.asarray(rules.spawn_positions(u_phase, u_jitter, u_evader, cfg, np), dtype=np.float64)
    vel = np.zeros_like(pos)
    alive = np.ones((W, D), dtype=bool)
    is_pursuer = (np.arange(D) < P)[None, :, None]

    tracked_steps = np.zeros(W, np.int32)
    length = np.zeros(W, np.int32)
    first_track = np.full(W, -1, np.int32)
    n_pursuer_oob = np.zeros(W, np.int32)
    n_pursuer_col = np.zeros(W, np.int32)
    n_evader_oob = np.zeros(W, np.int32)
    returns = np.zeros((W, 4), np.float64)
    returns_check = np.zeros(W, np.float64)
    d_min_sum = np.zeros(W, np.float64)
    final_d_min = np.full(W, np.nan)
    min_sep = np.full(W, np.inf)
    sat_p = np.zeros(W, np.int32)
    live_p = np.zeros(W, np.int32)
    sat_e = np.zeros(W, np.int32)
    live_e = np.zeros(W, np.int32)
    done = np.zeros(W, dtype=bool)
    win = np.zeros(W, dtype=bool)
    outcome = np.full(W, "", dtype=object)

    frames = max_steps + 1
    n_sample = len(sample_worlds)
    slot = {world: i for i, world in enumerate(sample_worlds)}
    pos_buf = np.zeros((n_sample, frames, D, 3), np.float32)
    vel_buf = np.zeros((n_sample, frames, D, 3), np.float32)
    alive_buf = np.ones((n_sample, frames, D), dtype=bool)
    action_buf = np.zeros((n_sample, frames, D, 3), np.float32)
    tracked_buf = np.zeros((n_sample, frames), dtype=bool)
    oob_buf = np.zeros((n_sample, frames, D), dtype=bool)
    col_buf = np.zeros((n_sample, frames, D), dtype=bool)
    length_frames = np.ones(n_sample, np.int32)

    def capture(frame: int, live_mask, tracked_now, just_oob, just_col):
        for world, i in slot.items():
            if not live_mask[world]:
                continue
            pos_buf[i, frame] = pos[world]
            vel_buf[i, frame] = vel[world]
            alive_buf[i, frame] = alive[world]
            if tracked_now is not None:
                tracked_buf[i, frame] = tracked_now[world]
                oob_buf[i, frame] = just_oob[world]
                col_buf[i, frame] = just_col[world]
            length_frames[i] = frame + 1

    spawn_tracked = np.asarray(rules.is_tracked(pos, alive, cfg, np))
    capture(0, np.ones(W, dtype=bool), spawn_tracked, np.zeros((W, D), dtype=bool),
            np.zeros((W, D), dtype=bool))

    for step in range(max_steps):
        if done.all():
            break
        if done.any():
            alive = np.where(done[:, None], False, alive)
            vel = np.where(done[:, None, None], 0.0, vel)
        live = ~done

        action = np.where(
            is_pursuer,
            pursuer_fn(pos, vel, alive, cfg, np),
            evader_fn(pos, vel, alive, cfg, np),
        )
        for world, i in slot.items():
            if live[world]:
                action_buf[i, step] = action[world]

        pos, vel = point_mass.step(pos, vel, action, alive, cfg, np)
        pos = np.asarray(pos)
        vel = np.asarray(vel)
        alive_before = alive
        oob = np.asarray(rules.out_of_bounds(pos, cfg, np))
        col = np.asarray(rules.pursuer_collisions(pos, alive_before, cfg, np))
        alive, crashed = rules.update_alive(alive_before, pos, cfg, np)
        alive = np.asarray(alive)
        crashed = np.asarray(crashed)
        just_oob = alive_before & oob
        just_col = alive_before & col
        tracked_now = np.asarray(rules.is_tracked(pos, alive, cfg, np))

        length[live] += 1
        tracked_steps[live & tracked_now] += 1
        newly_tracked = live & tracked_now & (first_track < 0)
        first_track[newly_tracked] = length[newly_tracked]
        n_pursuer_oob += (just_oob[:, :P] & live[:, None]).sum(axis=1)
        n_pursuer_col += (just_col[:, :P] & live[:, None]).sum(axis=1)
        n_evader_oob += (just_oob[:, cfg.evader_idx] & live).astype(np.int32)

        d = np.asarray(rules.pursuer_evader_dist(pos, cfg, np))
        d_min = np.where(alive[:, :P], d, 2.0 * cfg.arena_radius).min(axis=1)
        d_min_sum[live] += d_min[live]
        final_d_min[live] = d_min[live]
        sep = _separation(pos, alive_before, P)
        have_pair = live & np.isfinite(sep) & (sep < 1e5)
        min_sep[have_pair] = np.minimum(min_sep[have_pair], sep[have_pair])

        speed = np.linalg.norm(vel, axis=-1)
        sat_p += ((speed[:, :P] >= 0.99 * cfg.pursuer_max_speed) & alive_before[:, :P] & live[:, None]).sum(axis=1)
        live_p += (alive_before[:, :P] & live[:, None]).sum(axis=1)
        sat_e += ((speed[:, P] >= 0.99 * cfg.evader_max_speed) & alive_before[:, P] & live).astype(np.int32)
        live_e += (alive_before[:, P] & live).astype(np.int32)

        done_now, win_now = rules.resolve_outcome(
            alive, tracked_steps, length, cfg, np)
        done_now = np.asarray(done_now, dtype=bool)
        win_now = np.asarray(win_now, dtype=bool)
        newly = live & done_now
        evader_down = ~alive[:, cfg.evader_idx]
        pursuers_down = ~np.any(alive[:, :P], axis=1)
        timeout = length >= cfg.max_steps
        won_on_time = (tracked_steps / cfg.max_steps) >= cfg.win_tracked_fraction
        labels = outcome_labels(evader_down, pursuers_down, timeout, won_on_time)

        terms = rules.reward_terms(
            pos, alive, tracked_now, crashed, newly, win_now, cfg, np)
        for i, term in enumerate(terms):
            returns[live, i] += np.asarray(term, dtype=np.float64)[live]
        team, _ = rules.team_rewards(
            pos, alive, tracked_now, crashed, newly, win_now, cfg, np)
        returns_check[live] += np.asarray(team, dtype=np.float64)[live]

        capture(step + 1, live, tracked_now, just_oob, just_col)
        win = np.where(newly, win_now, win)
        outcome[newly] = labels[newly]
        done |= newly

    if not np.allclose(returns.sum(axis=1), returns_check):
        raise RuntimeError("reward terms do not sum to team_rewards")

    for i in range(n_sample):
        last = int(length_frames[i])
        if last < frames:
            pos_buf[i, last:] = pos_buf[i, last - 1]
            vel_buf[i, last:] = vel_buf[i, last - 1]
            alive_buf[i, last:] = alive_buf[i, last - 1]
            tracked_buf[i, last:] = tracked_buf[i, last - 1]

    episodes = []
    for w in range(W):
        sep_value = None if not np.isfinite(min_sep[w]) or min_sep[w] >= 1e5 else float(min_sep[w])
        episodes.append({
            "world": w,
            "outcome": str(outcome[w]),
            "pursuers_win": bool(win[w]),
            "length_steps": int(length[w]),
            "tracked_steps": int(tracked_steps[w]),
            "tracked_fraction": float(tracked_steps[w] / max_steps),
            "time_to_first_track_steps": None if first_track[w] < 0 else int(first_track[w]),
            "pursuer_oob": int(n_pursuer_oob[w]),
            "pursuer_collision": int(n_pursuer_col[w]),
            "evader_oob": int(n_evader_oob[w]),
            "return_tracking": float(returns[w, 0]),
            "return_shaping": float(returns[w, 1]),
            "return_crash": float(returns[w, 2]),
            "return_terminal": float(returns[w, 3]),
            "return_total": float(returns[w].sum()),
            "mean_d_min_m": float(d_min_sum[w] / length[w]) if length[w] else None,
            "final_d_min_m": None if np.isnan(final_d_min[w]) else float(final_d_min[w]),
            "min_pursuer_separation_m": sep_value,
            "pursuer_speed_saturated": float(sat_p[w] / live_p[w]) if live_p[w] else 0.0,
            "evader_speed_saturated": float(sat_e[w] / live_e[w]) if live_e[w] else 0.0,
        })
        if episodes[-1]["pursuers_win"] and episodes[-1]["outcome"] not in WIN_OUTCOMES:
            raise RuntimeError(f"win flag disagrees with outcome {episodes[-1]['outcome']}")
        if not episodes[-1]["pursuers_win"] and episodes[-1]["outcome"] not in LOSS_OUTCOMES:
            raise RuntimeError(f"loss flag disagrees with outcome {episodes[-1]['outcome']}")

    trajectories = {
        "pos": pos_buf,
        "vel": vel_buf,
        "alive": alive_buf,
        "action": action_buf,
        "tracked": tracked_buf,
        "crashed_oob": oob_buf,
        "crashed_collision": col_buf,
        "length_frames": length_frames,
        "worlds": np.asarray(sample_worlds, dtype=np.int32),
    }
    return episodes, trajectories


def record_scripted_run(
    *,
    pursuer: str,
    evader: str,
    n_pursuers: int = 3,
    seed: int = 0,
    n_episodes: int = 1024,
    n_worlds: int = 256,
    metrics_every: int = 64,
    n_trajectories: int = 8,
    video: str = "first",
    results_dir: Path | str = "results",
    cfg: GameConfig | None = None,
    command: str | None = None,
    max_video_frames: int | None = None,
) -> Path:
    """Play `n_episodes` and write a run directory. Returns that directory.

    `video` is `first` (one clip), `all` (every saved trajectory), or `none`.
    """
    if video not in {"first", "all", "none"}:
        raise ValueError("video must be 'first', 'all', or 'none'")
    if n_episodes < 1:
        raise ValueError("need at least one episode")
    pursuer_name, pursuer_fn = _scripted(pursuer)
    evader_name, evader_fn = _scripted(evader)
    cfg = cfg or GameConfig(n_pursuers=n_pursuers)
    n_worlds = max(1, min(n_worlds, n_episodes))
    metrics_every = max(1, metrics_every)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()

    with RunLog.open(
        results_dir,
        method="scripted",
        seed=seed,
        cfg=cfg,
        pursuer_policy=pursuer_name,
        evader_policy=evader_name,
        pursuer_label=pursuer,
        evader_label=evader,
        backend="point_mass",
        n_worlds=n_worlds,
        n_episodes=n_episodes,
        hparams={},
        command=command,
    ) as log:
        remaining_traj = max(0, n_trajectories)
        episode_id = 0
        env_steps = 0
        interval: list[dict] = []
        traj_pos = []
        traj_meta_ids = []
        traj_meta_worlds = []
        traj_meta_outcomes = []
        traj_meta_wins = []
        traj_meta_steps = []
        pieces = {key: [] for key in
                  ("pos", "vel", "alive", "action", "tracked", "crashed_oob", "crashed_collision",
                   "length_frames")}

        while episode_id < n_episodes:
            batch = min(n_worlds, n_episodes - episode_id)
            sample = list(range(min(remaining_traj, batch)))
            episodes, trajectories = _play_batch(cfg, pursuer_fn, evader_fn, batch, rng, sample)
            if sample:
                for key in pieces:
                    pieces[key].append(trajectories[key])
                remaining_traj -= len(sample)
            for record in episodes:
                env_steps += record["length_steps"]
                record["episode_id"] = episode_id
                record["env_steps_at_end"] = env_steps
                log.log_episode(record)
                interval.append(record)
                if sample and record["world"] in sample:
                    traj_meta_ids.append(episode_id)
                    traj_meta_worlds.append(record["world"])
                    traj_meta_outcomes.append(record["outcome"])
                    traj_meta_wins.append(record["pursuers_win"])
                    traj_meta_steps.append(record["length_steps"])
                episode_id += 1
                if episode_id % metrics_every == 0:
                    wall = time.perf_counter() - started
                    log.log_metrics(log.metrics_record(
                        interval, env_steps=env_steps, episodes_so_far=episode_id, wall_s=wall))
                    interval = []

        if interval:
            wall = time.perf_counter() - started
            log.log_metrics(log.metrics_record(
                interval, env_steps=env_steps, episodes_so_far=episode_id, wall_s=wall))

        if pieces["pos"]:
            arrays = {key: np.concatenate(chunks, axis=0) for key, chunks in pieces.items()}
            # length_frames was concatenated; the meta list is filled below from the array
            # so it stays aligned even if a batch boundary splits the sample.
            frames = arrays["length_frames"]
            meta = {
                "step": 0,
                "episode_ids": traj_meta_ids,
                "worlds": traj_meta_worlds,
                "outcomes": traj_meta_outcomes,
                "pursuers_win": traj_meta_wins,
                "length_frames": frames.astype(int).tolist(),
                "length_steps": traj_meta_steps,
            }
            if len(meta["episode_ids"]) != len(frames):
                raise RuntimeError("trajectory metadata is misaligned with the saved arrays")
            log.log_trajectory(arrays, meta, step=0)
            traj_pos.append(arrays)

        counts: dict[str, int] = {}
        for ep in log.episodes:
            counts[ep["outcome"]] = counts.get(ep["outcome"], 0) + 1
        preview = log.summary()
        log.log_eval({
            "step": 0,
            "kind": "scripted",
            "n_episodes": preview["n_episodes"],
            "pursuer_policy": pursuer_name,
            "evader_policy": evader_name,
            "pursuer_win_rate": preview["pursuer_win_rate"],
            "tracked_fraction": preview["tracked_fraction"],
            "outcome_counts": counts,
            "episodes_file": "episodes.jsonl",
        }, step=0)

        if video != "none" and traj_pos:
            _write_videos(log, video, max_video_frames)

    return log.run_dir


def _write_videos(log: RunLog, video: str, max_frames: int | None) -> None:
    try:
        from drague.video import render_run
    except ImportError as exc:
        print(f"skipping video: {exc}", file=sys.stderr)
        return
    indices = None if video == "all" else [0]
    for path in render_run(log.run_dir, indices=indices, max_frames=max_frames):
        log.note_video(path)
