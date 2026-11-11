"""Self-play, PFSP, and league training on TrackingEnv.

One call is one paper cell: one strategy, one team size, one env-step budget, one seed.
Every step is written through RunLog, and a top-down clip is rendered from the saved trajectory.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from drague.algos.mappo import (
    EPOCHS,
    LEARNING_RATE,
    collect_rollout,
    deterministic_role_actions,
    init_role,
    make_optimizer,
    update_evader,
    update_pursuer,
)
from drague.algos.pfsp import CheckpointPool
from drague.algos.policy import ActorCritic
from drague.config import GameConfig
from drague.env import OUTCOME_NAMES, TrackingEnv, finished_records
from drague.eval import play_matches
from drague.logbook import RunLog
from drague.rollout import SCRIPTED
from drague.video import render_episode

SMOKE_STEPS = 4096
PAPER_STEPS = 2_000_000
_INFO_KEYS = (
    "pursuers_win", "tracked_fraction", "outcome_code", "length_steps", "tracked_steps",
    "time_to_first_track_steps", "pursuer_oob", "pursuer_collision", "evader_oob",
    "return_tracking", "return_shaping", "return_crash", "return_terminal", "return_total",
    "mean_d_min_m", "final_d_min_m", "min_pursuer_separation_m",
    "pursuer_speed_saturated", "evader_speed_saturated",
)


def resolve_budget(budget, n_worlds: int | None) -> tuple[int, int]:
    if budget == "smoke":
        steps, worlds = SMOKE_STEPS, 8
    elif budget == "paper":
        steps, worlds = PAPER_STEPS, 32
    else:
        steps, worlds = int(budget), 8
    if n_worlds is not None:
        worlds = int(n_worlds)
    steps = steps - (steps % worlds)
    if steps < worlds:
        raise ValueError("budget is smaller than one batched step")
    return steps, worlds


def train(
    *,
    method: str,
    n_pursuers: int = 3,
    seed: int = 0,
    budget: int | str = "smoke",
    n_worlds: int | None = None,
    video: str = "first",
    results_dir: str | Path = "results",
    cfg: GameConfig | None = None,
    exploiter_fraction: float = 0.2,
    measure_exploitability: bool | None = None,
    fresh_fraction: float = 0.1,
    fresh_budget: int | None = None,
    command: str | None = None,
    hidden: int = 64,
    eval_episodes: int = 16,
    pool_cap: int = 32,
) -> Path:
    if method not in {"selfplay", "pfsp", "league"}:
        raise ValueError(f"unknown method {method!r}")
    if video not in {"first", "all", "none"}:
        raise ValueError(f"unknown video mode {video!r}")
    budget_steps, n_worlds = resolve_budget(budget, n_worlds)
    cfg = cfg or GameConfig(n_pursuers=n_pursuers)
    if measure_exploitability is None:
        measure_exploitability = budget == "paper"
    fresh_steps = fresh_budget if fresh_budget is not None else max(
        n_worlds, int(round((PAPER_STEPS if budget == "paper" else budget_steps) * fresh_fraction)))
    fresh_steps -= fresh_steps % n_worlds

    env = TrackingEnv(cfg, n_worlds=n_worlds)
    net = ActorCritic(hidden=hidden)
    optimizer = make_optimizer()
    key = jax.random.PRNGKey(seed)
    key, k_reset, k_p, k_e = jax.random.split(key, 4)
    state, obs = env.reset(k_reset)
    P = cfg.n_pursuers
    p_params, p_opt = init_role(k_p, net, obs["self"][:, :P], obs["others"][:, :P])
    e_params, e_opt = init_role(k_e, net, obs["self"][:, P:P + 1], obs["others"][:, P:P + 1])
    exploiter_params, exploiter_opt = e_params, e_opt

    exp_budget = int(round(budget_steps * exploiter_fraction)) if method == "league" else 0
    exp_budget -= exp_budget % n_worlds
    main_budget = budget_steps - exp_budget
    window = max(exp_budget // 4, n_worlds * cfg.max_steps) if exp_budget else n_worlds * cfg.max_steps

    pool_p, pool_e = CheckpointPool(pool_cap), CheckpointPool(pool_cap)
    pool_p.add(p_params, 0.5)
    pool_e.add(e_params, 0.5)
    rng = np.random.default_rng(seed)
    main_steps = exp_steps = 0
    episode_id = 0
    last_reset = 0
    last_save = 0
    save_every = n_worlds * cfg.max_steps
    started = time.perf_counter()
    hparams = {
        "budget_env_steps": budget_steps,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "hidden": hidden,
        "exploiter_fraction": exploiter_fraction if method == "league" else 0.0,
        "pool_cap": pool_cap,
        "save_every": save_every,
        "rollout_steps": cfg.max_steps,
    }

    with RunLog.open(
        results_dir,
        method=method,
        seed=seed,
        cfg=cfg,
        pursuer_policy="mappo",
        evader_policy="mappo",
        pursuer_label="main",
        evader_label="main",
        backend="point_mass",
        n_worlds=n_worlds,
        n_episodes=0,
        hparams=hparams,
        command=command,
    ) as log:
        while main_steps + exp_steps < budget_steps:
            phase = _phase(method, main_steps, exp_steps, main_budget, exp_budget)
            horizon = _horizon(phase, main_steps, exp_steps, main_budget, exp_budget,
                               budget_steps, n_worlds, cfg.max_steps)
            if horizon < 1 and method == "league":
                phase = "main" if phase == "exploiter" else "exploiter"
                horizon = _horizon(phase, main_steps, exp_steps, main_budget, exp_budget,
                                   budget_steps, n_worlds, cfg.max_steps)
            if horizon < 1:
                break
            opponent_id = "latest"
            learner = "main"
            update_p = update_e = False
            update_e_params, update_e_opt = e_params, e_opt
            sampled = None
            if phase == "exploiter":
                if exp_steps == last_reset:
                    key, k_init = jax.random.split(key)
                    obs_now = env.observe(state)
                    exploiter_params, exploiter_opt = init_role(
                        k_init, net, obs_now["self"][:, P:P + 1], obs_now["others"][:, P:P + 1])
                evader_used = exploiter_params
                pursuer_used = p_params
                update_e_params = exploiter_params
                update_e_opt = exploiter_opt
                update_e = True
                learner = "exploiter"
                opponent_id = "frozen_pursuer"
            elif method == "selfplay":
                pursuer_used, evader_used = p_params, e_params
                update_p = update_e = True
                update_e_params, update_e_opt = e_params, e_opt
            elif method == "pfsp":
                learn_pursuer = ((main_steps // (horizon * n_worlds)) % 2) == 0
                if learn_pursuer:
                    entry = pool_e.sample(rng)
                    evader_used = entry["params"] if entry is not None else e_params
                    pursuer_used = p_params
                    update_p = True
                    sampled = (pool_e, entry)
                    opponent_id = f"evader_{entry['id']}" if entry is not None else "latest"
                else:
                    entry = pool_p.sample(rng)
                    pursuer_used = entry["params"] if entry is not None else p_params
                    evader_used = e_params
                    update_e = True
                    update_e_params, update_e_opt = e_params, e_opt
                    sampled = (pool_p, entry)
                    opponent_id = f"pursuer_{entry['id']}" if entry is not None else "latest"
            else:
                # League main: alternate the pursuer (vs PFSP or the exploiter) and the main evader.
                learn_pursuer = ((main_steps // max(horizon * n_worlds, 1)) % 2) == 0
                if learn_pursuer:
                    if rng.random() < 0.5:
                        evader_used = exploiter_params
                        opponent_id = "exploiter"
                    else:
                        entry = pool_e.sample(rng)
                        evader_used = entry["params"] if entry is not None else e_params
                        sampled = (pool_e, entry)
                        opponent_id = f"evader_{entry['id']}" if entry is not None else "latest"
                    pursuer_used = p_params
                    update_p = True
                else:
                    entry = pool_p.sample(rng)
                    pursuer_used = entry["params"] if entry is not None else p_params
                    evader_used = e_params
                    update_e = True
                    update_e_params, update_e_opt = e_params, e_opt
                    sampled = (pool_p, entry)
                    opponent_id = f"pursuer_{entry['id']}" if entry is not None else "latest"

            key, state, frames, p_boot, e_boot = collect_rollout(
                key, state, pursuer_used, evader_used, horizon, env, net)
            metrics = {}
            if update_p:
                p_params, p_opt, metrics = update_pursuer(
                    p_params, p_opt, optimizer, frames, p_boot, net, EPOCHS)
            if update_e:
                new_e, new_opt, e_metrics = update_evader(
                    update_e_params, update_e_opt, optimizer, frames, e_boot, net, EPOCHS)
                if learner == "exploiter":
                    exploiter_params, exploiter_opt = new_e, new_opt
                else:
                    e_params, e_opt = new_e, new_opt
                if not metrics:
                    metrics = e_metrics
            taken_before = main_steps + exp_steps
            taken = horizon * n_worlds
            if learner == "exploiter":
                exp_steps += taken
                if exp_steps - last_reset >= window:
                    last_reset = exp_steps
            else:
                main_steps += taken
            records = _log_frames(log, frames, taken_before, learner, episode_id)
            episode_id += len(records)
            _log_metrics(log, records, metrics, main_steps + exp_steps, episode_id, started, opponent_id)
            if learner == "main" and method != "selfplay":
                _record_match(sampled, records)
                if main_steps - last_save >= save_every:
                    pool_p.add(p_params, 0.5)
                    pool_e.add(e_params, 0.5)
                    last_save = main_steps

        log.manifest["steps_taken_main"] = main_steps
        log.manifest["steps_taken_exploiter"] = exp_steps
        ckpt = _save_checkpoints(log.run_dir, p_params, e_params, exploiter_params if method == "league" else None,
                                 hidden, cfg.n_pursuers, method, seed)
        log.manifest["hparams"]["checkpoints"] = ckpt

        step_fn = jax.jit(env.step)
        key = _evaluate_held_out(
            env, key, net, p_params, log, step_fn, eval_episodes, video, cfg)
        _render_match(env, key, net, p_params, e_params, log, step_fn, "learned", video)
        if measure_exploitability and fresh_steps >= n_worlds:
            key = _fresh_exploiter(
                env, key, net, p_params, optimizer, fresh_steps, log, step_fn, eval_episodes, hidden)

    return log.run_dir


def _horizon(phase, main_steps, exp_steps, main_budget, exp_budget, budget_steps, n_worlds, max_steps):
    remaining_total = budget_steps - main_steps - exp_steps
    remaining_phase = (exp_budget - exp_steps) if phase == "exploiter" else (main_budget - main_steps)
    return min(max_steps, remaining_total // n_worlds, max(remaining_phase, 0) // n_worlds)


def _phase(method, main_steps, exp_steps, main_budget, exp_budget) -> str:
    if method != "league" or exp_budget <= 0:
        return "main"
    if main_steps == 0 and exp_steps == 0:
        return "main"
    if exp_steps >= exp_budget:
        return "main"
    if main_steps >= main_budget:
        return "exploiter"
    main_frac = main_steps / main_budget if main_budget else 1.0
    exp_frac = exp_steps / exp_budget
    return "exploiter" if exp_frac <= main_frac else "main"


def _log_frames(log: RunLog, frames, steps_before: int, learner: str, episode_id: int) -> list[dict]:
    subset = {key: frames[key] for key in _INFO_KEYS}
    records = finished_records(subset, frames["dones"], steps_before=steps_before)
    for record in records:
        record["episode_id"] = episode_id
        record["learner"] = learner
        log.log_episode(record)
        episode_id += 1
    return records


def _log_metrics(log, records, metrics, env_steps, episode_id, started, opponent_id):
    cited = [record for record in records if record.get("learner") != "exploiter"] or records
    wall = time.perf_counter() - started
    row = log.metrics_record(cited, env_steps=env_steps, episodes_so_far=episode_id, wall_s=wall)
    for key in ("policy_loss", "value_loss", "entropy"):
        if key in metrics:
            row[key] = float(metrics[key])
    row["opponent_id"] = opponent_id
    log.log_metrics(row)


def _record_match(sampled, records):
    """Fold this rollout's pursuer win rate into the checkpoint that was played."""
    if sampled is None or sampled[1] is None or not records:
        return
    pool, entry = sampled
    wins = [1.0 if record["pursuers_win"] else 0.0 for record in records]
    pool.update_win(entry["id"], float(np.mean(wins)))


def _save_checkpoints(run_dir, p_params, e_params, exploiter_params, hidden, n_pursuers, method, seed):
    folder = Path(run_dir) / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, params in (("pursuer", p_params), ("evader", e_params), ("exploiter", exploiter_params)):
        if params is None:
            continue
        path = folder / f"{name}.msgpack"
        path.write_bytes(serialization.to_bytes(params))
        paths[name] = str(path)
    meta = {"hidden": hidden, "n_pursuers": n_pursuers, "method": method, "seed": seed}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return paths


def _learned_actions(env, net, pursuer_params, evader_params):
    P = env.cfg.n_pursuers

    def act(key, state):
        del key
        obs = env.observe(state)
        pursuer = deterministic_role_actions(
            pursuer_params, net, obs["self"][:, :P], obs["others"][:, :P])
        evader = deterministic_role_actions(
            evader_params, net, obs["self"][:, P:P + 1], obs["others"][:, P:P + 1])
        return jnp.concatenate([pursuer, evader], axis=1)

    return act


def _scripted_actions(name: str, cfg: GameConfig):
    fn = SCRIPTED[name][1]

    def act(key, state):
        del key
        return fn(state["pos"], state["vel"], state["alive"], cfg, jnp)

    return act


def _mix(env, pursuer_act, evader_act):
    P = env.cfg.n_pursuers

    def act(key, state):
        k1, k2 = jax.random.split(key)
        pursuer = pursuer_act(k1, state)
        evader = evader_act(k2, state)
        return pursuer.at[:, P:].set(evader[:, P:])

    return act


def _evaluate_held_out(env, key, net, p_params, log, step_fn, n_episodes, video, cfg):
    pursuer = _learned_actions(env, net, p_params, p_params)
    for name in ("idle", "flee"):
        act = _mix(env, pursuer, _scripted_actions(name, cfg))
        result = play_matches(env, key, act, n_episodes, cfg.max_steps, step_fn=step_fn)
        key, _ = jax.random.split(key)
        log.log_eval({
            "step": log.manifest.get("steps_taken_main", 0) + log.manifest.get("steps_taken_exploiter", 0),
            "kind": "held_out",
            "opponent": name,
            **result,
        }, step=0 if name == "idle" else 1)
        if video != "none":
            _render_match(env, key, net, p_params, None, log, step_fn, name, video,
                          evader_act=_scripted_actions(name, cfg))
            key, _ = jax.random.split(key)
    return key


def _render_match(env, key, net, p_params, e_params, log, step_fn, opponent, video, evader_act=None):
    if video == "none":
        return
    pursuer = _learned_actions(env, net, p_params, p_params if e_params is None else e_params)
    if evader_act is None:
        act = pursuer
    else:
        act = _mix(env, pursuer, evader_act)
    key, k_reset = jax.random.split(key)
    state, _ = env.reset(k_reset)
    initial = {name: np.asarray(state[name]) for name in ("pos", "vel", "alive")}
    horizon = env.cfg.max_steps
    stored = {name: [] for name in (
        "pos", "vel", "alive", "actions", "tracked", "crashed_oob", "crashed_collision",
        "dones", "outcome_code")}
    for _ in range(horizon):
        key, k_act, k_step = jax.random.split(key, 3)
        actions = act(k_act, state)
        state, _, _, dones, info = step_fn(k_step, state, actions)
        stored["actions"].append(np.asarray(actions))
        stored["dones"].append(np.asarray(dones))
        for name in ("pos", "vel", "alive", "tracked", "crashed_oob", "crashed_collision", "outcome_code"):
            stored[name].append(np.asarray(info[name]))
    frames = {name: np.stack(values, axis=0) for name, values in stored.items()}
    arrays, meta = _pack(initial, frames, horizon)
    step = {"learned": 0, "idle": 1, "flee": 2}.get(opponent, 3)
    log.log_trajectory(arrays, meta, step=step)
    indices = range(arrays["pos"].shape[0]) if video == "all" else (0,)
    for index in indices:
        out = log.run_dir / "videos" / f"ep{meta['episode_ids'][index]}_vs_{opponent}.mp4"
        written = render_episode(
            pos=arrays["pos"][index],
            alive=arrays["alive"][index],
            tracked=arrays["tracked"][index],
            crashed_oob=arrays["crashed_oob"][index],
            crashed_collision=arrays["crashed_collision"][index],
            length=int(arrays["length_frames"][index]),
            cfg=env.cfg,
            out_path=out,
            run_id=log.run_id,
            pursuer_policy=log.manifest["pursuer_policy"],
            evader_policy=opponent,
            outcome=meta["outcomes"][index],
        )
        log.note_video(written)


def _pack(initial, frames, max_steps: int):
    t_steps, n_worlds, n_drones, _ = frames["pos"].shape
    pad = max_steps + 1
    length = np.full(n_worlds, t_steps + 1, dtype=np.int32)
    outcomes = []
    wins = []
    for world in range(n_worlds):
        hits = np.flatnonzero(frames["dones"][:, world])
        end = int(hits[0]) if len(hits) else t_steps - 1
        length[world] = min(pad, end + 2)
        code = int(frames["outcome_code"][end, world])
        outcomes.append(OUTCOME_NAMES.get(code, "timeout_loss"))
        wins.append(code in (1, 3))
    pos = np.zeros((n_worlds, pad, n_drones, 3), np.float32)
    vel = np.zeros_like(pos)
    alive = np.ones((n_worlds, pad, n_drones), dtype=bool)
    action = np.zeros_like(pos)
    tracked = np.zeros((n_worlds, pad), dtype=bool)
    oob = np.zeros((n_worlds, pad, n_drones), dtype=bool)
    col = np.zeros_like(oob)
    pos[:, 0] = initial["pos"]
    vel[:, 0] = initial["vel"]
    alive[:, 0] = initial["alive"]
    limit = min(t_steps, pad - 1)
    pos[:, 1:limit + 1] = np.moveaxis(frames["pos"][:limit], 0, 1)
    vel[:, 1:limit + 1] = np.moveaxis(frames["vel"][:limit], 0, 1)
    alive[:, 1:limit + 1] = np.moveaxis(frames["alive"][:limit], 0, 1)
    action[:, :limit] = np.moveaxis(frames["actions"][:limit], 0, 1)
    tracked[:, 1:limit + 1] = np.moveaxis(frames["tracked"][:limit], 0, 1)
    oob[:, 1:limit + 1] = np.moveaxis(frames["crashed_oob"][:limit], 0, 1)
    col[:, 1:limit + 1] = np.moveaxis(frames["crashed_collision"][:limit], 0, 1)
    for world in range(n_worlds):
        last = int(length[world])
        if last < pad:
            pos[world, last:] = pos[world, last - 1]
            vel[world, last:] = vel[world, last - 1]
            alive[world, last:] = alive[world, last - 1]
            tracked[world, last:] = tracked[world, last - 1]
    arrays = {
        "pos": pos, "vel": vel, "alive": alive, "action": action, "tracked": tracked,
        "crashed_oob": oob, "crashed_collision": col, "length_frames": length,
    }
    meta = {
        "step": 0,
        "episode_ids": list(range(n_worlds)),
        "worlds": list(range(n_worlds)),
        "outcomes": outcomes,
        "pursuers_win": wins,
        "length_frames": length.tolist(),
        "length_steps": (length - 1).tolist(),
    }
    return arrays, meta


def _fresh_exploiter(env, key, net, p_params, optimizer, budget, log, step_fn, eval_episodes, hidden):
    del hidden
    P = env.cfg.n_pursuers
    key, k_init = jax.random.split(key)
    obs = env.observe(env.reset(jax.random.PRNGKey(0))[0])
    params, opt_state = init_role(k_init, net, obs["self"][:, P:P + 1], obs["others"][:, P:P + 1])
    key, k_reset = jax.random.split(key)
    state, _ = env.reset(k_reset)
    taken = 0
    horizon = min(env.cfg.max_steps, budget // env.n_worlds)
    while taken + horizon * env.n_worlds <= budget and horizon >= 1:
        key, state, frames, _, e_boot = collect_rollout(
            key, state, p_params, params, horizon, env, net)
        params, opt_state, _ = update_evader(params, opt_state, optimizer, frames, e_boot, net, EPOCHS)
        taken += horizon * env.n_worlds
        horizon = min(env.cfg.max_steps, (budget - taken) // env.n_worlds)

    def exploiter_act(key, state):
        del key
        obs_now = env.observe(state)
        evader = deterministic_role_actions(
            params, net, obs_now["self"][:, P:P + 1], obs_now["others"][:, P:P + 1])
        pursuer = deterministic_role_actions(
            p_params, net, obs_now["self"][:, :P], obs_now["others"][:, :P])
        return jnp.concatenate([pursuer, evader], axis=1)

    result = play_matches(env, key, exploiter_act, eval_episodes, env.cfg.max_steps, step_fn=step_fn)
    log.log_eval({
        "step": taken,
        "kind": "exploitability",
        "opponent": "fresh_exploiter",
        "exploiter_budget": budget,
        **result,
    }, step=2)
    return key
