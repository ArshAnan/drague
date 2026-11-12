"""The run logbook writes a labeled directory, and a short clip can be rendered from it."""

import json
from pathlib import Path

import numpy as np

from drague.config import GameConfig
from drague.logbook import EPISODE_FIELDS, LOSS_OUTCOMES, WIN_OUTCOMES
from drague.rollout import outcome_labels, record_scripted_run
from drague.video import render_episode


def test_outcome_priority():
    evader_down = np.array([True, False, False, False, True])
    pursuers_down = np.array([True, True, False, False, False])
    timeout = np.array([True, True, True, True, False])
    won_on_time = np.array([False, False, True, False, False])
    labels = outcome_labels(evader_down, pursuers_down, timeout, won_on_time)
    assert list(labels) == [
        "evader_crashed",
        "pursuers_wiped",
        "timeout_win",
        "timeout_loss",
        "evader_crashed",
    ]


def _record(tmp_path: Path):
    cfg = GameConfig(n_pursuers=3, episode_seconds=0.25)
    return record_scripted_run(
        pursuer="standoff",
        evader="idle",
        seed=0,
        n_episodes=2,
        n_worlds=2,
        n_trajectories=2,
        metrics_every=64,
        video="none",
        results_dir=tmp_path,
        cfg=cfg,
        command="test",
    )


def test_rollout_writes_the_schema(tmp_path):
    run = _record(tmp_path)
    manifest = json.loads((run / "manifest.json").read_text())
    for key in ("schema_version", "run_id", "started_at", "ended_at", "command", "git_commit",
                "git_dirty", "host", "platform", "device", "jax_version", "drague_version",
                "method", "seed", "backend", "n_worlds", "n_episodes", "pursuer_policy",
                "evader_policy", "game", "budget", "hparams"):
        assert key in manifest
    assert manifest["ended_at"]
    assert manifest["method"] == "scripted"
    assert manifest["pursuer_policy"] == "standoff_pursuers"
    assert manifest["evader_policy"] == "idle"
    assert manifest["game"]["n_pursuers"] == 3
    assert manifest["game"]["episode_seconds"] == 0.25
    assert manifest["budget"]["env_steps_exploiter"] == 0
    assert manifest["backend"] == "point_mass"
    assert manifest["run_id"].endswith("_scripted_standoff_vs_idle_p3_s0")
    assert manifest["run_id"][4] == "-" and manifest["run_id"][10] == "_"

    episodes = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
    assert len(episodes) == 2
    max_steps = int(round(manifest["game"]["episode_seconds"] / manifest["game"]["control_dt"]))
    env_steps = 0
    for episode in episodes:
        assert set(EPISODE_FIELDS) <= set(episode)
        parts = (episode["return_tracking"] + episode["return_shaping"]
                 + episode["return_crash"] + episode["return_terminal"])
        assert abs(parts - episode["return_total"]) < 1e-4
        if episode["pursuers_win"]:
            assert episode["outcome"] in WIN_OUTCOMES
        else:
            assert episode["outcome"] in LOSS_OUTCOMES
        env_steps += episode["length_steps"]
        assert abs(episode["tracked_fraction"] - episode["tracked_steps"] / max_steps) < 1e-6

    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 1
    assert metrics[0]["episodes"] == 2
    assert metrics[0]["env_steps"] == env_steps
    assert "pursuer_win_rate_se" in metrics[0]

    summary = json.loads((run / "summary.json").read_text())
    assert summary["n_episodes"] == 2
    assert summary["env_steps_total"] == env_steps
    assert abs(summary["pursuer_win_rate"] - np.mean([ep["pursuers_win"] for ep in episodes])) < 1e-6
    assert sum(summary["outcome_counts"].values()) == 2

    eval_path = run / "eval" / "000000000.json"
    evaluation = json.loads(eval_path.read_text())
    assert evaluation["kind"] == "scripted"
    assert evaluation["n_episodes"] == 2
    assert evaluation["outcome_counts"] == summary["outcome_counts"]

    npz_files = list((run / "trajectories").glob("*.npz"))
    assert len(npz_files) == 1
    sidecar = json.loads(npz_files[0].with_suffix(".json").read_text())
    with np.load(npz_files[0]) as data:
        assert data["pos"].shape[0] == 2
        assert data["pos"].shape[1] >= int(data["length_frames"].max())
        assert data["pos"].shape[-1] == 3
        assert sidecar["episode_ids"] == [0, 1]
        assert len(sidecar["outcomes"]) == 2

    index = [json.loads(line) for line in (tmp_path / "index.jsonl").read_text().splitlines()]
    assert index[-1]["run_id"] == manifest["run_id"]
    assert index[-1]["pursuer_win_rate"] == summary["pursuer_win_rate"]
    assert Path(index[-1]["path"]) == run


def test_two_frame_gif(tmp_path):
    run = _record(tmp_path)
    manifest = json.loads((run / "manifest.json").read_text())
    npz_path = next((run / "trajectories").glob("*.npz"))
    meta = json.loads(npz_path.with_suffix(".json").read_text())
    with np.load(npz_path) as data:
        out = render_episode(
            pos=data["pos"][0],
            alive=data["alive"][0],
            tracked=data["tracked"][0],
            crashed_oob=data["crashed_oob"][0],
            crashed_collision=data["crashed_collision"][0],
            length=int(data["length_frames"][0]),
            cfg=GameConfig(**manifest["game"]),
            out_path=tmp_path / "clip.gif",
            run_id=manifest["run_id"],
            pursuer_policy=manifest["pursuer_policy"],
            evader_policy=manifest["evader_policy"],
            outcome=meta["outcomes"][0],
            max_frames=2,
        )
    assert out.suffix == ".gif"
    assert out.stat().st_size > 0


def test_exploiter_steps_are_not_added_twice(tmp_path):
    from drague.logbook import EPISODE_FIELDS, RunLog

    def episode(length, win, learner):
        record = {key: 0 for key in EPISODE_FIELDS}
        record.update({
            "episode_id": 0, "world": 0, "env_steps_at_end": length,
            "outcome": "timeout_win" if win else "timeout_loss",
            "pursuers_win": win, "length_steps": length, "tracked_steps": 0,
            "tracked_fraction": 0.0, "time_to_first_track_steps": None,
            "mean_d_min_m": 1.0, "final_d_min_m": 1.0, "min_pursuer_separation_m": None,
            "learner": learner,
        })
        return record

    with RunLog.open(
        tmp_path, method="league", seed=0, cfg=GameConfig(n_pursuers=2),
        pursuer_policy="mappo", evader_policy="mappo", n_worlds=2, n_episodes=2,
    ) as log:
        log.log_episode(episode(5, True, "main"))
        log.log_episode(episode(3, False, "exploiter"))
    summary = json.loads((log.run_dir / "summary.json").read_text())
    manifest = json.loads((log.run_dir / "manifest.json").read_text())
    assert manifest["budget"]["env_steps_main"] == 5
    assert manifest["budget"]["env_steps_exploiter"] == 3
    assert manifest["budget"]["env_steps_total"] == 8
    assert summary["pursuer_win_rate"] == 1.0
    assert summary["n_exploiter_episodes"] == 1
