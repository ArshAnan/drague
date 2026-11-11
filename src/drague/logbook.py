"""One labeled directory per simulation run.

A run is the unit you cite in the paper. `RunLog` writes a manifest (what was run, on which
code, with which config), one JSON line per episode, a scalar curve, a sample of trajectories,
an eval snapshot, and a one-row summary. `results/index.jsonl` lists every run.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from drague import __version__
from drague.config import GameConfig

SCHEMA_VERSION = 1

EPISODE_FIELDS = (
    "episode_id",
    "world",
    "env_steps_at_end",
    "outcome",
    "pursuers_win",
    "length_steps",
    "tracked_steps",
    "tracked_fraction",
    "time_to_first_track_steps",
    "pursuer_oob",
    "pursuer_collision",
    "evader_oob",
    "return_tracking",
    "return_shaping",
    "return_crash",
    "return_terminal",
    "return_total",
    "mean_d_min_m",
    "final_d_min_m",
    "min_pursuer_separation_m",
    "pursuer_speed_saturated",
    "evader_speed_saturated",
)

# Interval means written to metrics.jsonl. `pursuers_win` on the episode becomes `pursuer_win_rate`.
METRIC_FIELDS = (
    "pursuer_win_rate",
    "tracked_fraction",
    "return_total",
    "return_tracking",
    "return_shaping",
    "return_crash",
    "return_terminal",
    "length_steps",
    "pursuer_oob",
    "pursuer_collision",
    "evader_oob",
    "pursuer_speed_saturated",
    "evader_speed_saturated",
    "mean_d_min_m",
)

WIN_OUTCOMES = frozenset({"evader_crashed", "timeout_win"})
LOSS_OUTCOMES = frozenset({"pursuers_wiped", "timeout_loss"})


def to_jsonable(value):
    """Numpy scalars and arrays become plain JSON values. Everything else passes through."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _utc_stamp(when: datetime | None = None) -> str:
    when = when or datetime.now(timezone.utc)
    return when.strftime("%Y%m%dT%H%M%SZ")


def make_run_id(method: str, pursuer: str, evader: str, n_pursuers: int, seed: int,
                when: datetime | None = None) -> str:
    """`scripted_standoff_vs_flee_p3_s0_20261111T193000Z`."""
    return f"{method}_{pursuer}_vs_{evader}_p{n_pursuers}_s{seed}_{_utc_stamp(when)}"


def _git_info(cwd: Path | None = None) -> dict:
    """Commit hash and whether the tree has uncommitted changes. Null when git is unavailable."""
    def run(args):
        return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)

    try:
        commit = run(["git", "rev-parse", "HEAD"])
        dirty = run(["git", "status", "--porcelain"])
    except OSError:
        return {"git_commit": None, "git_dirty": None}
    if commit.returncode != 0:
        return {"git_commit": None, "git_dirty": None}
    return {
        "git_commit": commit.stdout.strip() or None,
        "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def _jax_info() -> dict:
    try:
        import jax
    except ImportError:
        return {"jax_version": None, "device": "cpu"}
    devices = [str(d) for d in jax.devices()]
    return {"jax_version": jax.__version__, "device": devices[0] if devices else "cpu"}


def _mean_se(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return 0.0, 0.0
    mean = float(arr.mean())
    if n == 1:
        return mean, 0.0
    return mean, float(arr.std(ddof=1) / np.sqrt(n))


def _episode_metric(episode: dict, field: str) -> float:
    if field == "pursuer_win_rate":
        return 1.0 if episode["pursuers_win"] else 0.0
    value = episode[field]
    if value is None:
        return float("nan")
    return float(value)


class RunLog:
    """Append-only writer for one run directory under `results/`."""

    def __init__(self, run_dir: Path, manifest: dict):
        self.run_dir = Path(run_dir)
        self.manifest = manifest
        self.episodes: list[dict] = []
        self._episodes_file = open(self.run_dir / "episodes.jsonl", "a", encoding="utf-8")
        self._metrics_file = open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8")
        self._closed = False
        self._final_summary: dict | None = None

    @classmethod
    def open(
        cls,
        results_dir: Path | str,
        *,
        method: str,
        seed: int,
        cfg: GameConfig,
        pursuer_policy: str,
        evader_policy: str,
        pursuer_label: str | None = None,
        evader_label: str | None = None,
        backend: str = "point_mass",
        n_worlds: int,
        n_episodes: int,
        hparams: dict | None = None,
        command: str | None = None,
        device: str | None = None,
        run_id: str | None = None,
        when: datetime | None = None,
    ) -> "RunLog":
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        when = when or datetime.now(timezone.utc)
        pursuer_label = pursuer_label or pursuer_policy
        evader_label = evader_label or evader_policy
        run_id = run_id or make_run_id(method, pursuer_label, evader_label, cfg.n_pursuers, seed, when)
        candidate = run_id
        suffix = 2
        while (results_dir / candidate).exists():
            candidate = f"{run_id}_{suffix}"
            suffix += 1
        run_id = candidate
        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "eval").mkdir()
        (run_dir / "trajectories").mkdir()
        (run_dir / "videos").mkdir()

        jax_info = _jax_info()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "started_at": when.isoformat(),
            "ended_at": None,
            "wall_s": None,
            "command": command,
            **_git_info(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "device": device or jax_info["device"],
            "jax_version": jax_info["jax_version"],
            "drague_version": __version__,
            "method": method,
            "seed": seed,
            "backend": backend,
            "n_worlds": n_worlds,
            "n_episodes": n_episodes,
            "pursuer_policy": pursuer_policy,
            "evader_policy": evader_policy,
            "game": asdict(cfg),
            "budget": {
                "env_steps_main": 0,
                "env_steps_exploiter": 0,
                "env_steps_total": 0,
            },
            "hparams": hparams or {},
            "videos": [],
        }
        log = cls(run_dir, manifest)
        log._write_manifest()
        return log

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def run_id(self) -> str:
        return self.manifest["run_id"]

    def log_episode(self, record: dict) -> None:
        missing = [key for key in EPISODE_FIELDS if key not in record]
        if missing:
            raise KeyError(f"episode record missing {missing}")
        record = to_jsonable(record)
        self.episodes.append(record)
        self._episodes_file.write(json.dumps(record) + "\n")
        self._episodes_file.flush()

    def log_metrics(self, record: dict) -> None:
        self._metrics_file.write(json.dumps(to_jsonable(record)) + "\n")
        self._metrics_file.flush()

    def log_trajectory(self, arrays: dict, meta: dict, step: int = 0) -> Path:
        """Save one sample of episodes. `arrays` values are numpy arrays of leading length E."""
        name = f"{step:09d}"
        path = self.run_dir / "trajectories" / f"{name}.npz"
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})
        sidecar = self.run_dir / "trajectories" / f"{name}.json"
        sidecar.write_text(json.dumps(to_jsonable(meta), indent=2) + "\n", encoding="utf-8")
        return path

    def log_eval(self, payload: dict, step: int = 0) -> Path:
        path = self.run_dir / "eval" / f"{step:09d}.json"
        path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n", encoding="utf-8")
        return path

    def note_video(self, path: Path) -> None:
        self.manifest["videos"].append(Path(path).name)

    def close(self) -> dict:
        """Write summary.json, finish the manifest, and append results/index.jsonl."""
        if self._closed:
            return self._final_summary or {}
        self._closed = True
        self._episodes_file.close()
        self._metrics_file.close()

        ended = datetime.now(timezone.utc)
        started = datetime.fromisoformat(self.manifest["started_at"])
        self.manifest["ended_at"] = ended.isoformat()
        self.manifest["wall_s"] = round((ended - started).total_seconds(), 3)
        summary = self.summary()
        main_steps, exploiter_steps = self._step_budget()
        self.manifest["budget"] = {
            "env_steps_main": main_steps,
            "env_steps_exploiter": exploiter_steps,
            "env_steps_total": main_steps + exploiter_steps,
        }
        summary["env_steps_main"] = main_steps
        summary["env_steps_exploiter"] = exploiter_steps
        summary["env_steps_total"] = main_steps + exploiter_steps
        (self.run_dir / "summary.json").write_text(
            json.dumps(to_jsonable(summary), indent=2) + "\n", encoding="utf-8")
        self._write_manifest()
        self._append_index(summary)
        self._final_summary = summary
        return summary

    def _cited_episodes(self) -> list[dict]:
        """Headline metrics use main-learner episodes. Scripted rows have no learner tag."""
        cited = [ep for ep in self.episodes if ep.get("learner", "main") != "exploiter"]
        return cited or self.episodes

    def _step_budget(self) -> tuple[int, int]:
        """Env steps from the trainer's counters, or from logged episode lengths."""
        if "steps_taken_main" in self.manifest:
            return (int(self.manifest["steps_taken_main"]),
                    int(self.manifest.get("steps_taken_exploiter", 0)))
        main = exploiter = 0
        for ep in self.episodes:
            if ep.get("learner") == "exploiter":
                exploiter += int(ep["length_steps"])
            else:
                main += int(ep["length_steps"])
        return main, exploiter

    def summary(self) -> dict:
        episodes = self._cited_episodes()
        n = len(episodes)
        env_steps = int(sum(ep["length_steps"] for ep in episodes)) if n else 0
        out: dict = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "method": self.manifest["method"],
            "n_pursuers": self.manifest["game"]["n_pursuers"],
            "seed": self.manifest["seed"],
            "pursuer_policy": self.manifest["pursuer_policy"],
            "evader_policy": self.manifest["evader_policy"],
            "backend": self.manifest["backend"],
            "n_episodes": n,
            "n_exploiter_episodes": sum(ep.get("learner") == "exploiter" for ep in self.episodes),
            "env_steps_total": env_steps,
        }
        for field in METRIC_FIELDS:
            values = [v for v in (_episode_metric(ep, field) for ep in episodes) if v == v]
            mean, se = _mean_se(values)
            out[field] = mean
            out[f"{field}_se"] = se
        finals = [ep["final_d_min_m"] for ep in episodes if ep["final_d_min_m"] is not None]
        seps = [ep["min_pursuer_separation_m"] for ep in episodes
                if ep["min_pursuer_separation_m"] is not None]
        first = [ep["time_to_first_track_steps"] for ep in episodes
                 if ep["time_to_first_track_steps"] is not None]
        out["final_d_min_m"] = _mean_se(finals)[0]
        out["min_pursuer_separation_m"] = _mean_se(seps)[0] if seps else None
        out["time_to_first_track_steps"] = _mean_se(first)[0] if first else None
        out["never_tracked_fraction"] = (
            sum(ep["time_to_first_track_steps"] is None for ep in episodes) / n if n else 0.0
        )
        counts: dict[str, int] = {}
        for ep in episodes:
            counts[ep["outcome"]] = counts.get(ep["outcome"], 0) + 1
        out["outcome_counts"] = counts
        return out

    def metrics_record(self, episodes: list[dict], *, env_steps: int, episodes_so_far: int,
                       wall_s: float) -> dict:
        """Scalars for one stretch of episodes. Rates are over `episodes`; counts are cumulative."""
        record = {
            "env_steps": env_steps,
            "episodes": episodes_so_far,
            "interval_episodes": len(episodes),
            "wall_s": round(wall_s, 3),
            "env_steps_per_s": round(env_steps / wall_s, 3) if wall_s > 0 else 0.0,
        }
        for field in METRIC_FIELDS:
            values = [v for v in (_episode_metric(ep, field) for ep in episodes) if v == v]
            mean, se = _mean_se(values)
            record[field] = mean
            record[f"{field}_se"] = se
        cited = self._cited_episodes()
        if cited:
            wins = [1.0 if ep["pursuers_win"] else 0.0 for ep in cited]
            tracked = [float(ep["tracked_fraction"]) for ep in cited]
            record["cumulative_pursuer_win_rate"] = _mean_se(wins)[0]
            record["cumulative_tracked_fraction"] = _mean_se(tracked)[0]
        return record

    def _write_manifest(self) -> None:
        (self.run_dir / "manifest.json").write_text(
            json.dumps(to_jsonable(self.manifest), indent=2) + "\n", encoding="utf-8")

    def _append_index(self, summary: dict) -> None:
        index = self.run_dir.parent / "index.jsonl"
        row = {
            "run_id": self.run_id,
            "method": self.manifest["method"],
            "n_pursuers": self.manifest["game"]["n_pursuers"],
            "seed": self.manifest["seed"],
            "pursuer_policy": self.manifest["pursuer_policy"],
            "evader_policy": self.manifest["evader_policy"],
            "pursuer_win_rate": summary["pursuer_win_rate"],
            "tracked_fraction": summary["tracked_fraction"],
            "env_steps_total": summary["env_steps_total"],
            "path": str(self.run_dir),
        }
        with index.open("a", encoding="utf-8") as f:
            f.write(json.dumps(to_jsonable(row)) + "\n")
