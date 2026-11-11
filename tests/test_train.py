"""A short training cell writes the run log, the budget, and a clip."""

import json
from pathlib import Path

import numpy as np

from drague.config import GameConfig
from drague.train import train


def _cfg():
    return GameConfig(n_pursuers=2, episode_seconds=0.15)


def _summary(run: Path):
    return json.loads((run / "summary.json").read_text(encoding="utf-8"))


def _manifest(run: Path):
    return json.loads((run / "manifest.json").read_text(encoding="utf-8"))


def test_selfplay_matches_budget_and_writes_a_clip(tmp_path):
    run = train(
        method="selfplay", seed=0, budget=12, n_worlds=4, video="first",
        results_dir=tmp_path, cfg=_cfg(), measure_exploitability=False,
        eval_episodes=4, command="test",
    )
    manifest = _manifest(run)
    assert manifest["budget"]["env_steps_total"] == 12
    assert manifest["budget"]["env_steps_exploiter"] == 0
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert metrics
    assert np.isfinite(metrics[-1]["policy_loss"])
    assert "pursuer_win_rate" in metrics[-1]
    clips = list((run / "videos").glob("ep0_vs_learned.*"))
    assert clips and clips[0].stat().st_size > 0
    assert (run / "checkpoints" / "pursuer.msgpack").is_file()
    assert _summary(run)["n_episodes"] >= 1


def test_pfsp_and_league_budgets(tmp_path):
    pfsp = train(
        method="pfsp", seed=0, budget=12, n_worlds=4, video="none",
        results_dir=tmp_path / "pfsp", cfg=_cfg(), measure_exploitability=False,
        eval_episodes=4, command="test",
    )
    assert _manifest(pfsp)["budget"]["env_steps_total"] == 12
    assert _manifest(pfsp)["budget"]["env_steps_exploiter"] == 0

    league = train(
        method="league", seed=0, budget=24, n_worlds=4, video="none",
        results_dir=tmp_path / "league", cfg=_cfg(), measure_exploitability=False,
        eval_episodes=4, exploiter_fraction=0.5, command="test",
    )
    budget = _manifest(league)["budget"]
    assert budget["env_steps_total"] == 24
    assert budget["env_steps_exploiter"] > 0
    assert budget["env_steps_main"] + budget["env_steps_exploiter"] == 24
