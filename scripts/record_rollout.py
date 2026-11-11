"""Play a scripted match and write a labeled run, including a demo video.

    uv run python scripts/record_rollout.py
    uv run python scripts/record_rollout.py --pursuer standoff --evader idle --n-pursuers 4
    uv run python scripts/record_rollout.py --video-episodes all

The cells from EXPERIMENTS.md are standoff-vs-idle, standoff-vs-flee, and idle-vs-flee,
for --n-pursuers 2, 3, 4, and 8. One invocation records one cell.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drague.rollout import SCRIPTED, record_scripted_run  # noqa: E402


def main():
    names = list(SCRIPTED)
    ap = argparse.ArgumentParser(description="Record a scripted tracking match")
    ap.add_argument("--pursuer", default="standoff", choices=names)
    ap.add_argument("--evader", default="flee", choices=names)
    ap.add_argument("--n-pursuers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=1024)
    ap.add_argument("--worlds", type=int, default=256)
    ap.add_argument("--metrics-every", type=int, default=64)
    ap.add_argument("--trajectories", type=int, default=8)
    ap.add_argument("--video-episodes", default="first", choices=["first", "all", "none"])
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    run_dir = record_scripted_run(
        pursuer=args.pursuer,
        evader=args.evader,
        n_pursuers=args.n_pursuers,
        seed=args.seed,
        n_episodes=args.episodes,
        n_worlds=args.worlds,
        metrics_every=args.metrics_every,
        n_trajectories=args.trajectories,
        video=args.video_episodes,
        results_dir=args.results,
        command=" ".join(sys.argv),
    )
    summary_path = run_dir / "summary.json"
    print(f"wrote {run_dir}")
    print(f"summary {summary_path}")


if __name__ == "__main__":
    main()
