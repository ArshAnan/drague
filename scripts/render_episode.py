"""Render a demo video from a run that was already recorded.

    uv run python scripts/render_episode.py --run results/<run_id>
    uv run python scripts/render_episode.py --run results/<run_id> --episode 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drague.video import render_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Render a top-and-side video of a saved episode")
    ap.add_argument("--run", required=True, help="path to results/<run_id>")
    ap.add_argument("--episode", type=int, default=None, help="episode id in the saved sample")
    ap.add_argument("--index", type=int, default=None, help="row in the trajectory file")
    ap.add_argument("--step", type=int, default=None, help="checkpoint step, if several samples exist")
    args = ap.parse_args()

    indices = None if args.index is None else [args.index]
    paths = render_run(
        args.run, episode=args.episode, indices=indices, step=args.step,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
