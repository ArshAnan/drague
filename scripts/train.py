"""Train one cell: a strategy, a team size, a budget, and a seed.

    uv run python scripts/train.py --method selfplay --n-pursuers 3 --budget smoke --seed 0
    uv run python scripts/train.py --method league --n-pursuers 3 --budget paper --seed 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drague.train import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Train a pursuer team and an evader")
    ap.add_argument("--method", default="selfplay", choices=["selfplay", "pfsp", "league"])
    ap.add_argument("--n-pursuers", type=int, default=3)
    ap.add_argument("--budget", default="smoke", help="smoke, paper, or an env-step count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--worlds", type=int, default=None)
    ap.add_argument("--video", default="first", choices=["first", "all", "none"])
    ap.add_argument("--results", default="results")
    ap.add_argument("--exploitability", action="store_true",
                    help="train a fresh evader against the frozen pursuer after the run")
    ap.add_argument("--no-exploitability", action="store_true")
    args = ap.parse_args()
    measure = None
    if args.exploitability:
        measure = True
    elif args.no_exploitability:
        measure = False
    budget = int(args.budget) if args.budget.isdigit() else args.budget
    run = train(
        method=args.method,
        n_pursuers=args.n_pursuers,
        seed=args.seed,
        budget=budget,
        n_worlds=args.worlds,
        video=args.video,
        results_dir=args.results,
        measure_exploitability=measure,
        command=" ".join(sys.argv),
    )
    print(f"wrote {run}")


if __name__ == "__main__":
    main()
