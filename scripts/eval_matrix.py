"""Cross-play learned pursuers against learned evaders and write Elo.

    uv run python scripts/eval_matrix.py --runs results/<run_a> results/<run_b>
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from drague.algos.mappo import deterministic_role_actions  # noqa: E402
from drague.algos.policy import ActorCritic  # noqa: E402
from drague.config import GameConfig  # noqa: E402
from drague.env import TrackingEnv  # noqa: E402
from drague.eval import cross_play  # noqa: E402
from flax import serialization  # noqa: E402


def _load(run_dir: Path):
    meta = json.loads((run_dir / "checkpoints" / "meta.json").read_text(encoding="utf-8"))
    cfg = GameConfig(n_pursuers=meta["n_pursuers"])
    env = TrackingEnv(cfg, n_worlds=1)
    net = ActorCritic(hidden=meta["hidden"])
    _, obs = env.reset(jax.random.PRNGKey(0))
    P = cfg.n_pursuers
    pursuer_template = net.init(jax.random.PRNGKey(1), obs["self"][:, :P], obs["others"][:, :P])
    evader_template = net.init(jax.random.PRNGKey(2), obs["self"][:, P:P + 1], obs["others"][:, P:P + 1])
    pursuer = serialization.from_bytes(pursuer_template, (run_dir / "checkpoints" / "pursuer.msgpack").read_bytes())
    evader = serialization.from_bytes(evader_template, (run_dir / "checkpoints" / "evader.msgpack").read_bytes())
    return meta, net, pursuer, evader


def main():
    ap = argparse.ArgumentParser(description="Cross-play saved pursuers and evaders")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    loaded = [_load(Path(path)) for path in args.runs]
    n_pursuers = loaded[0][0]["n_pursuers"]
    if any(meta["n_pursuers"] != n_pursuers for meta, *_ in loaded):
        raise SystemExit("all runs must use the same team size")
    cfg = GameConfig(n_pursuers=n_pursuers)
    env = TrackingEnv(cfg, n_worlds=args.worlds)
    nets = [net for _, net, _, _ in loaded]

    def role_fn(params, net, pursuer: bool):
        P = n_pursuers

        def act(key, state):
            del key
            obs = env.observe(state)
            if pursuer:
                actions = deterministic_role_actions(params, net, obs["self"][:, :P], obs["others"][:, :P])
                full = jnp.zeros((args.worlds, cfg.n_drones, 3))
                return full.at[:, :P].set(actions)
            actions = deterministic_role_actions(
                params, net, obs["self"][:, P:P + 1], obs["others"][:, P:P + 1])
            full = jnp.zeros((args.worlds, cfg.n_drones, 3))
            return full.at[:, P:].set(actions)

        return act

    names = [Path(path).name for path in args.runs]
    pursuers = [(name, role_fn(p, net, True)) for name, (_, net, p, _) in zip(names, loaded)]
    evaders = [(name, role_fn(e, net, False)) for name, (_, net, _, e) in zip(names, loaded)]
    del nets
    result = cross_play(env, jax.random.PRNGKey(0), pursuers, evaders, args.episodes, cfg.max_steps)
    out = Path(args.out) if args.out else Path(args.runs[0]).parent / "cross_play.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
