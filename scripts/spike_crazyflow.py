"""Week-1 spike: does Crazyflow do what the plan assumes, and how fast is it on this machine?

Run locally (CPU) first, then on Empire AI (GPU):

    uv run python scripts/spike_crazyflow.py                      # quick check, CPU
    uv run python scripts/spike_crazyflow.py --device gpu --sweep  # full throughput sweep

Each check prints PASS / FAIL and the script keeps going after a failure, so one run tells us
everything that needs fixing. Results are saved to results/spike_<host>_<time>.json.
Commit that file (or paste it to your teammate) so budgets are planned from real numbers.

Checks:
  1. versions        crazyflow / jax import, which devices JAX sees
  2. spawn           write our pursuer-ring spawn into the sim and read it back
  3. hover           drones hold their spawned positions under hover thrust
  4. masked_reset    sim.reset(mask) resets only the selected worlds
  5. per_drone_mass  physical parameters can differ per drone (needed if sides differ physically)
  6. contacts        sim.contacts() runs on a multi-drone scene
  7. throughput      control steps per second for several (worlds, team size) settings
"""

import argparse
import json
import platform
import socket
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drague.config import GameConfig  # noqa: E402
from drague import rules  # noqa: E402

RESULTS = {"host": socket.gethostname(), "platform": platform.platform(), "checks": {}}


def check(name):
    """Decorator: run a check, record PASS/FAIL plus whatever it returns, never crash the script."""
    def wrap(fn):
        def run(*a, **k):
            t0 = time.perf_counter()
            try:
                out = fn(*a, **k) or {}
                RESULTS["checks"][name] = {"ok": True, **out}
                print(f"PASS  {name:16s} {out}")
            except Exception as e:  # noqa: BLE001
                RESULTS["checks"][name] = {"ok": False, "error": repr(e)}
                print(f"FAIL  {name:16s} {e!r}")
                traceback.print_exc(limit=2)
            RESULTS["checks"][name]["seconds"] = round(time.perf_counter() - t0, 2)
        return run
    return wrap


def make_sim(n_worlds, n_drones, device):
    from crazyflow.control import Control
    from crazyflow.sim import Dynamics, Sim
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, control=Control.attitude,
              dynamics=Dynamics.so_rpy, device=device)
    sim.reset()
    return sim


def hover_cmd(sim):
    import jax.numpy as jnp
    mass = float(jnp.ravel(sim.data.params.mass)[0])
    cmd = np.zeros((sim.n_worlds, sim.n_drones, 4), dtype=np.float32)
    cmd[..., 3] = mass * 9.81
    return cmd


def spawn_into(sim, cfg, seed=0):
    import jax
    import jax.numpy as jnp
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    W, P = sim.n_worlds, cfg.n_pursuers
    pos = rules.spawn_positions(jax.random.uniform(k1, (W, 1)), jax.random.uniform(k2, (W, P, 3)),
                                jax.random.uniform(k3, (W, 2)), cfg, jnp)
    pos = pos.astype(sim.data.states.pos.dtype)
    sim.data = sim.data.replace(states=sim.data.states.replace(pos=pos))
    return pos


@check("versions")
def check_versions():
    import crazyflow
    import jax
    return {"crazyflow": getattr(crazyflow, "__version__", "unknown"), "jax": jax.__version__,
            "devices": [str(d) for d in jax.devices()]}


@check("spawn")
def check_spawn(device, cfg):
    import jax.numpy as jnp
    sim = make_sim(256, cfg.n_drones, device)
    pos = spawn_into(sim, cfg)
    back = sim.data.states.pos
    assert back.shape == (256, cfg.n_drones, 3), back.shape
    assert jnp.allclose(back, pos), "positions did not round-trip"
    alive = jnp.ones((256, cfg.n_drones), dtype=bool)
    assert not bool(jnp.any(rules.out_of_bounds(back, cfg, jnp))), "spawned out of bounds"
    assert not bool(jnp.any(rules.is_tracked(back, alive, cfg, jnp))), "spawned already tracking"
    return {"shape": list(back.shape)}


@check("hover")
def check_hover(device, cfg, seconds=2.0):
    import jax.numpy as jnp
    sim = make_sim(256, cfg.n_drones, device)
    pos0 = spawn_into(sim, cfg)
    sim.attitude_control(hover_cmd(sim))
    n_ctrl = int(seconds * sim.control_freq)
    for _ in range(n_ctrl):
        sim.step(sim.freq // sim.control_freq)
    drift = float(jnp.max(jnp.linalg.norm(sim.data.states.pos - pos0, axis=-1)))
    assert drift < 0.25, f"drones drifted {drift:.3f} m in {seconds}s of hover"
    return {"max_drift_m": round(drift, 4), "freq": sim.freq, "control_freq": sim.control_freq}


@check("masked_reset")
def check_masked_reset(device, cfg):
    import jax.numpy as jnp
    sim = make_sim(8, cfg.n_drones, device)
    default = sim.data.states.pos
    spawned = spawn_into(sim, cfg)
    mask = jnp.array([True, False] * 4)
    sim.reset(mask)
    after = sim.data.states.pos
    assert jnp.allclose(after[mask], default[mask]), "masked worlds were not reset"
    assert jnp.allclose(after[~mask], spawned[~mask]), "unmasked worlds changed"
    return {}


@check("per_drone_mass")
def check_per_drone_mass(device, cfg):
    import jax.numpy as jnp
    sim = make_sim(16, cfg.n_drones, device)
    m = sim.data.params.mass
    shape_before = list(m.shape)
    scale = jnp.ones((1, cfg.n_drones, 1)).at[0, -1, 0].set(0.8)   # lighter evader
    sim.data = sim.data.replace(params=sim.data.params.replace(mass=m * scale))
    sim.attitude_control(hover_cmd(sim))
    sim.step(sim.freq // sim.control_freq)
    assert bool(jnp.all(jnp.isfinite(sim.data.states.pos)))
    return {"mass_shape_default": shape_before, "mass_shape_after": list(sim.data.params.mass.shape)}


@check("contacts")
def check_contacts(device, cfg):
    sim = make_sim(4, cfg.n_drones, device)
    spawn_into(sim, cfg)
    c = sim.contacts()
    return {"contacts_shape": list(getattr(c, "shape", [])), "any": bool(np.asarray(c).any())}


@check("throughput")
def check_throughput(device, worlds_list, pursuers_list, n_ctrl=200):
    rows = []
    for P in pursuers_list:
        cfg = GameConfig(n_pursuers=P)
        for W in worlds_list:
            sim = make_sim(W, cfg.n_drones, device)
            spawn_into(sim, cfg)
            sim.attitude_control(hover_cmd(sim))
            ticks = sim.freq // sim.control_freq
            for _ in range(5):                       # warm-up / compile
                sim.step(ticks)
            sim.data.states.pos.block_until_ready()
            t0 = time.perf_counter()
            for _ in range(n_ctrl):
                sim.step(ticks)
            sim.data.states.pos.block_until_ready()
            dt = time.perf_counter() - t0
            row = {"pursuers": P, "worlds": W, "drones": cfg.n_drones,
                   "env_steps_per_s": round(W * n_ctrl / dt),          # one env step = one 20 Hz policy step in one world
                   "drone_physics_ticks_per_s": round(W * cfg.n_drones * n_ctrl * ticks / dt)}
            rows.append(row)
            print(f"      P={P:<2d} W={W:<6d} {row['env_steps_per_s']:>12,d} env steps/s")
    return {"rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--n-pursuers", type=int, default=3)
    ap.add_argument("--sweep", action="store_true", help="throughput over more world counts and team sizes")
    args = ap.parse_args()
    cfg = GameConfig(n_pursuers=args.n_pursuers)
    RESULTS["args"] = vars(args)

    check_versions()
    check_spawn(args.device, cfg)
    check_hover(args.device, cfg)
    check_masked_reset(args.device, cfg)
    check_per_drone_mass(args.device, cfg)
    check_contacts(args.device, cfg)
    if args.sweep:
        check_throughput(args.device, [1024, 4096, 16384], [2, 3, 4, 8])
    else:
        check_throughput(args.device, [256, 1024], [args.n_pursuers])

    out = Path("results") / f"spike_{socket.gethostname()}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(RESULTS, indent=2))
    n_ok = sum(c["ok"] for c in RESULTS["checks"].values())
    print(f"\n{n_ok}/{len(RESULTS['checks'])} checks passed. Saved {out}")


if __name__ == "__main__":
    main()
