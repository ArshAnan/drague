"""Replay of a saved episode: a top view with a side view directly under it.

Writes an mp4 when ffmpeg is on PATH, otherwise a gif. Matplotlib is optional
(`uv sync --extra viz`); importing this module does not import it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from drague.config import GameConfig

PURSUER_COLOR = "#2176c7"
EVADER_COLOR = "#e07a1f"
TRACK_COLOR = "#2e9b4f"


def _mpl():
    try:
        import matplotlib
        if "agg" not in matplotlib.get_backend().lower():
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
        from matplotlib.patches import Circle, Rectangle
    except ImportError as exc:
        raise ImportError(
            "video rendering needs the viz extra: uv sync --extra viz"
        ) from exc
    return plt, FuncAnimation, FFMpegWriter, PillowWriter, Circle, Rectangle


def _trajectory_file(run_dir: Path, step: int | None) -> Path:
    folder = run_dir / "trajectories"
    if step is not None:
        path = folder / f"{step:09d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    found = sorted(folder.glob("*.npz"))
    if not found:
        raise FileNotFoundError(f"no trajectories in {folder}")
    return found[-1]


def render_episode(
    pos,
    alive,
    tracked,
    crashed_oob,
    crashed_collision,
    length: int,
    cfg: GameConfig,
    out_path: Path | str,
    *,
    run_id: str,
    pursuer_policy: str,
    evader_policy: str,
    outcome: str,
    fps: int | None = None,
    max_frames: int | None = None,
) -> Path:
    """Draw one episode and return the path that was written (mp4, or gif as a fallback)."""
    plt, FuncAnimation, FFMpegWriter, PillowWriter, Circle, Rectangle = _mpl()
    pos = np.asarray(pos)
    alive = np.asarray(alive, dtype=bool)
    tracked = np.asarray(tracked, dtype=bool)
    crashed_oob = np.asarray(crashed_oob, dtype=bool)
    crashed_collision = np.asarray(crashed_collision, dtype=bool)
    length = int(length)
    n_frames = length if max_frames is None else min(length, int(max_frames))
    if n_frames < 1:
        raise ValueError("episode has no frames")

    fps = fps or max(1, int(round(1.0 / cfg.control_dt)))
    n_pursuers = cfg.n_pursuers
    n_drones = pos.shape[1]

    # The side view sits under the top view and shares its x scale, so a drone's
    # left-right position is the same pixel in both panels. The side view looks
    # in the +y direction: up on the top view is away from that camera.
    fig_w, fig_h = 6.7, 9.9
    square, side_h = 5.15, 2.4
    left, side_bottom, gap = 0.95, 0.78, 0.95
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=120)

    def _box(x_in, y_in, w_in, h_in):
        return [x_in / fig_w, y_in / fig_h, w_in / fig_w, h_in / fig_h]

    limit = cfg.arena_radius * 1.2
    ax = fig.add_axes(_box(left, side_bottom + side_h + gap, square, square))
    side = fig.add_axes(_box(left, side_bottom, square, side_h), sharex=ax)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("top view, looking down")
    ax.add_patch(Circle((0, 0), cfg.arena_radius, fill=False, ec="0.25", lw=1.5))
    ax.add_patch(Circle((0, 0), cfg.spawn_ring_radius, fill=False, ec="0.75", lw=0.8, ls="--"))
    ring = Circle((0, 0), cfg.tracking_radius, fill=False, ec=TRACK_COLOR, lw=1.5, ls="--")
    ring.set_visible(False)
    ax.add_patch(ring)

    z_pad = 0.25
    side.set_ylim(cfg.z_min - z_pad, cfg.z_max + z_pad)
    side.set_xlabel("x (m)")
    side.set_ylabel("z (m)")
    side.set_title("side view, looking along +y")
    side.add_patch(Rectangle(
        (-cfg.arena_radius, cfg.z_min), 2 * cfg.arena_radius, cfg.z_max - cfg.z_min,
        fill=False, ec="0.25", lw=1.5,
    ))
    side.axhline(cfg.spawn_height, color="0.75", lw=0.8, ls="--")
    fig.canvas.draw()
    top_box = ax.get_position()
    side_box = side.get_position()
    side.set_position([top_box.x0, side_box.y0, top_box.width, side_box.height])

    dots = []
    crosses = []
    side_dots = []
    side_crosses = []
    for i in range(n_drones):
        color = EVADER_COLOR if i == n_pursuers else PURSUER_COLOR
        dot, = ax.plot([], [], "o", color=color, ms=7)
        cross, = ax.plot([], [], "x", color=color, ms=8, mew=2)
        side_dot, = side.plot([], [], "o", color=color, ms=7)
        side_cross, = side.plot([], [], "x", color=color, ms=8, mew=2)
        dots.append(dot)
        crosses.append(cross)
        side_dots.append(side_dot)
        side_crosses.append(side_cross)
    ax.plot([], [], "o", color=PURSUER_COLOR, ms=7, label="pursuer")
    ax.plot([], [], "o", color=EVADER_COLOR, ms=7, label="evader")
    ax.legend(loc="upper right", frameon=False)
    fig.text(
        0.5, 0.012,
        f"{run_id}    {pursuer_policy} vs {evader_policy}",
        ha="center", va="bottom", fontsize=8,
    )

    def _xy(artist, x, y):
        artist.set_data(np.atleast_1d(x), np.atleast_1d(y))

    def update(frame: int):
        steps_done = min(frame, max(length - 1, 0))
        if steps_done == 0:
            fraction = 0.0
        else:
            fraction = float(tracked[1:steps_done + 1].sum()) / cfg.max_steps
        shown = outcome if frame >= length - 1 else ""
        flag = "yes" if tracked[frame] else "no"
        fig.suptitle(
            f"t={frame * cfg.control_dt:.2f}s   tracked={flag}   "
            f"tracked_fraction={fraction:.2f}   {shown}".rstrip(),
            y=0.985, fontsize=11,
        )
        ev = pos[frame, n_pursuers, :2]
        ring.center = (float(ev[0]), float(ev[1]))
        ring.set_visible(bool(tracked[frame]))
        empty = np.zeros(0)
        for i in range(n_drones):
            crashed = (
                not bool(alive[frame, i])
                or bool(crashed_oob[:frame + 1, i].any())
                or bool(crashed_collision[:frame + 1, i].any())
            )
            if crashed:
                dots[i].set_data(empty, empty)
                side_dots[i].set_data(empty, empty)
                _xy(crosses[i], pos[frame, i, 0], pos[frame, i, 1])
                _xy(side_crosses[i], pos[frame, i, 0], pos[frame, i, 2])
            else:
                _xy(dots[i], pos[frame, i, 0], pos[frame, i, 1])
                _xy(side_dots[i], pos[frame, i, 0], pos[frame, i, 2])
                crosses[i].set_data(empty, empty)
                side_crosses[i].set_data(empty, empty)
        return dots + crosses + side_dots + side_crosses

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    written = _save_animation(anim, Path(out_path), fps, FFMpegWriter, PillowWriter)
    plt.close(fig)
    return written


def _save_animation(anim, path: Path, fps: int, ffmpeg_writer, pillow_writer) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".mp4" and shutil.which("ffmpeg"):
        try:
            anim.save(path, writer=ffmpeg_writer(fps=fps))
            return path
        except Exception:
            path = path.with_suffix(".gif")
    if path.suffix != ".gif":
        path = path.with_suffix(".gif")
    anim.save(path, writer=pillow_writer(fps=fps))
    return path


def render_run(
    run_dir: Path | str,
    *,
    episode: int | None = None,
    indices: list[int] | None = None,
    step: int | None = None,
    max_frames: int | None = None,
) -> list[Path]:
    """Render saved trajectories from a run directory.

    `episode` is an episode id. `indices` selects rows of the sample. The default is the first row.
    """
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    npz_path = _trajectory_file(run_dir, step)
    meta = json.loads(npz_path.with_suffix(".json").read_text(encoding="utf-8"))
    data = np.load(npz_path)
    cfg = GameConfig(**manifest["game"])
    ids = [int(i) for i in meta["episode_ids"]]
    if episode is not None:
        if episode not in ids:
            raise ValueError(f"episode {episode} is not in the saved sample {ids}")
        indices = [ids.index(episode)]
    elif indices is None:
        indices = [0]

    paths = []
    for index in indices:
        if index < 0 or index >= len(ids):
            raise IndexError(f"trajectory index {index} outside 0..{len(ids) - 1}")
        out = run_dir / "videos" / f"ep{ids[index]}.mp4"
        paths.append(render_episode(
            pos=data["pos"][index],
            alive=data["alive"][index],
            tracked=data["tracked"][index],
            crashed_oob=data["crashed_oob"][index],
            crashed_collision=data["crashed_collision"][index],
            length=int(data["length_frames"][index]),
            cfg=cfg,
            out_path=out,
            run_id=manifest["run_id"],
            pursuer_policy=manifest["pursuer_policy"],
            evader_policy=manifest["evader_policy"],
            outcome=meta["outcomes"][index],
            max_frames=max_frames,
        ))
    return paths
