# Simulation logbook

Each simulation run writes one directory under `results/`. The directory is the source for a paper table row, a learning curve, an outcome breakdown, and a demo clip. Training code (self-play, PFSP, league) should call `RunLog` in `src/drague/logbook.py` instead of inventing another file format.

`results/` is gitignored. `results/index.jsonl` lists every run on this machine.

## Run a scripted match

```bash
uv sync --extra viz --extra dev
uv run python scripts/record_rollout.py --pursuer standoff --evader flee --n-pursuers 3 --seed 0
uv run python scripts/render_episode.py --run results/<run_id> --episode 0
```

`--pursuer` and `--evader` are `standoff`, `flee`, or `idle`. The matchups in `EXPERIMENTS.md` are standoff vs idle, standoff vs flee, and idle vs flee, at `--n-pursuers` 2, 3, 4, and 8. One command records one cell.

`--video-episodes first` (the default) writes a clip of the first saved episode. `all` writes every saved episode. `none` skips the video. Replay a later episode without simulating again:

```bash
uv run python scripts/render_episode.py --run results/<run_id> --episode 3
```

The clip is real time (20 fps, one frame per 50 ms policy step). It is an mp4 when `ffmpeg` is on `PATH`, and a gif otherwise. Install ffmpeg with `brew install ffmpeg` if you want mp4.

## Directory layout

```
results/
  index.jsonl
  <run_id>/
    manifest.json
    metrics.jsonl
    episodes.jsonl
    eval/<step>.json
    trajectories/<step>.npz
    trajectories/<step>.json
    videos/ep<id>.mp4
    summary.json
```

`<step>` is the checkpoint env-step, nine digits. Scripted runs use `000000000`. `<run_id>` looks like `scripted_standoff_vs_flee_p3_s0_20261111T193000Z`.

`method` is `scripted`, `selfplay`, `pfsp`, or `league`. Keys are snake_case. Fractions are in `[0, 1]`.

## What to cite for which claim

| File | Paper use |
|---|---|
| `manifest.json` | Setup: code version, hardware, full `GameConfig`, policies, seed, backend, compute budget |
| `metrics.jsonl` | Learning curves versus env steps. Scripted runs have one line per interval and are flat |
| `summary.json` and `index.jsonl` | The results table: one row per run, with mean and standard error |
| `episodes.jsonl` | Failure analysis: how episodes ended, crashes, reward parts, distances |
| `eval/<step>.json` | Held-out opponents, and later cross-play and exploitability |
| `trajectories/*.npz` | Figures of paths, splits, and crashes |
| `videos/ep<id>.*` | Demo and qualitative examples |

## Manifest

Written at the start and updated when the run closes.

- `schema_version`, `run_id`, `started_at`, `ended_at`, `wall_s`, `command`
- `git_commit`, `git_dirty` (a dirty tree is recorded as dirty)
- `host`, `platform`, `device`, `jax_version`, `drague_version`
- `method`, `seed`, `backend` (`point_mass` or, later, `crazyflow`), `n_worlds`, `n_episodes`
- `pursuer_policy`, `evader_policy` (function name, or a checkpoint id once training exists)
- `game`: every field of `GameConfig`
- `budget.env_steps_main`, `budget.env_steps_exploiter`, `budget.env_steps_total` (exploiter steps count toward the equal-compute budget)
- `hparams`: algorithm settings; `{}` for scripted runs
- `videos`: filenames written into `videos/`

## Episode record

One JSON object per line in `episodes.jsonl`. This is the dataset to analyze. Outcomes use the priority in `rules.resolve_outcome`: `evader_crashed`, then `pursuers_wiped`, then `timeout_win` or `timeout_loss`.

- `episode_id`, `world`, `env_steps_at_end`
- `outcome`, `pursuers_win`
- `length_steps`, `tracked_steps`, `tracked_fraction` (`tracked_steps / max_steps`, the timeout rule), `time_to_first_track_steps` (null if the evader was never tracked)
- `pursuer_oob`, `pursuer_collision`, `evader_oob` (a pursuer who leaves the arena and hits a teammate on the same step increments both pursuer counters)
- `return_tracking`, `return_shaping`, `return_crash`, `return_terminal`, `return_total` (shaping and crash are already negative; the four parts sum to `return_total`)
- `mean_d_min_m`, `final_d_min_m`, `min_pursuer_separation_m` (null when fewer than two pursuers were ever alive together)
- `pursuer_speed_saturated`, `evader_speed_saturated`: fraction of live drone-steps within 1% of that role's speed cap

## Metrics curve

`metrics.jsonl` appends every `--metrics-every` episodes and once at the end. `env_steps`, `episodes`, and `wall_s` are cumulative. The unsuffixed rates are the mean over the episodes since the previous line, and `*_se` is the standard error of that interval. `cumulative_pursuer_win_rate` and `cumulative_tracked_fraction` run from the start of the run. `env_steps_per_s` is cumulative env steps divided by wall time.

When a trainer exists, the same line gains `policy_loss`, `value_loss`, `entropy`, and for league or PFSP the opponent id and its sampling weight.

## Summary and index

`summary.json` is the mean over every episode, plus `*_se`, `n_episodes`, `outcome_counts`, `never_tracked_fraction`, and the mean `time_to_first_track_steps` among episodes that did track. `index.jsonl` stores `run_id`, `method`, `n_pursuers`, `seed`, both policies, `pursuer_win_rate`, `tracked_fraction`, `env_steps_total`, and `path`.

## Trajectories

`trajectories/<step>.npz` holds a sample of episodes (default 8):

- `pos`, `vel`, `action`: `(E, T, D, 3)`
- `alive`, `crashed_oob`, `crashed_collision`: `(E, T, D)`
- `tracked`: `(E, T)`
- `length_frames`: `(E,)` valid prefix. Frame 0 is the spawn; later frames are the state after each policy step. `T` is `max_steps + 1`, and frames past `length_frames` repeat the last pose.

The `.json` sidecar lists `episode_ids`, `worlds`, `outcomes`, `pursuers_win`, `length_frames`, and `length_steps`. Full observations are not stored; they are a function of `pos`, `vel`, and `alive`.

## Eval block

`eval/<step>.json` for a scripted run is that matchup: `n_episodes`, policies, `pursuer_win_rate`, `tracked_fraction`, and `outcome_counts`. A timeout win and an evader-crash win are different results, so the histogram is the number to report alongside the win rate.

Training runs will add, in the same file:

- Held-out evaders `idle` and `flee`, and the `standoff` pursuer baseline against `flee`.
- A cross-play matrix of saved pursuer checkpoints against saved evader checkpoints.
- Exploitability: opponent `fresh_exploiter`, its step budget, its win rate, and its tracked fraction against the frozen team.

## Demo video

`src/drague/video.py` draws the arena and the spawn ring, blue pursuers, an orange evader, a one-second trail, and a green dashed ring of `tracking_radius` on tracked frames. A crashed drone (from `alive`, `crashed_oob`, or `crashed_collision`) stops and is drawn as an x. The title shows time, whether the current frame is tracked, the running tracked fraction, and the outcome on the last frame. The footer is the run id and the two policy names.
