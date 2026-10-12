#!/bin/bash
# Empire AI: run the Crazyflow spike on one GPU.
#
# Before the first run, fill in PARTITION and ACCOUNT for your institution:
#   sacctmgr show assoc user=$USER format=User,Account   # your account name
#   sinfo                                                 # partitions you can use
# Default walltime on the Alpha cluster is 1 hour, so set --time explicitly.
#
# Submit from the repo root:   sbatch cluster/spike.sbatch
# Watch it:                    squeue -u $USER ; tail -f slurm-<jobid>.out

#SBATCH --job-name=drague-spike
#SBATCH --partition=PARTITION
#SBATCH --account=ACCOUNT
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# uv installs into .venv in the repo; install it once with:
#   curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra gpu --extra dev

nvidia-smi --query-gpu=name,memory.total --format=csv
uv run pytest -q
uv run python scripts/spike_crazyflow.py --device gpu --sweep
