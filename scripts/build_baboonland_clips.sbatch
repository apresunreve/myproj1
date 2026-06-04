#!/bin/bash
#SBATCH -M notchpeak
#SBATCH -p soc-np
#SBATCH --account=soc-np
#SBATCH --job-name=bbl_clips
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o /scratch/general/vast/j.yan/myproj1/logs/build_baboonland_clips.%j.out
#SBATCH -e /scratch/general/vast/j.yan/myproj1/logs/build_baboonland_clips.%j.err

set -euo pipefail

REPO=/scratch/general/vast/j.yan/myproj1
PY="$REPO/.venv/bin/python"

echo "host: $(hostname)"
echo "date: $(date)"
echo "python: $("$PY" --version)"

"$PY" "$REPO/scripts/build_baboonland_clips.py" \
    --source_root /scratch/general/nfs1/u1320844/baboon2/charades \
    --output_root "$REPO/baboonland_files"

echo "done: $(date)"
