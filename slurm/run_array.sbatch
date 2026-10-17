#!/bin/bash
# Run one line of a job file per array task. Usage (from the repo root):
#   N=$(wc -l < experiments/r1.txt)
#   sbatch --array=0-$((N-1))%4 slurm/run_array.sbatch experiments/r1.txt
# Finished jobs are skipped, so after a preemption just submit the same command again.
#SBATCH --job-name=owl
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
JOBFILE=${1:?job file}
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOBFILE")
[ -z "$LINE" ] && { echo "no line $SLURM_ARRAY_TASK_ID in $JOBFILE"; exit 0; }

# keep model weights and datasets out of the home quota
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}
source .venv/bin/activate

echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "args $LINE"
python -m owlrepro.run $LINE
