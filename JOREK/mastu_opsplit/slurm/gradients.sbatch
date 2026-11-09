#!/bin/bash
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --job-name=mastu-gradient-training
set -euo pipefail
: "${STUDY_ROOT:?Set frozen study snapshot}"
: "${OUTPUT_ROOT:?Set experiment directory}"
PYTHON=/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
cd "$STUDY_ROOT"
sha256sum --check --quiet source.sha256
task=${SLURM_ARRAY_TASK_ID:?Submit array 0-5}
[[ "$task" =~ ^[0-5]$ ]] || exit 2
variants=(learned native)
branches=(lspline_continuity native_continuity)
variant=${variants[$((task % 2))]}
branch=${branches[$((task % 2))]}
seed=$((17 + task / 2))
config=configs/ensemble_training.yaml
if [[ "${SMOKE:-0}" == 1 ]]; then
    config=configs/ensemble_smoke.yaml
    OUTPUT_ROOT="$OUTPUT_ROOT/smoke"
    mkdir -p "$OUTPUT_ROOT"
fi
exec 9>"$OUTPUT_ROOT/seed${seed}_${variant}.lock"
flock -n 9
exec "$PYTHON" scripts/train_ensemble.py --config "$config" \
    --model-config configs/lsplinefno.yaml --output "$OUTPUT_ROOT/seed${seed}_${variant}" \
    --seed "$seed" --family opsplit --linear-branch "$branch" --resume
