#!/bin/bash
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --requeue
set -euo pipefail
: "${STUDY_ROOT:?Set frozen study snapshot}"
: "${OUTPUT_ROOT:?Set experiment directory}"
PYTHON=/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
cd "$STUDY_ROOT"
sha256sum --check --quiet source.sha256
if [[ "${SMOKE:-0}" == 1 ]]; then
    exec 9>"$OUTPUT_ROOT/smoke.lock"
    flock -n 9
    exec "$PYTHON" scripts/train_ensemble.py --config configs/ensemble_smoke.yaml \
        --model-config configs/lsplinefno.yaml --output "$OUTPUT_ROOT/smoke" --resume
fi
families=(ar node opsplit)
task=${SLURM_ARRAY_TASK_ID:?Submit array 0-8}
seed=$((17 + task / 3))
family=${families[$((task % 3))]}
exec 9>"$OUTPUT_ROOT/seed${seed}_${family}.lock"
flock -n 9
exec "$PYTHON" scripts/train_ensemble.py --config configs/ensemble_training.yaml \
    --model-config configs/lsplinefno.yaml --output "$OUTPUT_ROOT/seed${seed}_${family}" \
    --seed "$seed" --family "$family" --resume
