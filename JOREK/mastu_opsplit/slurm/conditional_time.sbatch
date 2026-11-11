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
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
cd "$STUDY_ROOT"
sha256sum --check --quiet source.sha256
task=${SLURM_ARRAY_TASK_ID:?Submit smoke array 0-1 or production array 0-5}
if [[ "${SMOKE:-0}" == 1 ]]; then
    if (( task < 0 || task > 1 )); then exit 2; fi
    arm=conditional
    config=configs/conditional_time.yaml
    if (( task == 1 )); then arm=conditional_matched; config=configs/conditional_time_matched.yaml; fi
    exec 9>"$OUTPUT_ROOT/smoke_${arm}.lock"
    flock -n 9
    exec "$PYTHON" scripts/smoke_conditional.py --config "$config" \
        --model-config configs/lsplinefno.yaml --output "$OUTPUT_ROOT/smoke_${arm}"
fi
if (( task < 0 || task > 5 )); then exit 2; fi
seed=$((17 + task % 3))
arm=conditional
config=configs/conditional_time.yaml
if (( task >= 3 )); then arm=conditional_matched; config=configs/conditional_time_matched.yaml; fi
exec 9>"$OUTPUT_ROOT/seed${seed}_${arm}.lock"
flock -n 9
exec "$PYTHON" scripts/train_ensemble.py --config "$config" \
    --model-config configs/lsplinefno.yaml --output "$OUTPUT_ROOT/seed${seed}_${arm}" \
    --seed "$seed" --family conditional --resume
