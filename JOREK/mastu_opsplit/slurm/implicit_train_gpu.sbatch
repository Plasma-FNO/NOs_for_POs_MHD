#!/bin/bash
#SBATCH --account=fupb2_fauna
#SBATCH --partition=boost_fua_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --requeue
# GPU version of implicit_train.sbatch (torch backend, cuDSS LU, deterministic): same arms, seeds and outputs.
# Array over arms/seeds: 0-2 implicit (seeds 17-19), 3-5 implicit_rollout (17-19), 6 physics.
set -euo pipefail
: "${STUDY_ROOT:?Set frozen study snapshot}"
: "${OUTPUT_ROOT:?Set experiment directory}"
PYTHON=/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$STUDY_ROOT"
sha256sum --check --quiet source.sha256
if [[ "${SMOKE:-0}" == 1 ]]; then
    arm=${ARM:-implicit}
    exec 9>"$OUTPUT_ROOT/smoke_${arm}.lock"
    flock -n 9
    exec "$PYTHON" scripts/train_implicit.py --config configs/implicit_opsplit_smoke.yaml --device cuda \
        --model-config configs/lsplinefno.yaml --output "$OUTPUT_ROOT/smoke_${arm}" --arm "$arm" --resume
fi
task=${SLURM_ARRAY_TASK_ID:?Submit array 0-6}
if (( task == 6 )); then arm=physics; seed=17
elif (( task >= 3 )); then arm=implicit_rollout; seed=$((17 + task - 3))
else arm=implicit; seed=$((17 + task)); fi
name=seed${seed}_${arm}
exec 9>"$OUTPUT_ROOT/${name}.lock"
flock -n 9
exec "$PYTHON" scripts/train_implicit.py --config configs/implicit_opsplit.yaml --device cuda \
    --model-config configs/lsplinefno.yaml --output "$OUTPUT_ROOT/${name}" --arm "$arm" --seed "$seed" --resume
