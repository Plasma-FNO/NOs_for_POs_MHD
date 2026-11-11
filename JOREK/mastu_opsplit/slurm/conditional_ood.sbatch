#!/bin/bash
#SBATCH --job-name=mastu-conditional-ood
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail
: "${CAMPAIGN_ROOT:?Set trained conditional campaign}" "${OUTPUT_ROOT:?Set fresh OOD campaign}"
: "${OOD_ROOT:?Set accepted density-diffusivity dataset}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
cd "$OUTPUT_ROOT"
sha256sum --check --quiet evaluator.sha256
task=${SLURM_ARRAY_TASK_ID:?Submit smoke array 0-3 or full array 0-11}
extra=()
if [[ "${SMOKE:-0}" == 1 ]]; then
    if (( task < 0 || task > 3 )); then exit 2; fi
    seed=17
    arm=conditional
    if (( task >= 2 )); then arm=conditional_matched; fi
    extra=(--smoke)
    prefix=smoke_
else
    if (( task < 0 || task > 11 )); then exit 2; fi
    seed=$((17 + task / 4))
    arm=conditional
    if (( task % 4 >= 2 )); then arm=conditional_matched; fi
    prefix=
fi
mode=fixed
if (( task % 2 == 1 )); then mode=coefficient; fi
output="$OUTPUT_ROOT/${prefix}seed${seed}_${arm}_${mode}"
exec 9>"${output}.lock"
flock -n 9
exec /pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python evaluate_ood_fixed_conditioning.py \
    --run "$CAMPAIGN_ROOT/seed${seed}_${arm}" --family conditional --conditioning "$mode" \
    --ood-root "$OOD_ROOT" --loader "$CAMPAIGN_ROOT/study/src/mastu_opsplit/coefficient_ood.py" \
    --output "$output" "${extra[@]}"
