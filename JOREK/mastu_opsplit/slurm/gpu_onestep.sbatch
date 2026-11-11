#!/bin/bash
#SBATCH --account=fupb2_fauna
#SBATCH --partition=boost_fua_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --requeue
# GPU campaign: every model takes one state to the next and is trained/evaluated on one H100.
# One array task per (model, seed) runs, in order and resumably:
#   1. training (2,000 four-step rollout updates; implicit adds 2,000 label updates) + held-out test evaluation
#   2. fixed-conditioning D_perp OOD evaluation (network conditioning held at its training value 1)
#   3. conditional only: OOD with the true D_perp/D0 supplied to FiLM (extrapolation from 1)
#   4. implicit arms only: OOD with the true D_perp in the frozen implicit physics (coefficient physics)
# Array map (seeds 17-19): 0-2 ar, 3-5 node, 6-8 opsplit, 9-11 conditional,
#                          12-14 implicit, 15-17 implicit_rollout, 18 physics (seed 17; nothing trained).
# SMOKE=1 runs the six-update smoke config for FAMILY (default conditional) without OOD.
set -uo pipefail
: "${STUDY_ROOT:?Set frozen study snapshot}"
: "${OUTPUT_ROOT:?Set experiment directory}"
PYTHON=/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python
OOD_ROOT=/pitagora_scratch/userexternal/vgopakum/mastu_coefficient_ood10_20261106
LOADER=/pitagora_scratch/userexternal/vgopakum/mastu_density_diffusion_seed17_20261106T163039Z/study/src/mastu_opsplit/coefficient_ood.py
NATIVE_CHECK=/pitagora_scratch/userexternal/vgopakum/mastu_implicit_opsplit_20261108/coefficient/verify_native.json
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$STUDY_ROOT" || exit 1
sha256sum --check --quiet source.sha256 || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ "${SMOKE:-0}" == 1 ]]; then
    family=${FAMILY:-conditional}
    exec 9>"$OUTPUT_ROOT/smoke_${family}.lock"; flock -n 9 || exit 1
    if [[ $family == implicit* || $family == physics ]]; then
        exec "$PYTHON" scripts/train_implicit.py --config configs/implicit_opsplit_smoke.yaml --device cuda \
            --model-config configs/lsplinefno.yaml --output "$OUTPUT_ROOT/smoke_${family}" --arm "$family" --resume
    fi
    exec "$PYTHON" scripts/train_ensemble.py --config configs/gpu_onestep_smoke.yaml --device cuda \
        --model-config configs/lsplinefno.yaml --output "$OUTPUT_ROOT/smoke_${family}" --family "$family" --resume
fi

task=${SLURM_ARRAY_TASK_ID:?Submit array 0-18}
families=(ar node opsplit conditional implicit implicit_rollout)
if (( task == 18 )); then family=physics; seed=17
else family=${families[task / 3]}; seed=$((17 + task % 3)); fi
name=seed${seed}_${family}
run="$OUTPUT_ROOT/$name"
exec 9>"$OUTPUT_ROOT/${name}.lock"; flock -n 9 || exit 1
status=0
echo "task $task: $family seed $seed -> $run ($(date -u +%FT%TZ))"

if [[ $family == implicit* || $family == physics ]]; then
    "$PYTHON" scripts/train_implicit.py --config configs/implicit_opsplit.yaml --device cuda \
        --model-config configs/lsplinefno.yaml --output "$run" --arm "$family" --seed "$seed" --resume || status=$?
else
    "$PYTHON" scripts/train_ensemble.py --config configs/gpu_onestep.yaml --device cuda \
        --model-config configs/lsplinefno.yaml --output "$run" --family "$family" --seed "$seed" --resume || status=$?
fi
if [[ ! -f "$run/results.json" ]]; then
    echo "training/test evaluation did not finish (status $status); skipping OOD"; exit 1
fi
echo "test done ($(date -u +%FT%TZ)), status $status"

"$PYTHON" scripts/evaluate_ood_fixed_conditioning.py --run "$run" --family "$family" \
    --output "$OUTPUT_ROOT/ood/$name" --ood-root "$OOD_ROOT" --loader "$LOADER" || status=$?
echo "fixed-conditioning OOD done ($(date -u +%FT%TZ)), status $status"
if [[ $family == conditional ]]; then
    "$PYTHON" scripts/evaluate_ood_fixed_conditioning.py --run "$run" --family conditional --conditioning coefficient \
        --output "$OUTPUT_ROOT/ood_film/$name" --ood-root "$OOD_ROOT" --loader "$LOADER" || status=$?
    echo "true-D FiLM OOD done ($(date -u +%FT%TZ)), status $status"
fi
if [[ $family == implicit* || $family == physics ]]; then
    "$PYTHON" scripts/evaluate_ood_coefficient_physics.py --run "$run" --family "$family" --device cuda \
        --output "$OUTPUT_ROOT/ood_coefficient/$name" --coefficient-module src/mastu_opsplit/implicit_coefficient.py \
        --native-check "$NATIVE_CHECK" --ood-root "$OOD_ROOT" --loader "$LOADER" || status=$?
    echo "coefficient-physics OOD done ($(date -u +%FT%TZ)), status $status"
fi
exit $status
