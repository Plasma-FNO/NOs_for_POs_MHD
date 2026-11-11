#!/bin/bash
#SBATCH --account=fupb2_fauna
#SBATCH --partition=boost_fua_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --requeue
# Larger remainder network and learned tangent correction Delta L for the implicit OpsSplit (one H100 per task).
# Array map (seeds 17-19):
#    0-2 implicit_large     / implicit_rollout     larger N only
#    3-5 implicit_large     / implicit             larger N, exact-label pretraining + rollout
#    6-8 implicit_dl        / implicit_rollout     Delta L with the original small N
#   9-11 implicit_large_dl  / implicit_rollout     larger N + Delta L
# References (same protocol, small N): mastu_gpu_onestep_20261109 seed*_implicit{,_rollout}, seed17_physics.
# Each task: train + test, fixed-conditioning D_perp OOD, coefficient-physics OOD (Delta L kept as trained).
# The training-only Dirichlet/wall-row caches are copied from the GPU campaign (validated by case ids).
# SMOKE=1 VARIANT=... ARM=... runs the matching *_smoke.yaml.
set -uo pipefail
: "${STUDY_ROOT:?Set frozen study snapshot}"
: "${OUTPUT_ROOT:?Set experiment directory}"
PYTHON=/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python
CACHE=/pitagora_scratch/userexternal/vgopakum/mastu_gpu_onestep_20261109/seed17_implicit_rollout
OOD_ROOT=/pitagora_scratch/userexternal/vgopakum/mastu_coefficient_ood10_20261106
LOADER=/pitagora_scratch/userexternal/vgopakum/mastu_density_diffusion_seed17_20261106T163039Z/study/src/mastu_opsplit/coefficient_ood.py
NATIVE_CHECK=/pitagora_scratch/userexternal/vgopakum/mastu_implicit_opsplit_20261108/coefficient/verify_native.json
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$STUDY_ROOT" || exit 1
sha256sum --check --quiet source.sha256 || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

seed_cache() { mkdir -p "$1"; for f in fixed_dofs.npz wall_rows.npz; do [[ -f "$1/$f" ]] || cp "$CACHE/$f" "$1/$f"; done; }

if [[ "${SMOKE:-0}" == 1 ]]; then
    variant=${VARIANT:-implicit_large_dl}; arm=${ARM:-implicit_rollout}; run="$OUTPUT_ROOT/smoke_${variant}_${arm}"
    exec 9>"$run.lock"; flock -n 9 || exit 1
    seed_cache "$run"
    exec "$PYTHON" scripts/train_implicit.py --config "configs/${variant}_smoke.yaml" --device cuda \
        --model-config configs/lsplinefno.yaml --output "$run" --arm "$arm" --resume
fi

task=${SLURM_ARRAY_TASK_ID:?Submit array 0-11}
variants=(implicit_large implicit_large implicit_dl implicit_large_dl)
arms=(implicit_rollout implicit implicit_rollout implicit_rollout)
variant=${variants[task / 3]}; arm=${arms[task / 3]}; seed=$((17 + task % 3))
name=seed${seed}_${variant}_${arm}
run="$OUTPUT_ROOT/$name"
exec 9>"$OUTPUT_ROOT/${name}.lock"; flock -n 9 || exit 1
status=0
echo "task $task: $variant / $arm seed $seed -> $run ($(date -u +%FT%TZ))"
seed_cache "$run"
"$PYTHON" scripts/train_implicit.py --config "configs/${variant}.yaml" --device cuda \
    --model-config configs/lsplinefno.yaml --output "$run" --arm "$arm" --seed "$seed" --resume || status=$?
if [[ ! -f "$run/results.json" ]]; then
    echo "training/test evaluation did not finish (status $status); skipping OOD"; exit 1
fi
echo "test done ($(date -u +%FT%TZ)), status $status"
"$PYTHON" scripts/evaluate_ood_fixed_conditioning.py --run "$run" --family "$arm" \
    --output "$OUTPUT_ROOT/ood/$name" --ood-root "$OOD_ROOT" --loader "$LOADER" || status=$?
echo "fixed-conditioning OOD done ($(date -u +%FT%TZ)), status $status"
"$PYTHON" scripts/evaluate_ood_coefficient_physics.py --run "$run" --family "$arm" --device cuda \
    --output "$OUTPUT_ROOT/ood_coefficient/$name" --coefficient-module src/mastu_opsplit/implicit_coefficient.py \
    --native-check "$NATIVE_CHECK" --ood-root "$OOD_ROOT" --loader "$LOADER" || status=$?
echo "coefficient-physics OOD done ($(date -u +%FT%TZ)), status $status"
exit $status
