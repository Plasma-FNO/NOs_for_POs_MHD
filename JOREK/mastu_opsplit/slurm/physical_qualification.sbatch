#!/bin/bash
#SBATCH --account=fupb2_fauna
#SBATCH --partition=boost_fua_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --requeue
# Tasks 0-2: fresh learned seeds 17-19. Task 3: shared deterministic physics control.
set -euo pipefail
: "${STUDY_ROOT:?Set frozen study snapshot}"
: "${OUTPUT_ROOT:?Set qualification campaign directory}"
PYTHON=/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
cd "$OUTPUT_ROOT"
sha256sum --check --quiet campaign.sha256
cd "$STUDY_ROOT"
sha256sum --check --quiet source.sha256
case "${SLURM_ARRAY_TASK_ID:?Submit array 0-3}" in
    0|1|2) arm=implicit_rollout; seed=$((17 + SLURM_ARRAY_TASK_ID));;
    3) arm=physics; seed=17;;
    *) echo 'Expected array task 0, 1, 2 or 3' >&2; exit 2;;
esac
name=seed${seed}_${arm}
exec 9>"$OUTPUT_ROOT/${name}.lock"
flock -n 9
"$PYTHON" - <<'PROBE'
import resource, torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
limit = resource.getrlimit(resource.RLIMIT_CPU)[0]
assert limit == resource.RLIM_INFINITY or limit > 12300, limit
print(dict(device=torch.cuda.get_device_name(0), torch=str(torch.__version__), cpu_limit=limit), flush=True)
PROBE
exec "$PYTHON" scripts/train_physical_defect.py \
    --config "$OUTPUT_ROOT/run_config.yaml" --model-config configs/lsplinefno.yaml \
    --output "$OUTPUT_ROOT/$name" --arm "$arm" --seed "$seed" --resume \
    --setup-cache /pitagora_scratch/userexternal/vgopakum/mastu_gpu_onestep_20261109/seed17_physics \
    --prepared-data-cache "$OUTPUT_ROOT/${name}_prepared" --cpu-budget 12000
