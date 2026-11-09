#!/bin/bash
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=240G
#SBATCH --time=04:00:00
# Native export workers: CONCURRENCY exports x 4 MPI ranks per array task.
set -euo pipefail
: "${STUDY_ROOT:?Set frozen study snapshot}"
: "${LABEL_ROOT:?Set label root containing plan.json}"
: "${TASK_COUNT:?Set the array size}"
CONCURRENCY=${CONCURRENCY:-16}
PYTHON=/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python
source /pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/mastu_full_opsplit_20261104/prepared_tangent/jorek-env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$STUDY_ROOT"
sha256sum --check --quiet source.sha256
exec "$PYTHON" scripts/export_remainder_labels.py worker --root "$LABEL_ROOT" \
    --scratch "$LABEL_ROOT/work" --task-count "$TASK_COUNT" --concurrency "$CONCURRENCY"
