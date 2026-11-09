#!/bin/bash
#SBATCH --job-name=mastu-density-diffusion-ood
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --requeue
set -euo pipefail
: "${STUDY_ROOT:?Set frozen study snapshot}"
: "${OUTPUT_ROOT:?Set experiment directory}"
: "${OOD_ROOT:?Set completed ten-case OOD campaign}"
PYTHON=/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
cd "$STUDY_ROOT"
sha256sum --check --quiet source.sha256
exec 9>"$OUTPUT_ROOT/seed17_opsplit_ood.lock"
flock -n 9
normalization_sha256=$(sha256sum "$OUTPUT_ROOT/seed17_opsplit/normalization.npz" | cut -d' ' -f1)
exec "$PYTHON" scripts/evaluate_density_diffusion_ood.py \
    --training "$OUTPUT_ROOT/seed17_opsplit" --ood-root "$OOD_ROOT" \
    --training-normalization-sha256 "$normalization_sha256" \
    --output "$OUTPUT_ROOT/seed17_opsplit_ood"
