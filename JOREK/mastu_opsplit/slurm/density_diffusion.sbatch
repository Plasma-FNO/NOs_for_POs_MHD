#!/bin/bash
#SBATCH --job-name=mastu-density-diffusion
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --requeue
set -euo pipefail
: "${STUDY_ROOT:?Set frozen study snapshot}"
: "${OUTPUT_ROOT:?Set fresh experiment directory}"
PYTHON=/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
cd "$STUDY_ROOT"
sha256sum --check --quiet source.sha256
config=configs/ensemble_density_diffusion.yaml
output="$OUTPUT_ROOT/seed17_opsplit"
if [[ "${SMOKE:-0}" == 1 ]]; then
    config=configs/ensemble_density_diffusion_smoke.yaml
    output="$OUTPUT_ROOT/smoke"
fi
exec 9>"${output}.lock"
flock -n 9
exec "$PYTHON" scripts/train_ensemble.py --config "$config" \
    --model-config configs/lsplinefno.yaml --family opsplit --seed 17 \
    --output "$output" --resume
