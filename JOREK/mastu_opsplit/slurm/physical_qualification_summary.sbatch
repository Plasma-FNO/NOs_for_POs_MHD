#!/bin/bash
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
# Submit with afterany dependency so failed/incomplete runs remain visible.
set -euo pipefail
: "${STUDY_ROOT:?Set frozen study snapshot}"
: "${OUTPUT_ROOT:?Set qualification campaign directory}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$OUTPUT_ROOT"
sha256sum --check --quiet campaign.sha256
cd "$STUDY_ROOT"
sha256sum --check --quiet source.sha256
exec /pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python scripts/summarize_physical_qualification.py \
    --campaign "$OUTPUT_ROOT/campaign.json" --output "$OUTPUT_ROOT/qualification.json"
