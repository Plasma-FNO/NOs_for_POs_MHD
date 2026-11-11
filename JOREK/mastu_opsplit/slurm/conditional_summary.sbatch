#!/bin/bash
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
set -euo pipefail
: "${STUDY_ROOT:?Set frozen study snapshot}"
: "${OUTPUT_ROOT:?Set conditional campaign directory}"
: "${ENSEMBLE_ROOT:?Set existing ensemble campaign}"
: "${GRADIENT_ROOT:?Set existing gradient campaign}"
: "${IMPLICIT_ROOT:?Set existing implicit campaign}"
export OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$STUDY_ROOT"
sha256sum --check --quiet source.sha256
exec /pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python scripts/summarize_conditional.py \
    --root "$OUTPUT_ROOT" --ensemble "$ENSEMBLE_ROOT" --gradient "$GRADIENT_ROOT" \
    --implicit "$IMPLICIT_ROOT" --output "$OUTPUT_ROOT/comparison.json"
