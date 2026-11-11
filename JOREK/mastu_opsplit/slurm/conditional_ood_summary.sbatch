#!/bin/bash
#SBATCH --job-name=mastu-conditional-ood-summary
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
set -euo pipefail
: "${CAMPAIGN_ROOT:?Set trained conditional campaign}" "${OUTPUT_ROOT:?Set OOD campaign}"
export PYTHONDONTWRITEBYTECODE=1
cd "$OUTPUT_ROOT"
sha256sum --check --quiet summary.sha256
exec /pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python summarize_conditional_ood.py \
    --root "$OUTPUT_ROOT" --training-root "$CAMPAIGN_ROOT" --output "$OUTPUT_ROOT/comparison.json"
