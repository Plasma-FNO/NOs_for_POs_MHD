#!/bin/bash
#SBATCH --job-name=mastu-node-coefficient-ood
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --requeue
set -euo pipefail
: "${STUDY_ROOT:?Set evaluator snapshot}"
: "${TRAINING_STUDY:?Set original frozen training study}"
: "${TRAINING_RUN:?Set original seed17_node output}"
: "${LOADER_STUDY:?Set frozen coefficient-aware study}"
: "${OUTPUT_ROOT:?Set fresh evaluation experiment}"
: "${OOD_ROOT:?Set accepted ten-case diffusivity OOD release}"
PYTHON=/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TRAINING_STUDY/src"
for snapshot in "$STUDY_ROOT" "$TRAINING_STUDY" "$LOADER_STUDY"; do
    (cd "$snapshot" && sha256sum --check --quiet source.sha256)
done
options=()
output="$OUTPUT_ROOT/seed17_node_ood"
if [[ "${SMOKE:-0}" == 1 ]]; then
    options+=(--smoke)
    output="$OUTPUT_ROOT/smoke"
fi
exec 9>"${output}.lock"
flock -n 9
normalization_sha256=$(sha256sum "$TRAINING_RUN/normalization.npz" | cut -d' ' -f1)
exec "$PYTHON" "$STUDY_ROOT/scripts/evaluate_node_coefficient_ood.py" \
    --training "$TRAINING_RUN" --loader-study "$LOADER_STUDY" --ood-root "$OOD_ROOT" \
    --training-normalization-sha256 "$normalization_sha256" --output "$output" "${options[@]}"
