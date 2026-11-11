#!/bin/bash
#SBATCH --job-name=gos-research
#SBATCH --account=fupb2_ukaea_ml
#SBATCH --partition=boost_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
cd /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/GenericOpsSplit
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python
# One whitespace-delimited argument list per task; no shell evaluation of manifest content.
read -ra arguments <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MANIFEST")"
test "${#arguments[@]}" -gt 0
"$PY" -m "${MODULE:-gos.research}" run --root "$ROOT" "${arguments[@]}"
