#!/bin/bash
#SBATCH --job-name=mastu-balanced-spatial-probe
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=512G
#SBATCH --time=06:00:00
set -euo pipefail
source /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/combined_balanced2_20261104/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/combined_balanced2_20261104/source
timeout --signal=TERM --kill-after=10s 1200s make -j2 jorek_model303 > /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/combined_balanced2_20261104/build.out 2> /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/combined_balanced2_20261104/build.err
/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/combined_balanced2_20261104/worker.py /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/combined_balanced2_20261104 /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit control3
/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/combined_balanced2_20261104/worker.py /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/combined_balanced2_20261104 /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit probe3
