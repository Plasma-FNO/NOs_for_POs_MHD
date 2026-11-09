#!/bin/bash
#SBATCH --job-name=mastu-geometry-quality
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
set -euo pipefail
source /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/native_wall_mesh_20261103/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/native_wall_mesh_20261103/mesh_quality_driver.py /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/native_wall_mesh_20261103 /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit
