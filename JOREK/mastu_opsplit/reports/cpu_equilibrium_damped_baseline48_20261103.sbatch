#!/bin/bash
#SBATCH --job-name=mastu-damped-baseline48
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
set -euo pipefail
source /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/spatial_damped_pair_20261103/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/equilibrium_damped_baseline48_20261103/worker.py /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/equilibrium_damped_baseline48_20261103 /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit
