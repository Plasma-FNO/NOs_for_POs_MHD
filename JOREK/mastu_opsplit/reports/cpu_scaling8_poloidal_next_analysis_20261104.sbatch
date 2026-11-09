#!/bin/bash
#SBATCH --job-name=mastu-scaling8-poloidal-analysis
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit/reports/cpu_scaling8_spatial_analysis_20261104.py poloidal_next
