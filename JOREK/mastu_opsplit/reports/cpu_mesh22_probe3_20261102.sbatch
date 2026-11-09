#!/bin/bash
#SBATCH --job-name=mastu-opsplit
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
set -euo pipefail
source /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit/runs/prepared_cpu_mesh22_probe3_20261102/jorek-env.sh
/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit/scripts/run_case.py --config /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit/configs/cpu_mesh22_probe3_20261102.yaml --prepared /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit/runs/prepared_cpu_mesh22_probe3_20261102 --output /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/cpu_mesh22_probe3_20261102
