#!/bin/bash
#SBATCH --job-name=mastu-opsplit
#SBATCH --account=fupb2_ukaea_ml
#SBATCH --partition=boost_fua_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=8G
set -euo pipefail
source /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit/runs/prepared_equilibrium_20261102_v3/jorek-env.sh
/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit/scripts/run_case.py --config /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit/configs/equilibrium_request.yaml --prepared /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit/runs/prepared_equilibrium_20261102_v3 --output /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit/runs/equilibrium_regression_20261102
