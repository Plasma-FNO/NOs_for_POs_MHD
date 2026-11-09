#!/bin/bash
#SBATCH --job-name=tear199-fine
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_dbg
#SBATCH --array=0-1%2
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:16:00
#SBATCH --output=/pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/tearing_pilot/growth-%j.out
set -euo pipefail
source /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/tearing_pilot/snapshot/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
pilot_steps=(187.5 93.75)
pilot_dt=${pilot_steps[$SLURM_ARRAY_TASK_ID]}
/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/tearing_pilot/run_growth.py --dt "${pilot_dt}"
