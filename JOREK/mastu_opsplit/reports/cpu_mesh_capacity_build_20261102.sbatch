#!/bin/bash
#SBATCH --job-name=mastu-cpu-mesh-build
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_dbg
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
source /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit/runs/prepared_equilibrium_dcgp_authorized_20261102/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/mesh_capacity_build_20261102/source
./util/config.sh model=303 n_tor=1 n_coord_tor=1 l_pol_domm=0 n_plane=1 n_period=1 n_coord_period=1 n_boundary_max=4001
make -j2 jorek_model303
sha256sum jorek_model303 models/mod_settings.f90 models/model303/mod_model_settings.f90
