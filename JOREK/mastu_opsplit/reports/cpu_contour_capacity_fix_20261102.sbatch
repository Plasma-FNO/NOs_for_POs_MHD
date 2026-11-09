#!/bin/bash
#SBATCH --job-name=mastu-contour-fix
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
set -euo pipefail
source /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit/runs/prepared_equilibrium_dcgp_authorized_20261102/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/contour_capacity_fix_20261102/source
make -j2 .obj/reorder_and_clean_flux_surfaces.o F90FLAGS='-g -fcheck=bounds -fbacktrace' > /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/contour_capacity_fix_20261102/build.out 2> /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/contour_capacity_fix_20261102/build.err
make -j2 jorek_model303 >> /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/contour_capacity_fix_20261102/build.out 2>> /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/contour_capacity_fix_20261102/build.err
sha256sum jorek_model303 > /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/contour_capacity_fix_20261102/binary.sha256
/pitagora_work/FUPB2_UKAEA_ML/vgopakum/envs/cfo/bin/python /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit/reports/cpu_contour_fix_worker_source_20261102.py /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/contour_capacity_fix_20261102
