#!/bin/bash
#SBATCH --job-name=mastu-contour-bounds
#SBATCH --account=fupa2_fauna
#SBATCH --partition=dcgp_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=00:20:00
set -euo pipefail
source /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/JOREK/mastu_opsplit/runs/prepared_equilibrium_dcgp_authorized_20261102/jorek-env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/contour_bounds_debug_20261102/source
make -j2 .obj/reorder_and_clean_flux_surfaces.o F90FLAGS='-g -fcheck=bounds -fbacktrace' > /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/contour_bounds_debug_20261102/build.out 2> /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/contour_bounds_debug_20261102/build.err
make -j2 jorek_model303 >> /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/contour_bounds_debug_20261102/build.out 2>> /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/contour_bounds_debug_20261102/build.err
sha256sum jorek_model303 > /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/contour_bounds_debug_20261102/binary.sha256
cd /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/contour_bounds_debug_20261102/work
set +e
timeout --signal=TERM --kill-after=10s 1050s mpirun --bind-to none -n 4 /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/contour_bounds_debug_20261102/source/jorek_model303 < input > /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/contour_bounds_debug_20261102/stdout.log 2> /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/contour_bounds_debug_20261102/stderr.log
result=$?
printf '%s\n' "$result" > /pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/contour_bounds_debug_20261102/returncode.txt
exit "$result"
