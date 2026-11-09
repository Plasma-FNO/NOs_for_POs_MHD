#!/bin/bash
#SBATCH --job-name=gos-data
#SBATCH --account=fupb2_ukaea_ml
#SBATCH --partition=boost_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
cd /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/GenericOpsSplit
PY=/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python
$PY -m gos.train data --problem "$PROBLEM" --root "$ROOT"
