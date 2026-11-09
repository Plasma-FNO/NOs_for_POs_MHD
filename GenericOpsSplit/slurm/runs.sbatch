#!/bin/bash
#SBATCH --job-name=gos-runs
#SBATCH --account=fupb2_ukaea_ml
#SBATCH --partition=boost_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -uo pipefail
cd /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/GenericOpsSplit
PY=/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python
ARMS=${ARMS:-"imex_exact implicit_known implicit_struct implicit_free implicit_known_cn node ar explicit_known"}
for arm in $ARMS; do
  if [ -f "$ROOT/runs/$PROBLEM/${arm}_s${SEED}/result.json" ]; then echo "skip $arm"; continue; fi
  $PY -m gos.train run --problem "$PROBLEM" --arm "$arm" --seed "$SEED" --root "$ROOT" || echo "FAILED $arm"
done
for arm in ${MDT_ARMS:-}; do
  if [ -f "$ROOT/runs/$PROBLEM/${arm}_mdt_s${SEED}/result.json" ]; then echo "skip ${arm}_mdt"; continue; fi
  $PY -m gos.train run --problem "$PROBLEM" --arm "$arm" --seed "$SEED" --root "$ROOT" --dt-mults 1,2,4 || echo "FAILED ${arm}_mdt"
done
