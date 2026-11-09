#!/bin/bash
#SBATCH --job-name=gos-followup
#SBATCH --account=fupb2_ukaea_ml
#SBATCH --partition=boost_fua_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
# SPECS: ';'-separated "arm [flags]" entries run in order for one PROBLEM/SEED.
set -uo pipefail
cd /pitagora/home/userexternal/vgopakum/NOs_for_POs_MHD/GenericOpsSplit
PY=/pitagora/home/userexternal/vgopakum/NOs_for_POs/.venv/bin/python
IFS=';' read -ra ENTRIES <<< "$SPECS"
for entry in "${ENTRIES[@]}"; do
  read -ra a <<< "$entry"
  $PY -m gos.train run --problem "$PROBLEM" --seed "$SEED" --root "$ROOT" --skip-existing --arm "${a[@]}" || echo "FAILED $entry"
done
