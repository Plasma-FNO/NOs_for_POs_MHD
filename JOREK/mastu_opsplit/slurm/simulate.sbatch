#!/bin/bash
# Intentionally no historic account or partition. Render a concrete request with:
# python scripts/run_case.py --config CONFIG --prepared CASE --output RUN --render-slurm
# Review that request and obtain required resource authorization before sbatch.
set -euo pipefail
printf '%s\n' 'Render the explicit, capped request with scripts/run_case.py --render-slurm.' >&2
exit 2
