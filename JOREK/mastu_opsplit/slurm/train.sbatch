#!/bin/bash
# Explicit requests belong in the reviewed submission command; no historic account defaults.
set -euo pipefail
: "${SLURM_JOB_ID:?Run this script through Slurm}"
: "${SLURM_JOB_ACCOUNT:?Provide an explicit account}"
: "${SLURM_JOB_PARTITION:?Provide an explicit partition}"
: "${STUDY_ROOT:?Set absolute study root}"
: "${PYTHON:?Set an absolute Python interpreter}"
: "${TRAIN_CONFIG:?Set training config}"
: "${MODEL_CONFIG:?Set pinned model config}"
: "${OUTPUT:?Set a fresh output directory}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
exec "$PYTHON" "$STUDY_ROOT/scripts/train.py" --config "$TRAIN_CONFIG" --model-config "$MODEL_CONFIG" --output "$OUTPUT"
