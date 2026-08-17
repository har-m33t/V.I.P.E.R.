#!/usr/bin/env bash
# scripts/run_gpu.sh — full VIPER training + evaluation run on a RunPod GPU pod.
#
#   bash scripts/run_gpu.sh            # single GPU
#   NPROC=2 bash scripts/run_gpu.sh    # 2-GPU DistributedDataParallel
#
# Hyperparameters come from src/config.py; only the scale/precision knobs are
# set here. Everything is logged to results/ so the run can be audited after
# the pod is torn down.

set -euo pipefail

cd "$(dirname "$0")/.."

NPROC="${NPROC:-1}"
export DEVICE="${DEVICE:-cuda}"
export MAX_SAMPLES="${MAX_SAMPLES:-0}"        # 0 = full corpus
export NUM_EPOCHS="${NUM_EPOCHS:-100}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-20}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export NUM_WORKERS="${NUM_WORKERS:-16}"
export SEED="${SEED:-42}"

mkdir -p results checkpoints

echo "=============================================================="
echo "VIPER GPU RUN"
echo "  date        : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  GPUs        : ${NPROC}"
echo "  epochs      : ${NUM_EPOCHS} (early stop patience ${EARLY_STOP_PATIENCE})"
echo "  max samples : ${MAX_SAMPLES} (0 = all)"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo
echo "--- Data audit -----------------------------------------------"
python scripts/prepare_data.py audit

echo
echo "--- Training -------------------------------------------------"
if [[ "${NPROC}" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="${NPROC}" src/train.py 2>&1 | tee results/train_stdout.log
else
  python src/train.py 2>&1 | tee results/train_stdout.log
fi

echo
echo "--- Evaluation -----------------------------------------------"
python src/evaluate.py 2>&1 | tee results/eval_stdout.log

echo
echo "--- Artifacts ------------------------------------------------"
ls -la results/ checkpoints/
echo "DONE"
