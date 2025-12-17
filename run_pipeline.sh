#!/bin/bash

# Configuration
SUFFIX="test-2000" # Change this suffix to update all filenames (e.g., "multi", "final", etc.)
ENV_NAME="maze"
MODEL_CKPT="/root/logs/ppo/meta-rl-maze-dense-long-n1280meta-40gpu1/model.tar"
NUM_TASKS=2000
TRIALS_PER_TASK=5
DETERMINISTIC=0
NUM_PROCESSES=128
HIDDEN_SIZE=256
ARCH="large"
DEVICE="cuda:1" # Updated to GPU 1

# Derived filenames
ROUTES_NPZ="analysis_data/routes_${SUFFIX}.npz"
CYCLES_NPZ="analysis_data/pkd_cycles_${SUFFIX}.npz"
OUT_DIR="analysis_out_${SUFFIX}"

echo "=================================================="
echo "Running Pipeline with Suffix: ${SUFFIX}"
echo "Routes File: ${ROUTES_NPZ}"
echo "Cycles File: ${CYCLES_NPZ}"
echo "Output Dir:  ${OUT_DIR}"
echo "Device:      ${DEVICE}"
echo "=================================================="

# 1. Collect Routes
echo "Step 1: Collecting Meta-RL Routes..."
python eval/collect_meta_routes.py \
  --env_name "${ENV_NAME}" \
  --model_ckpt "${MODEL_CKPT}" \
  --num_tasks "${NUM_TASKS}" \
  --trials_per_task "${TRIALS_PER_TASK}" \
  --deterministic "${DETERMINISTIC}" \
  --num_processes "${NUM_PROCESSES}" \
  --hidden_size "${HIDDEN_SIZE}" \
  --arch "${ARCH}" \
  --device "${DEVICE}" \
  --out_npz "${ROUTES_NPZ}"

if [ $? -ne 0 ]; then
    echo "Error: Route collection failed."
    exit 1
fi

# 2. PKD Cycle Sampling
echo "Step 2: Sampling PKD Cycles..."
python analysis/pkd_cycle_sampler.py \
  --model_ckpt "${MODEL_CKPT}" \
  --routes_npz "${ROUTES_NPZ}" \
  --num_h0 20 \
  --warmup_periods 8 \
  --sample_periods 2 \
  --ac_match_thresh 0.95 \
  --device "${DEVICE}" \
  --out_npz "${CYCLES_NPZ}"

if [ $? -ne 0 ]; then
    echo "Error: PKD cycle sampling failed."
    exit 1
fi

# 3. CCA Alignment
echo "Step 3: Running CCA Alignment..."
python analysis/cca_alignment.py \
  --cycles_npz "${CYCLES_NPZ}" \
  --routes_npz "${ROUTES_NPZ}" \
  --out_dir "${OUT_DIR}" \
  --num_modes 10

if [ $? -ne 0 ]; then
    echo "Error: CCA alignment failed."
    exit 1
fi

echo "=================================================="
echo "Pipeline Completed Successfully!"
echo "Outputs in: ${OUT_DIR}"
echo "=================================================="

