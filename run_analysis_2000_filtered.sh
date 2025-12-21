#!/bin/bash
# =============================================================================
# Meta-RL Behavior-Neural Alignment Pipeline (Analysis Only)
# Analyzes the run_easy_2000 dataset with updated filtering
# =============================================================================

set -e  # Exit on error

# =============================================================================
# CONFIGURATION
# =============================================================================

# Experiment name for this analysis run
EXP_NAME="run_easy_2000_analysis_filtered_ac95"

# Source routes data (from the original 2000 collection)
SOURCE_ROUTES="/root/backup/kinematics/experiments/run_easy_2000_original/data/routes.npz"

# Model checkpoint
MODEL_CKPT="/root/logs/ppo/meta-rl-maze-dense-long-n1280meta-40gpu1/model.tar"

# Base output directory
BASE_OUT_DIR="/root/backup/kinematics/experiments"

# =============================================================================
# PKD CYCLE SAMPLER PARAMETERS
# =============================================================================

NUM_H0=20               # Number of random h0 to sample per route
WARMUP_PERIODS=8        # Periods to warmup
SAMPLE_PERIODS=2        # Periods to check convergence
AC_MATCH_THRESH=0.95     # Action consistency threshold
SEED=42                 # Random seed for reproducibility

# Length filtering
MIN_LENGTH="3"           # Minimum sequence length
MAX_LENGTH="15"          # Maximum sequence length

# =============================================================================
# CCA PARAMETERS
# =============================================================================

NUM_MODES=10            # Number of CCA modes to visualize
FILTER_OUTLIERS="true"  # Enable outlier filtering for the alignment plot

# =============================================================================
# DERIVED PATHS
# =============================================================================

EXP_DIR="${BASE_OUT_DIR}/${EXP_NAME}"
DATA_DIR="${EXP_DIR}/data"
FIGURES_DIR="${EXP_DIR}/figures"
LOGS_DIR="${EXP_DIR}/logs"

CYCLES_NPZ="${DATA_DIR}/pkd_cycles.npz"
LOG_FILE="${LOGS_DIR}/pipeline.log"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Activate conda environment
. /root/miniconda3/etc/profile.d/conda.sh
conda activate ede

cd /root/backup/kinematics

# Create output directories
mkdir -p "${DATA_DIR}" "${FIGURES_DIR}" "${LOGS_DIR}"

# Copy/link source routes to this experiment's data folder for reference
if [ ! -f "${DATA_DIR}/routes.npz" ]; then
    ln -s "${SOURCE_ROUTES}" "${DATA_DIR}/routes.npz" 2>/dev/null || cp "${SOURCE_ROUTES}" "${DATA_DIR}/routes.npz"
fi

# Start logging (tee to both console and log file)
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================================"
echo "Meta-RL Alignment Pipeline (Analysis Only)"
echo "============================================================"
echo "Experiment: ${EXP_NAME}"
echo "Model: ${MODEL_CKPT}"
echo "Source routes: ${SOURCE_ROUTES}"
echo "Started: $(date)"
echo ""
echo "Output directory: ${EXP_DIR}/"
echo "  ├── data/     - ${DATA_DIR}"
echo "  ├── figures/  - ${FIGURES_DIR}"
echo "  └── logs/     - ${LOGS_DIR}"
echo "============================================================"
echo ""

# Verify source routes exist
if [ ! -f "${SOURCE_ROUTES}" ]; then
    echo "[ERROR] Source routes not found: ${SOURCE_ROUTES}"
    exit 1
fi

echo "[INFO] Using existing routes: ${SOURCE_ROUTES}"
echo ""

# =============================================================================
# STEP 1: PKD Cycle Sampling
# =============================================================================

echo ""
echo "============================================================"
echo "STEP 1: PKD Cycle Sampling"
echo "============================================================"
echo "Parameters:"
echo "  num_h0=${NUM_H0}"
echo "  warmup_periods=${WARMUP_PERIODS}"
echo "  sample_periods=${SAMPLE_PERIODS}"
echo "  ac_match_thresh=${AC_MATCH_THRESH}"
echo "  min_length=${MIN_LENGTH:-none}"
echo "  max_length=${MAX_LENGTH:-none}"
echo "============================================================"
echo ""

PKD_START=$(date +%s)

# Build optional length filter arguments
LENGTH_ARGS=""
if [ -n "${MIN_LENGTH}" ]; then
    LENGTH_ARGS="${LENGTH_ARGS} --min_length=${MIN_LENGTH}"
fi
if [ -n "${MAX_LENGTH}" ]; then
    LENGTH_ARGS="${LENGTH_ARGS} --max_length=${MAX_LENGTH}"
fi

python analysis/pkd_cycle_sampler.py \
    --model_ckpt="${MODEL_CKPT}" \
    --routes_npz="${SOURCE_ROUTES}" \
    --out_npz="${CYCLES_NPZ}" \
    --num_h0=${NUM_H0} \
    --warmup_periods=${WARMUP_PERIODS} \
    --sample_periods=${SAMPLE_PERIODS} \
    --ac_match_thresh=${AC_MATCH_THRESH} \
    --seed=${SEED} \
    ${LENGTH_ARGS}

PKD_END=$(date +%s)
PKD_TIME=$((PKD_END - PKD_START))

echo ""
echo "[STEP 1 COMPLETE] PKD sampling took ${PKD_TIME} seconds"
echo ""

# =============================================================================
# STEP 2: CCA Alignment Analysis
# =============================================================================

echo ""
echo "============================================================"
echo "STEP 2: CCA Alignment Analysis"
echo "============================================================"
echo "Parameters:"
echo "  num_modes=${NUM_MODES}"
echo "  filter_outliers=${FILTER_OUTLIERS}"
echo "Output directory: ${FIGURES_DIR}"
echo "============================================================"
echo ""

CCA_START=$(date +%s)

# Build CCA arguments
CCA_ARGS="--num_modes=${NUM_MODES}"
if [ "${FILTER_OUTLIERS}" = "true" ]; then
    CCA_ARGS="${CCA_ARGS} --filter_outliers"
fi

python analysis/cca_alignment.py \
    --cycles_npz="${CYCLES_NPZ}" \
    --routes_npz="${SOURCE_ROUTES}" \
    --out_dir="${FIGURES_DIR}" \
    ${CCA_ARGS}

CCA_END=$(date +%s)
CCA_TIME=$((CCA_END - CCA_START))

echo ""
echo "[STEP 2 COMPLETE] CCA analysis took ${CCA_TIME} seconds"
echo ""

# =============================================================================
# SUMMARY
# =============================================================================

TOTAL_TIME=$((PKD_TIME + CCA_TIME))

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo "Experiment: ${EXP_NAME}"
echo "Finished: $(date)"
echo ""
echo "Timing:"
echo "  PKD Sampling:  ${PKD_TIME}s"
echo "  CCA Analysis:  ${CCA_TIME}s"
echo "  Total:         ${TOTAL_TIME}s"
echo ""
echo "Output directory: ${EXP_DIR}/"
echo "============================================================"

