#!/bin/bash
# =============================================================================
# Meta-RL Behavior-Neural Alignment Pipeline
# Runs: collect_meta_routes -> pkd_cycle_sampler -> cca_alignment
# =============================================================================

set -e  # Exit on error

# =============================================================================
# CONFIGURATION - MODIFY THESE LINES TO CHANGE OUTPUT NAMES
# =============================================================================

# Experiment name (used for subfolder naming)
EXP_NAME="run_easy_2e5"

# Model checkpoint
MODEL_CKPT="/root/logs/ppo/meta-rl-maze-dense-long-n1280meta-40gpu1/model.tar"

# Base output directory (all experiment outputs go under here)
BASE_OUT_DIR="/root/backup/kinematics/experiments"

# =============================================================================
# COLLECTION PARAMETERS
# =============================================================================

NUM_TASKS=200000           # Number of maze seeds to evaluate
NUM_PROCESSES=128        # Parallel environments
ADAPT_EPISODES=5        # Episodes for adaptation (explore)
RECORD_EPISODES=2       # Episodes to record (deterministic)
SEED_OFFSET=0           # Starting seed
MAX_STEPS=1024          # Max steps per episode (safety cap)
MAX_EP_LEN=100           # Max episode length to keep (0=no limit, filters long exploration runs)
DISTRIBUTION_MODE="easy" # Procgen distribution mode: "easy" or "hard" (must match training!)

# =============================================================================
# PKD CYCLE SAMPLER PARAMETERS
# =============================================================================

NUM_H0=20               # Number of random h0 to sample per route
WARMUP_PERIODS=8        # Periods to warmup
SAMPLE_PERIODS=2        # Periods to check convergence
AC_MATCH_THRESH=0.8    # Action consistency threshold

# =============================================================================
# CCA PARAMETERS
# =============================================================================

NUM_MODES=10            # Number of CCA modes to visualize

# =============================================================================
# DERIVED PATHS (auto-generated from EXP_NAME)
# All outputs organized under: experiments/<EXP_NAME>/
#   data/     - NPZ data files (routes, cycles)
#   figures/  - Output images (CCA plots, alignment figures)
#   logs/     - Pipeline logs
# =============================================================================

EXP_DIR="${BASE_OUT_DIR}/${EXP_NAME}"
DATA_DIR="${EXP_DIR}/data"
FIGURES_DIR="${EXP_DIR}/figures"
LOGS_DIR="${EXP_DIR}/logs"

ROUTES_NPZ="${DATA_DIR}/routes.npz"
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

# Start logging (tee to both console and log file)
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================================"
echo "Meta-RL Behavior-Neural Alignment Pipeline"
echo "============================================================"
echo "Experiment: ${EXP_NAME}"
echo "Model: ${MODEL_CKPT}"
echo "Started: $(date)"
echo ""
echo "Output directory: ${EXP_DIR}/"
echo "  ├── data/     - ${DATA_DIR}"
echo "  ├── figures/  - ${FIGURES_DIR}"
echo "  └── logs/     - ${LOGS_DIR}"
echo ""
echo "Output files:"
echo "  Routes:  ${ROUTES_NPZ}"
echo "  Cycles:  ${CYCLES_NPZ}"
echo "  Log:     ${LOG_FILE}"
echo "============================================================"
echo ""

# =============================================================================
# STEP 1: Collect Routes
# =============================================================================

echo ""
echo "============================================================"
echo "STEP 1: Collecting Meta-RL Routes"
echo "============================================================"
echo "Parameters:"
echo "  num_tasks=${NUM_TASKS}"
echo "  num_processes=${NUM_PROCESSES}"
echo "  adapt_episodes=${ADAPT_EPISODES}"
echo "  record_episodes=${RECORD_EPISODES}"
echo "  seed_offset=${SEED_OFFSET}"
echo "  max_steps=${MAX_STEPS}"
echo "  max_ep_len=${MAX_EP_LEN}"
echo "  distribution_mode=${DISTRIBUTION_MODE}"
echo "============================================================"
echo ""

COLLECT_START=$(date +%s)

python eval/collect_meta_routes.py \
    --model_ckpt="${MODEL_CKPT}" \
    --out_npz="${ROUTES_NPZ}" \
    --num_tasks=${NUM_TASKS} \
    --num_processes=${NUM_PROCESSES} \
    --adapt_episodes=${ADAPT_EPISODES} \
    --record_episodes=${RECORD_EPISODES} \
    --seed_offset=${SEED_OFFSET} \
    --max_steps=${MAX_STEPS} \
    --max_ep_len=${MAX_EP_LEN} \
    --distribution_mode=${DISTRIBUTION_MODE} \
    --prefer_success=1 \
    --xy_fail_policy="drop_task" \
    --xy_fail_threshold=0.0

COLLECT_END=$(date +%s)
COLLECT_TIME=$((COLLECT_END - COLLECT_START))

echo ""
echo "[STEP 1 COMPLETE] Collection took ${COLLECT_TIME} seconds"
echo ""

# =============================================================================
# STEP 2: PKD Cycle Sampling
# =============================================================================

echo ""
echo "============================================================"
echo "STEP 2: PKD Cycle Sampling"
echo "============================================================"
echo "Parameters:"
echo "  num_h0=${NUM_H0}"
echo "  warmup_periods=${WARMUP_PERIODS}"
echo "  sample_periods=${SAMPLE_PERIODS}"
echo "  ac_match_thresh=${AC_MATCH_THRESH}"
echo "============================================================"
echo ""

PKD_START=$(date +%s)

python analysis/pkd_cycle_sampler.py \
    --model_ckpt="${MODEL_CKPT}" \
    --routes_npz="${ROUTES_NPZ}" \
    --out_npz="${CYCLES_NPZ}" \
    --num_h0=${NUM_H0} \
    --warmup_periods=${WARMUP_PERIODS} \
    --sample_periods=${SAMPLE_PERIODS} \
    --ac_match_thresh=${AC_MATCH_THRESH}

PKD_END=$(date +%s)
PKD_TIME=$((PKD_END - PKD_START))

echo ""
echo "[STEP 2 COMPLETE] PKD sampling took ${PKD_TIME} seconds"
echo ""

# =============================================================================
# STEP 3: CCA Alignment Analysis
# =============================================================================

echo ""
echo "============================================================"
echo "STEP 3: CCA Alignment Analysis"
echo "============================================================"
echo "Parameters:"
echo "  num_modes=${NUM_MODES}"
echo "Output directory: ${FIGURES_DIR}"
echo "============================================================"
echo ""

CCA_START=$(date +%s)

python analysis/cca_alignment.py \
    --cycles_npz="${CYCLES_NPZ}" \
    --routes_npz="${ROUTES_NPZ}" \
    --out_dir="${FIGURES_DIR}" \
    --num_modes=${NUM_MODES}

CCA_END=$(date +%s)
CCA_TIME=$((CCA_END - CCA_START))

echo ""
echo "[STEP 3 COMPLETE] CCA analysis took ${CCA_TIME} seconds"
echo ""

# =============================================================================
# SUMMARY
# =============================================================================

TOTAL_TIME=$((COLLECT_TIME + PKD_TIME + CCA_TIME))

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo "Experiment: ${EXP_NAME}"
echo "Finished: $(date)"
echo ""
echo "Timing:"
echo "  Collection:    ${COLLECT_TIME}s"
echo "  PKD Sampling:  ${PKD_TIME}s"
echo "  CCA Analysis:  ${CCA_TIME}s"
echo "  Total:         ${TOTAL_TIME}s"
echo ""
echo "Output directory: ${EXP_DIR}/"
echo "  data/"
echo "    └── routes.npz"
echo "    └── pkd_cycles.npz"
echo "  figures/"
echo "    └── cca_lollipop.png"
echo "    └── figure5_alignment.png"
echo "  logs/"
echo "    └── pipeline.log"
echo "============================================================"

