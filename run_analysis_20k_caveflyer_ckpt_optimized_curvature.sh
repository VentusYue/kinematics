#!/bin/bash
# =============================================================================
# Meta-RL Behavior-Neural Alignment Analysis Pipeline (CaveFlyer 20k collection)
#
# Target collection:
#   run_20k_caveflyer_ckpt_optimized (optimized collector + checkpointing)
#
# Runs:
#   Step 0 (optional): Export partial routes from checkpoint
#   Step 1: PKD Cycle Sampling
#   Step 2: CCA Alignment Analysis
#
# Usage:
#   ./run_analysis_20k_caveflyer_ckpt_optimized.sh                    # Full pipeline
#   ./run_analysis_20k_caveflyer_ckpt_optimized.sh --skip-pkd         # Skip PKD, use existing cycles
#   ./run_analysis_20k_caveflyer_ckpt_optimized.sh --use-partial      # Export from checkpoint first
# =============================================================================

set -e

# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================

SKIP_PKD=false
USE_PARTIAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-pkd)
            SKIP_PKD=true
            shift
            ;;
        --use-partial)
            USE_PARTIAL=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-pkd      Skip PKD Cycle Sampling, use existing pkd_cycles.npz"
            echo "  --use-partial   Export partial routes from checkpoint before analysis"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# CONFIGURATION - MODIFY THESE
# =============================================================================

# Source collection experiment (from run_collect_20k_caveflyer_ckpt_optimized.sh)
COLLECT_EXP_NAME="run_20k_caveflyer_ckpt_optimized"

# Output experiment name (analysis outputs go under BASE_OUT_DIR/EXP_NAME/)
EXP_NAME="run_20k_caveflyer_ckpt_optimized_analysis"

# Model checkpoint (same as collection)
MODEL_CKPT="/root/logs/ppo/meta-rl-caveflyer-easy-step1024-n1k-trial10-gpu1=lr2e4/saved/model_step_195952640.tar"

# Base output directory
BASE_OUT_DIR="/root/backup/kinematics/experiments"

# Device
DEVICE="cuda:0"

# =============================================================================
# PKD CYCLE SAMPLER PARAMETERS
# =============================================================================

NUM_H0=50               # Number of random h0 to sample per route
WARMUP_PERIODS=10        # Periods to warmup
SAMPLE_PERIODS=2        # Periods to check convergence
AC_MATCH_THRESH=0.5     # Action consistency threshold
SEED=42                 # Random seed

# Length filtering
MIN_LENGTH="5"          # Minimum sequence length
MAX_LENGTH="512"        # Maximum sequence length

# =============================================================================
# CCA PARAMETERS (Ridge embedding)
# =============================================================================

NUM_MODES=10
FILTER_OUTLIERS="true"

# CaveFlyer movement is not grid/axis-aligned, so median Euclidean step is usually
# more stable than axis-mode.
RIDGE_NORM="global"                 # global | per_episode
GRID_UNIT_ESTIMATOR="median_euclid" # median_euclid | axis_mode
GLOBAL_SCALE_QUANTILE="0.95"
GLOBAL_TARGET_RADIUS="9.0"
RIDGE_RADIUS_SCALE="0.6"
RIDGE_AGGREGATE="max"               # max | sum
RIDGE_NORMALIZE_PATH="false"        # true | false

# =============================================================================
# DERIVED PATHS
# =============================================================================

COLLECT_EXP_DIR="${BASE_OUT_DIR}/${COLLECT_EXP_NAME}"
COLLECT_DATA_DIR="${COLLECT_EXP_DIR}/data"
CKPT_DIR="${COLLECT_DATA_DIR}/ckpt"
SOURCE_ROUTES="${COLLECT_DATA_DIR}/routes.npz"
PARTIAL_ROUTES="${COLLECT_DATA_DIR}/routes_partial.npz"

EXP_DIR="${BASE_OUT_DIR}/${EXP_NAME}"
DATA_DIR="${EXP_DIR}/data"
FIGURES_DIR="${EXP_DIR}/figures"
LOGS_DIR="${EXP_DIR}/logs"

CYCLES_NPZ="${DATA_DIR}/pkd_cycles.npz"
LOG_FILE="${LOGS_DIR}/analysis.log"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

. /root/miniconda3/etc/profile.d/conda.sh
conda activate ede
cd /root/backup/kinematics

mkdir -p "${DATA_DIR}" "${FIGURES_DIR}" "${LOGS_DIR}"

# Log to both console and file
exec > >(tee -a "${LOG_FILE}") 2>&1

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       META-RL ALIGNMENT ANALYSIS PIPELINE (CaveFlyer)        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Collect exp:   ${COLLECT_EXP_NAME}"
echo "Experiment:    ${EXP_NAME}"
echo "Model:         ${MODEL_CKPT}"
echo "Device:        ${DEVICE}"
echo "Started:       $(date)"
echo ""
echo "Output: ${EXP_DIR}/"
echo "  ├── data/     - cycles and routes"
echo "  ├── figures/  - CCA plots"
echo "  └── logs/     - pipeline log"
echo ""

# =============================================================================
# STEP 0 (optional): Export partial routes from checkpoint
# =============================================================================

EXPORT_TIME=0

if [ "${USE_PARTIAL}" = true ]; then
    echo "╭──────────────────────────────────────────────────────────────╮"
    echo "│ STEP 0: Export Partial Routes from Checkpoint                │"
    echo "╰──────────────────────────────────────────────────────────────╯"
    echo ""

    if [ ! -f "${CKPT_DIR}/manifest.json" ]; then
        echo "[ERROR] Checkpoint not found: ${CKPT_DIR}"
        echo "Please run the collection first."
        exit 1
    fi

    EXPORT_START=$(date +%s)

    python -W ignore eval/routes_ckpt_tools.py build \
        --ckpt_dir "${CKPT_DIR}" \
        --out_npz "${PARTIAL_ROUTES}"

    EXPORT_END=$(date +%s)
    EXPORT_TIME=$((EXPORT_END - EXPORT_START))

    echo ""
    echo "[STEP 0 COMPLETE] Export: ${EXPORT_TIME}s"
    echo ""

    ROUTES_NPZ="${PARTIAL_ROUTES}"
else
    ROUTES_NPZ="${SOURCE_ROUTES}"
fi

if [ ! -f "${ROUTES_NPZ}" ]; then
    echo "[ERROR] Routes not found: ${ROUTES_NPZ}"
    if [ "${USE_PARTIAL}" = false ]; then
        echo "Try running with --use-partial to export from checkpoint."
    fi
    exit 1
fi

# Validate routes file is readable by numpy. If corrupted, rebuild a minimal pkd+cca file from ckpt.
python -W ignore - <<PY
import numpy as np
try:
    np.load("${ROUTES_NPZ}", allow_pickle=True).close()
except Exception:
    raise SystemExit(2)
PY
ROUTES_OK=$?
if [ "${ROUTES_OK}" -ne 0 ]; then
    echo "[WARN] routes file appears corrupted/unreadable: ${ROUTES_NPZ}"
    echo "[WARN] Rebuilding a minimal pkd+cca routes file from checkpoint shards..."
    if [ ! -f "${CKPT_DIR}/manifest.json" ]; then
        echo "[ERROR] Cannot rebuild: checkpoint not found at ${CKPT_DIR}"
        exit 1
    fi
    REBUILT_ROUTES="${COLLECT_DATA_DIR}/routes_pkd_cca.npz"
    python -W ignore eval/routes_ckpt_tools.py build \
        --ckpt_dir "${CKPT_DIR}" \
        --out_npz "${REBUILT_ROUTES}" \
        --mode pkd_cca \
        --no_compress
    ROUTES_NPZ="${REBUILT_ROUTES}"
    echo "[INFO] Using rebuilt routes: ${ROUTES_NPZ}"
fi

# Link routes to analysis data dir (symlink if possible, else copy)
if [ ! -f "${DATA_DIR}/routes.npz" ]; then
    ln -s "${ROUTES_NPZ}" "${DATA_DIR}/routes.npz" 2>/dev/null || cp "${ROUTES_NPZ}" "${DATA_DIR}/routes.npz"
fi

# =============================================================================
# STEP 1: PKD Cycle Sampling
# =============================================================================

PKD_TIME=0

if [ "${SKIP_PKD}" = true ]; then
    echo "╭──────────────────────────────────────────────────────────────╮"
    echo "│ STEP 1: PKD Cycle Sampling [SKIPPED]                         │"
    echo "╰──────────────────────────────────────────────────────────────╯"
    echo ""

    if [ ! -f "${CYCLES_NPZ}" ]; then
        echo "[ERROR] Cycles file not found: ${CYCLES_NPZ}"
        echo "Run without --skip-pkd first."
        exit 1
    fi
    echo "[INFO] Using existing: ${CYCLES_NPZ}"
    echo ""
else
    echo "╭──────────────────────────────────────────────────────────────╮"
    echo "│ STEP 1: PKD Cycle Sampling                                   │"
    echo "╰──────────────────────────────────────────────────────────────╯"
    echo ""

    PKD_START=$(date +%s)

    LENGTH_ARGS=""
    [ -n "${MIN_LENGTH}" ] && LENGTH_ARGS="${LENGTH_ARGS} --min_length=${MIN_LENGTH}"
    [ -n "${MAX_LENGTH}" ] && LENGTH_ARGS="${LENGTH_ARGS} --max_length=${MAX_LENGTH}"

    python -W ignore analysis/pkd_cycle_sampler.py \
        --model_ckpt="${MODEL_CKPT}" \
        --routes_npz="${ROUTES_NPZ}" \
        --out_npz="${CYCLES_NPZ}" \
        --device="${DEVICE}" \
        --num_h0=${NUM_H0} \
        --warmup_periods=${WARMUP_PERIODS} \
        --sample_periods=${SAMPLE_PERIODS} \
        --ac_match_thresh=${AC_MATCH_THRESH} \
        --seed=${SEED} \
        ${LENGTH_ARGS}

    PKD_END=$(date +%s)
    PKD_TIME=$((PKD_END - PKD_START))

    echo ""
    echo "[STEP 1 COMPLETE] PKD sampling: ${PKD_TIME}s"
    echo ""
fi

# =============================================================================
# STEP 2: CCA Alignment Analysis
# =============================================================================

echo "╭──────────────────────────────────────────────────────────────╮"
echo "│ STEP 2: CCA Alignment Analysis                               │"
echo "╰──────────────────────────────────────────────────────────────╯"
echo ""

CCA_START=$(date +%s)

CCA_ARGS="--num_modes=${NUM_MODES}"
[ "${FILTER_OUTLIERS}" = "true" ] && CCA_ARGS="${CCA_ARGS} --filter_outliers"
CCA_ARGS="${CCA_ARGS} --ridge_norm=${RIDGE_NORM} --grid_unit_estimator=${GRID_UNIT_ESTIMATOR} --global_scale_quantile=${GLOBAL_SCALE_QUANTILE} --global_target_radius=${GLOBAL_TARGET_RADIUS} --ridge_radius_scale=${RIDGE_RADIUS_SCALE} --ridge_aggregate=${RIDGE_AGGREGATE}"
[ "${RIDGE_NORMALIZE_PATH}" = "true" ] && CCA_ARGS="${CCA_ARGS} --ridge_normalize_path"

python -W ignore analysis/cca_alignment.py \
    --cycles_npz="${CYCLES_NPZ}" \
    --routes_npz="${ROUTES_NPZ}" \
    --out_dir="${FIGURES_DIR}" \
    ${CCA_ARGS}

CCA_END=$(date +%s)
CCA_TIME=$((CCA_END - CCA_START))

echo ""
echo "[STEP 2 COMPLETE] CCA analysis: ${CCA_TIME}s"
echo ""

# =============================================================================
# SUMMARY
# =============================================================================

TOTAL_TIME=$((EXPORT_TIME + PKD_TIME + CCA_TIME))

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    ANALYSIS COMPLETE                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Experiment: ${EXP_NAME}"
echo "Finished:   $(date)"
echo ""
echo "Timing:"
if [ "${USE_PARTIAL}" = true ]; then
    echo "  Export:        ${EXPORT_TIME}s"
fi
if [ "${SKIP_PKD}" = true ]; then
    echo "  PKD Sampling:  skipped"
else
    echo "  PKD Sampling:  ${PKD_TIME}s"
fi
echo "  CCA Analysis:  ${CCA_TIME}s"
echo "  Total:         ${TOTAL_TIME}s"
echo ""
echo "Output: ${EXP_DIR}/"
echo "  data/"
echo "    ├── routes.npz (symlink/copy)"
echo "    └── pkd_cycles.npz"
echo "  figures/"
echo "    ├── cca_lollipop.png"
echo "    ├── fig5_by_length.png"
echo "    ├── fig5_by_displacement.png"
echo "    ├── fig5_by_angle.png"
echo "    ├── alignment_3d.html"
echo "    ├── all_paths_overlay.png"
echo "    └── cca_results.npz"
echo "  logs/"
echo "    └── analysis.log"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"


