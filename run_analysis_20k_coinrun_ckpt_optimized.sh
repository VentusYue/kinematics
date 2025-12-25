#!/bin/bash
# =============================================================================
# Meta-RL Behavior-Neural Alignment Analysis Pipeline (CoinRun 20k collection)
#
# Target collection:
#   run_20k_coinrun_ckpt_optimized (optimized collector + checkpointing)
#
# Runs:
#   Step 0 (optional): Export partial routes from checkpoint
#   Step 1: PKD Cycle Sampling
#   Step 2: CCA Alignment Analysis
#   Step 3: Trajectory Statistics
#
# Usage:
#   ./run_analysis_20k_coinrun_ckpt_optimized.sh                    # Full pipeline
#   ./run_analysis_20k_coinrun_ckpt_optimized.sh --skip-pkd         # Skip PKD, use existing cycles
#   ./run_analysis_20k_coinrun_ckpt_optimized.sh --use-partial      # Export from checkpoint first
#   ./run_analysis_20k_coinrun_ckpt_optimized.sh --use-partial --skip-pkd  # Use partial + skip PKD
# =============================================================================

set -e  # Exit on error

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

# Source collection experiment (from run_collect_20k_coinrun_ckpt_optimized.sh)
COLLECT_EXP_NAME="run_20k_coinrun_ckpt_optimized"

# Output experiment name (analysis outputs go under BASE_OUT_DIR/EXP_NAME/)
EXP_NAME="run_20k_coinrun_ckpt_optimized_analysis_h50"

# Model checkpoint (same as collection)
MODEL_CKPT="/root/logs/ppo/meta-rl-coinrun-easy-step1024-n1k-trial10-gpu0=lr2e4/saved/model_step_132775936.tar"

# Base output directory
BASE_OUT_DIR="/root/backup/kinematics/experiments"

# Device
DEVICE="cuda:0"

# Target collection tag (for printouts only)
COLLECTION_TAG="CoinRun 20k with checkpointing (optimized collector)"

# =============================================================================
# PKD CYCLE SAMPLER PARAMETERS
# =============================================================================

NUM_H0=50               # Number of random h0 to sample per route
WARMUP_PERIODS=8        # Periods to warmup
SAMPLE_PERIODS=2        # Periods to check convergence
AC_MATCH_THRESH=0.5     # Action consistency threshold (0.5 = 50% match)
SEED=42                 # Random seed

# Length filtering
MIN_LENGTH="5"          # Minimum sequence length
MAX_LENGTH="256"        # Maximum sequence length

# PCA dims for CCA inputs
PCA_DIM_X=50            # PCA dims for Neural state (X)
PCA_DIM_Y=50            # PCA dims for Behavior Ridge (Y)

# =============================================================================
# CCA PARAMETERS
# =============================================================================

NUM_MODES=10            # Number of CCA modes to visualize
FILTER_OUTLIERS="true"  # Filter outliers in alignment plot
TEST_FRAC="0.2"         # Held-out fraction for reporting test correlations

# Ridge defaults
RIDGE_NORM="global"               # global | per_episode
GRID_UNIT_ESTIMATOR="axis_mode"   # axis_mode | median_euclid
GLOBAL_SCALE_QUANTILE="0.95"
GLOBAL_TARGET_RADIUS="9.0"
RIDGE_RADIUS_SCALE="0.8"
RIDGE_AGGREGATE="max"             # max | sum
RIDGE_NORMALIZE_PATH="false"      # true | false

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
echo "║       META-RL ALIGNMENT ANALYSIS PIPELINE (CoinRun)          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Collection:    ${COLLECTION_TAG}"
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

    # Use partial routes for analysis
    ROUTES_NPZ="${PARTIAL_ROUTES}"
else
    # Use full routes
    ROUTES_NPZ="${SOURCE_ROUTES}"
fi

# Verify routes exist
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
except Exception as e:
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

# Link routes to analysis data dir
if [ ! -f "${DATA_DIR}/routes.npz" ]; then
    ln -s "${ROUTES_NPZ}" "${DATA_DIR}/routes.npz" 2>/dev/null || cp "${ROUTES_NPZ}" "${DATA_DIR}/routes.npz"
fi

# Show routes info
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Routes Info:"
python -W ignore -c "
import numpy as np
data = np.load('${ROUTES_NPZ}', allow_pickle=True)
seeds = data['routes_seed']
ep_lens = data['routes_ep_len']
success = data['routes_success'] if 'routes_success' in data.files else None
print(f'  Trajectories: {len(seeds)}')
print(f'  Unique seeds: {len(np.unique(seeds))}')
print(f'  Ep lengths:   min={ep_lens.min()}, max={ep_lens.max()}, mean={ep_lens.mean():.1f}')
if success is not None:
    print(f'  Success rate: {100.0*success.mean():.1f}%')
"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

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
    echo "Parameters:"
    echo "  ├── num_h0:          ${NUM_H0}"
    echo "  ├── warmup_periods:  ${WARMUP_PERIODS}"
    echo "  ├── sample_periods:  ${SAMPLE_PERIODS}"
    echo "  ├── ac_match_thresh: ${AC_MATCH_THRESH}"
    echo "  ├── min_length:      ${MIN_LENGTH:-none}"
    echo "  └── max_length:      ${MAX_LENGTH:-none}"
    echo ""

    PKD_START=$(date +%s)

    # Build length filter arguments
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
echo "Parameters:"
echo "  ├── num_modes:       ${NUM_MODES}"
echo "  ├── filter_outliers: ${FILTER_OUTLIERS}"
echo "  ├── ridge_norm:      ${RIDGE_NORM}"
echo "  ├── grid_unit:       ${GRID_UNIT_ESTIMATOR}"
echo "  ├── global_scale_q:  ${GLOBAL_SCALE_QUANTILE}"
echo "  ├── target_radius:   ${GLOBAL_TARGET_RADIUS}"
echo "  └── ridge_radius:    ${RIDGE_RADIUS_SCALE}"
echo ""

CCA_START=$(date +%s)

# Build CCA arguments
CCA_ARGS="--num_modes=${NUM_MODES} --pca_dim_x=${PCA_DIM_X} --pca_dim_y=${PCA_DIM_Y}"
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
# STEP 3: Trajectory Statistics
# =============================================================================

echo "╭──────────────────────────────────────────────────────────────╮"
echo "│ STEP 3: Trajectory Statistics                                │"
echo "╰──────────────────────────────────────────────────────────────╯"
echo ""

STATS_START=$(date +%s)

python -W ignore analysis/trajectory_stats.py \
    --routes_npz="${ROUTES_NPZ}" \
    --out_dir="${FIGURES_DIR}"

STATS_END=$(date +%s)
STATS_TIME=$((STATS_END - STATS_START))

echo ""
echo "[STEP 3 COMPLETE] Statistics: ${STATS_TIME}s"
echo ""

# =============================================================================
# SUMMARY
# =============================================================================

TOTAL_TIME=$((EXPORT_TIME + PKD_TIME + CCA_TIME + STATS_TIME))

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    ANALYSIS COMPLETE                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Collection: ${COLLECTION_TAG}"
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
echo "  Statistics:    ${STATS_TIME}s"
echo "  Total:         ${TOTAL_TIME}s"
echo ""
echo "Output: ${EXP_DIR}/"
echo "  data/"
echo "    ├── routes.npz (symlink)"
echo "    └── pkd_cycles.npz"
echo "  figures/"
echo "    ├── cca_lollipop.png"
echo "    ├── fig5_by_length.png"
echo "    ├── fig5_by_displacement.png"
echo "    ├── fig5_by_angle.png"
echo "    ├── alignment_3d.html"
echo "    ├── all_paths_overlay.png"
echo "    ├── cca_results.npz"
echo "    ├── length_distribution.png"
echo "    ├── xy_trajectories.png"
echo "    ├── seed_coverage.png"
echo "    └── action_distribution.png"
echo "  logs/"
echo "    └── analysis.log"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"


