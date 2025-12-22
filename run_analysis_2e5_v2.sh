#!/bin/bash
# =============================================================================
# Meta-RL Behavior-Neural Alignment Analysis Pipeline (2e5 collection) - v2
#
# Difference vs run_analysis_2e5.sh:
#   - Step 2 runs analysis/cca_alignment_v2.py (full PCA, reference-style)
#   - Does NOT pass --pca_dim_x/--pca_dim_y (dims are intentionally not configurable)
#
# Usage:
#   ./run_analysis_2e5_v2.sh              # Run full pipeline
#   ./run_analysis_2e5_v2.sh --skip-pkd   # Skip PKD, use existing cycles
#
# Optional env overrides:
#   CCA_SCRIPT=analysis/cca_alignment_v2.py   (default)
# =============================================================================

set -e  # Exit on error

# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================

SKIP_PKD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-pkd)
            SKIP_PKD=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-pkd    Skip PKD Cycle Sampling, use existing pkd_cycles.npz"
            echo "  -h, --help    Show this help message"
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

# Source routes (from collection output)
SOURCE_ROUTES="/root/backup/kinematics/experiments/run_random_seeds_2e5/data/routes.npz"

# Output experiment name
EXP_NAME="run_random_seeds_2e5_analysis_cca_ac80_gemini_debug"

# Model checkpoint (same as collection)
MODEL_CKPT="/root/logs/ppo/meta-rl-maze-dense-long-n1280meta-40gpu1/model.tar"

# Base output directory
BASE_OUT_DIR="/root/backup/kinematics/experiments"

# Device
DEVICE="cuda:0"

# CCA script (v2)
CCA_SCRIPT="${CCA_SCRIPT:-analysis/cca_alignment_v2.py}"

# =============================================================================
# PKD CYCLE SAMPLER PARAMETERS
# =============================================================================

NUM_H0=20               # Number of random h0 to sample per route
WARMUP_PERIODS=8        # Periods to warmup
SAMPLE_PERIODS=2        # Periods to check convergence
AC_MATCH_THRESH=0.8     # Action consistency threshold (0.8 = 80% match)
SEED=42                 # Random seed

# Length filtering
MIN_LENGTH="5"          # Minimum sequence length
MAX_LENGTH="30"         # Maximum sequence length

# =============================================================================
# CCA PARAMETERS
# =============================================================================

NUM_MODES=10            # Number of CCA modes to visualize
FILTER_OUTLIERS="true"  # Filter outliers in alignment plot

# =============================================================================
# DERIVED PATHS
# =============================================================================

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

# Link source routes
if [ ! -f "${DATA_DIR}/routes.npz" ]; then
    ln -s "${SOURCE_ROUTES}" "${DATA_DIR}/routes.npz" 2>/dev/null || cp "${SOURCE_ROUTES}" "${DATA_DIR}/routes.npz"
fi

# Log to both console and file
exec > >(tee -a "${LOG_FILE}") 2>&1

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         META-RL ALIGNMENT ANALYSIS PIPELINE (v2)             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Experiment:    ${EXP_NAME}"
echo "Source routes: ${SOURCE_ROUTES}"
echo "Model:         ${MODEL_CKPT}"
echo "Device:        ${DEVICE}"
echo "CCA script:    ${CCA_SCRIPT}"
echo "Started:       $(date)"
echo ""
echo "Output: ${EXP_DIR}/"
echo "  ├── data/     - cycles and routes"
echo "  ├── figures/  - CCA plots"
echo "  └── logs/     - pipeline log"
echo ""

# Verify source routes exist
if [ ! -f "${SOURCE_ROUTES}" ]; then
    echo "[ERROR] Source routes not found: ${SOURCE_ROUTES}"
    echo "Please run the collection first."
    exit 1
fi

# Show source routes info
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Source Routes Info:"
python -W ignore -c "
import numpy as np
data = np.load('${SOURCE_ROUTES}', allow_pickle=True)
seeds = data['routes_seed']
ep_lens = data['routes_ep_len']
print(f'  Trajectories: {len(seeds)}')
print(f'  Unique seeds: {len(np.unique(seeds))}')
print(f'  Ep lengths:   min={ep_lens.min()}, max={ep_lens.max()}, mean={ep_lens.mean():.1f}')
"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# =============================================================================
# STEP 1: PKD CYCLE SAMPLING
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
        --routes_npz="${SOURCE_ROUTES}" \
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
# STEP 2: CCA ALIGNMENT ANALYSIS (v2)
# =============================================================================

echo "╭──────────────────────────────────────────────────────────────╮"
echo "│ STEP 2: CCA Alignment Analysis (v2)                          │"
echo "╰──────────────────────────────────────────────────────────────╯"
echo ""
echo "Parameters:"
echo "  ├── num_modes:       ${NUM_MODES}"
echo "  └── filter_outliers: ${FILTER_OUTLIERS}"
echo ""

CCA_START=$(date +%s)

# Build CCA arguments (no PCA dims here; v2 script runs full PCA).
CCA_ARGS="--num_modes=${NUM_MODES}"
[ "${FILTER_OUTLIERS}" = "true" ] && CCA_ARGS="${CCA_ARGS} --filter_outliers"

python -W ignore "${CCA_SCRIPT}" \
    --cycles_npz="${CYCLES_NPZ}" \
    --routes_npz="${SOURCE_ROUTES}" \
    --out_dir="${FIGURES_DIR}" \
    ${CCA_ARGS}

CCA_END=$(date +%s)
CCA_TIME=$((CCA_END - CCA_START))

echo ""
echo "[STEP 2 COMPLETE] CCA analysis: ${CCA_TIME}s"
echo ""

# =============================================================================
# STEP 3: TRAJECTORY STATISTICS
# =============================================================================

echo "╭──────────────────────────────────────────────────────────────╮"
echo "│ STEP 3: Trajectory Statistics                                │"
echo "╰──────────────────────────────────────────────────────────────╯"
echo ""

STATS_START=$(date +%s)

python -W ignore analysis/trajectory_stats.py \
    --routes_npz="${SOURCE_ROUTES}" \
    --out_dir="${FIGURES_DIR}"

STATS_END=$(date +%s)
STATS_TIME=$((STATS_END - STATS_START))

echo ""
echo "[STEP 3 COMPLETE] Statistics: ${STATS_TIME}s"
echo ""

# =============================================================================
# SUMMARY
# =============================================================================

TOTAL_TIME=$((PKD_TIME + CCA_TIME + STATS_TIME))

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    ANALYSIS COMPLETE                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Experiment: ${EXP_NAME}"
echo "Finished:   $(date)"
echo ""
echo "Timing:"
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
echo "    ├── figure5_alignment.png"
echo "    ├── length_distribution.png"
echo "    ├── xy_trajectories.png"
echo "    ├── seed_coverage.png"
echo "    └── action_distribution.png"
echo "  logs/"
echo "    └── analysis.log"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"


