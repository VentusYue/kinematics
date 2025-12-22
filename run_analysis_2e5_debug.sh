#!/bin/bash
# =============================================================================
# Meta-RL Behavior-Neural Alignment Analysis Pipeline (2e5 collection) - DEBUG
#
# - Step 1: PKD Cycle Sampling (optional)
# - Step 2: CCA Alignment Analysis (defaults to analysis/cca_alignment_debug.py)
# - Step 3: Trajectory Statistics
#
# Usage:
#   ./run_analysis_2e5_debug.sh              # Run full pipeline (PKD + CCA + stats)
#   ./run_analysis_2e5_debug.sh --skip-pkd   # Skip PKD, use existing pkd_cycles.npz
#
# Optional env overrides:
#   SOURCE_ROUTES=/abs/path/to/routes.npz
#   MODEL_CKPT=/abs/path/to/model.tar
#   DEVICE=cuda:0
#   EXP_NAME=run_random_seeds_2e5_analysis_cca_ac80_debug
#   CCA_SCRIPT=analysis/cca_alignment_debug.py
# =============================================================================

set -e

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
            echo ""
            echo "Env overrides:"
            echo "  SOURCE_ROUTES, MODEL_CKPT, DEVICE, EXP_NAME, CCA_SCRIPT"
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# CONFIGURATION (defaults; override via env)
# =============================================================================

SOURCE_ROUTES="${SOURCE_ROUTES:-/root/backup/kinematics/experiments/run_random_seeds_2e5/data/routes.npz}"
EXP_NAME="${EXP_NAME:-run_random_seeds_2e5_analysis_cca_ac80_debug}"
MODEL_CKPT="${MODEL_CKPT:-/root/logs/ppo/meta-rl-maze-dense-long-n1280meta-40gpu1/model.tar}"
BASE_OUT_DIR="${BASE_OUT_DIR:-/root/backup/kinematics/experiments}"
DEVICE="${DEVICE:-cuda:0}"
CCA_SCRIPT="${CCA_SCRIPT:-analysis/cca_alignment_debug.py}"

# =============================================================================
# PKD CYCLE SAMPLER PARAMETERS (same defaults as run_analysis_2e5*.sh)
# =============================================================================

NUM_H0=20
WARMUP_PERIODS=8
SAMPLE_PERIODS=2
AC_MATCH_THRESH=0.8
SEED=42

# Length filtering
MIN_LENGTH="5"
MAX_LENGTH="30"

# =============================================================================
# CCA PARAMETERS
# =============================================================================

NUM_MODES=10
FILTER_OUTLIERS="true"

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

# Link source routes into experiment dir
if [ ! -f "${DATA_DIR}/routes.npz" ]; then
    ln -s "${SOURCE_ROUTES}" "${DATA_DIR}/routes.npz" 2>/dev/null || cp "${SOURCE_ROUTES}" "${DATA_DIR}/routes.npz"
fi

# Log to both console and file
exec > >(tee -a "${LOG_FILE}") 2>&1

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         META-RL ALIGNMENT ANALYSIS PIPELINE (DEBUG)          ║"
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
        echo "Run without --skip-pkd first, or point EXP_NAME to an experiment that already has pkd_cycles.npz."
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
# STEP 2: CCA ALIGNMENT ANALYSIS (DEBUG SCRIPT)
# =============================================================================

echo "╭──────────────────────────────────────────────────────────────╮"
echo "│ STEP 2: CCA Alignment Analysis (DEBUG)                       │"
echo "╰──────────────────────────────────────────────────────────────╯"
echo ""
echo "Parameters:"
echo "  ├── num_modes:       ${NUM_MODES}"
echo "  └── filter_outliers: ${FILTER_OUTLIERS}"
echo ""

CCA_START=$(date +%s)

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
echo "  figures/"
echo "    ├── cca_lollipop.png"
echo "    ├── figure5_alignment.png"
echo "    └── ... (more)"
echo "  logs/"
echo "    └── analysis.log"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"


