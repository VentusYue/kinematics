#!/bin/bash
# =============================================================================
# Meta-RL Behavior-Neural Alignment Analysis Pipeline (100k Maze) — EXTRA (skip PKD)
#
# IMPORTANT:
# - Defaults to skipping PKD and re-running CCA + stats in-place.
# - SKIP_PKD only works if EXP_NAME matches the existing analysis folder that
#   already contains data/pkd_cycles.npz.
#
# Source of truth for params: experiment log
#   /root/backup/kinematics/experiments/run_optimized_100k_maze_collect_analysis/logs/analysis.log
#
# Usage:
#   ./run_analysis_100k_maze_optimized_extra.sh          # default: skip PKD
#   ./run_analysis_100k_maze_optimized_extra.sh --run-pkd  # recompute PKD then CCA+stats
# =============================================================================

set -e

# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================

SKIP_PKD=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-pkd)
            SKIP_PKD=true
            shift
            ;;
        --run-pkd)
            SKIP_PKD=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-pkd    Skip PKD Cycle Sampling (default)"
            echo "  --run-pkd     Run PKD Cycle Sampling (overrides default)"
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
# CONFIGURATION (from analysis.log)
# =============================================================================

SOURCE_ROUTES="/root/backup/kinematics/experiments/run_optimized_100k_maze_collect/data/routes.npz"
EXP_NAME="run_optimized_100k_maze_collect_analysis"  # must match existing folder for --skip-pkd
MODEL_CKPT="/root/logs/ppo/meta-rl-maze-easy-n10k-trial10-dense-gpu-opt/model_step_170196992.tar"
BASE_OUT_DIR="/root/backup/kinematics/experiments"
DEVICE="cuda:0"

# =============================================================================
# PKD CYCLE SAMPLER PARAMETERS (from analysis.log)
# =============================================================================

NUM_H0=20
WARMUP_PERIODS=8
SAMPLE_PERIODS=2
AC_MATCH_THRESH=0.5
SEED=42
MIN_LENGTH="5"
MAX_LENGTH="256"

# =============================================================================
# CCA PARAMETERS (from analysis.log)
# =============================================================================

NUM_MODES=10
FILTER_OUTLIERS="true"

RIDGE_NORM="global"
GRID_UNIT_ESTIMATOR="axis_mode"
GLOBAL_SCALE_QUANTILE="0.95"
GLOBAL_TARGET_RADIUS="9.0"
RIDGE_RADIUS_SCALE="0.8"
RIDGE_AGGREGATE="max"
RIDGE_NORMALIZE_PATH="false"

# =============================================================================
# DERIVED PATHS
# =============================================================================

EXP_DIR="${BASE_OUT_DIR}/${EXP_NAME}"
DATA_DIR="${EXP_DIR}/data"
FIGURES_DIR="${EXP_DIR}/figures"
LOGS_DIR="${EXP_DIR}/logs"

CYCLES_NPZ="${DATA_DIR}/pkd_cycles.npz"
LOG_FILE="${LOGS_DIR}/analysis_extra.log"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

. /root/miniconda3/etc/profile.d/conda.sh
conda activate ede
cd /root/backup/kinematics

mkdir -p "${DATA_DIR}" "${FIGURES_DIR}" "${LOGS_DIR}"

# Link source routes into analysis dir for consistency
if [ ! -f "${DATA_DIR}/routes.npz" ]; then
    ln -s "${SOURCE_ROUTES}" "${DATA_DIR}/routes.npz" 2>/dev/null || cp "${SOURCE_ROUTES}" "${DATA_DIR}/routes.npz"
fi

# Log to both console and file (separate log to avoid polluting the original analysis.log)
exec > >(tee -a "${LOG_FILE}") 2>&1

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   META-RL ALIGNMENT ANALYSIS PIPELINE (100k Maze) — EXTRA    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Experiment:    ${EXP_NAME}"
echo "Source routes: ${SOURCE_ROUTES}"
echo "Model:         ${MODEL_CKPT}"
echo "Device:        ${DEVICE}"
echo "Started:       $(date)"
echo ""
echo "Output: ${EXP_DIR}/"
echo "  ├── data/"
echo "  ├── figures/"
echo "  └── logs/"
echo ""

if [ ! -f "${SOURCE_ROUTES}" ]; then
    echo "[ERROR] Source routes not found: ${SOURCE_ROUTES}"
    exit 1
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
        echo "This script skips PKD by default, but it can only skip if EXP_NAME matches an existing run."
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
    echo "  ├── min_length:      ${MIN_LENGTH}"
    echo "  └── max_length:      ${MAX_LENGTH}"
    echo ""

    PKD_START=$(date +%s)
    LENGTH_ARGS="--min_length=${MIN_LENGTH} --max_length=${MAX_LENGTH}"

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

CCA_ARGS="--num_modes=${NUM_MODES}"
[ "${FILTER_OUTLIERS}" = "true" ] && CCA_ARGS="${CCA_ARGS} --filter_outliers"
CCA_ARGS="${CCA_ARGS} --ridge_norm=${RIDGE_NORM} --grid_unit_estimator=${GRID_UNIT_ESTIMATOR} --global_scale_quantile=${GLOBAL_SCALE_QUANTILE} --global_target_radius=${GLOBAL_TARGET_RADIUS} --ridge_radius_scale=${RIDGE_RADIUS_SCALE} --ridge_aggregate=${RIDGE_AGGREGATE}"
[ "${RIDGE_NORMALIZE_PATH}" = "true" ] && CCA_ARGS="${CCA_ARGS} --ridge_normalize_path"
CCA_ARGS="${CCA_ARGS} --color_by=all"

python -W ignore analysis/cca_alignment.py \
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
# STEP 3: Trajectory Statistics
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
echo "║                ANALYSIS EXTRA COMPLETE                       ║"
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
echo "Logs: ${LOG_FILE}"
echo ""


