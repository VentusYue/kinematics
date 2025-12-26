#!/bin/bash
# =============================================================================
# Meta-RL Behavior-Neural Alignment Analysis Pipeline (CaveFlyer 20k) — EXTRA (skip PKD)
#
# IMPORTANT:
# - Defaults to skipping PKD and re-running CCA in-place.
# - SKIP_PKD only works if EXP_NAME matches the existing analysis folder that
#   already contains data/pkd_cycles.npz.
#
# Source of truth for params: experiment log
#   /root/backup/kinematics/experiments/run_20k_caveflyer_ckpt_optimized_analysis/logs/analysis.log
#
# Usage:
#   ./run_analysis_20k_caveflyer_ckpt_optimized_extra.sh          # default: skip PKD
#   ./run_analysis_20k_caveflyer_ckpt_optimized_extra.sh --run-pkd  # recompute PKD then CCA
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

COLLECT_EXP_NAME="run_20k_caveflyer_ckpt_optimized"
EXP_NAME="run_20k_caveflyer_ckpt_optimized_analysis"  # must match existing folder for --skip-pkd

MODEL_CKPT="/root/logs/ppo/meta-rl-caveflyer-easy-step1024-n1k-trial10-gpu1=lr2e4/saved/model_step_195952640.tar"
BASE_OUT_DIR="/root/backup/kinematics/experiments"
DEVICE="cuda:1"

# =============================================================================
# PKD CYCLE SAMPLER PARAMETERS (inferred/consistent with analysis.log)
# - num_h0 is inferable from log: total candidates tested = 400000 = 20000*20
# =============================================================================

NUM_H0=20
WARMUP_PERIODS=8
SAMPLE_PERIODS=2
AC_MATCH_THRESH=0.5
SEED=42
MIN_LENGTH="5"
MAX_LENGTH="256"

# =============================================================================
# CCA PARAMETERS (from analysis.log diagnostics)
# =============================================================================

NUM_MODES=10
FILTER_OUTLIERS="true"             # outlier filtering block appears in log

RIDGE_NORM="global"
GRID_UNIT_ESTIMATOR="median_euclid"  # log: Grid unit (median_euclid)
GLOBAL_SCALE_QUANTILE="0.95"
GLOBAL_TARGET_RADIUS="9.0"

# NOTE: CaveFlyer log does not print ridge_radius_scale; we intentionally do NOT
# pass --ridge_radius_scale here so cca_alignment.py uses its default.
RIDGE_RADIUS_SCALE=""
RIDGE_AGGREGATE="max"
RIDGE_NORMALIZE_PATH="false"

# =============================================================================
# DERIVED PATHS
# =============================================================================

COLLECT_EXP_DIR="${BASE_OUT_DIR}/${COLLECT_EXP_NAME}"
COLLECT_DATA_DIR="${COLLECT_EXP_DIR}/data"
ROUTES_NPZ="${COLLECT_DATA_DIR}/routes.npz"

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

# Link routes into analysis dir for consistency
if [ ! -f "${DATA_DIR}/routes.npz" ]; then
    ln -s "${ROUTES_NPZ}" "${DATA_DIR}/routes.npz" 2>/dev/null || cp "${ROUTES_NPZ}" "${DATA_DIR}/routes.npz"
fi

# Log to both console and file (separate log to avoid polluting the original analysis.log)
exec > >(tee -a "${LOG_FILE}") 2>&1

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  META-RL ALIGNMENT ANALYSIS PIPELINE (CaveFlyer) — EXTRA     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Collect exp:   ${COLLECT_EXP_NAME}"
echo "Experiment:    ${EXP_NAME}"
echo "Model:         ${MODEL_CKPT}"
echo "Device:        ${DEVICE}"
echo "Started:       $(date)"
echo ""
echo "Output: ${EXP_DIR}/"
echo "  ├── data/"
echo "  ├── figures/"
echo "  └── logs/"
echo ""

if [ ! -f "${ROUTES_NPZ}" ]; then
    echo "[ERROR] Routes not found: ${ROUTES_NPZ}"
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
CCA_ARGS="${CCA_ARGS} --ridge_norm=${RIDGE_NORM} --grid_unit_estimator=${GRID_UNIT_ESTIMATOR} --global_scale_quantile=${GLOBAL_SCALE_QUANTILE} --global_target_radius=${GLOBAL_TARGET_RADIUS} --ridge_aggregate=${RIDGE_AGGREGATE}"
[ -n "${RIDGE_RADIUS_SCALE}" ] && CCA_ARGS="${CCA_ARGS} --ridge_radius_scale=${RIDGE_RADIUS_SCALE}"
[ "${RIDGE_NORMALIZE_PATH}" = "true" ] && CCA_ARGS="${CCA_ARGS} --ridge_normalize_path"
CCA_ARGS="${CCA_ARGS} --color_by=all"

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

TOTAL_TIME=$((PKD_TIME + CCA_TIME))

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
echo "  Total:         ${TOTAL_TIME}s"
echo ""
echo "Logs: ${LOG_FILE}"
echo ""


