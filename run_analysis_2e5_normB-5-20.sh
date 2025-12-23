#!/bin/bash
# =============================================================================
# Meta-RL Behavior-Neural Alignment Analysis Pipeline (2e5 collection) — NormB Ablations
#
# This script mirrors `run_analysis_2e5.sh` but runs multiple CCA configurations
# (baseline vs Update-B global normalization, plus optional ridge-radius sweep).
#
# Runs:
#   Step 1: PKD Cycle Sampling (optional / once)
#   Step 2: CCA Alignment Analysis (multiple variants)
#   Step 3: Trajectory Statistics (optional)
#
# Usage:
#   ./run_analysis_2e5_normB.sh
#   ./run_analysis_2e5_normB.sh --skip-pkd
#   ./run_analysis_2e5_normB.sh --skip-stats
#   ./run_analysis_2e5_normB.sh --no-sweep
#   ./run_analysis_2e5_normB.sh --radius-list "1.414,1.0,0.8,0.6,0.4"
# =============================================================================

set -euo pipefail

# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================

SKIP_PKD=false
SKIP_STATS=false
DO_SWEEP=true
RADIUS_LIST="1.414,1.0,0.8,0.6,0.4"

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-pkd)
            SKIP_PKD=true
            shift
            ;;
        --skip-stats)
            SKIP_STATS=true
            shift
            ;;
        --no-sweep)
            DO_SWEEP=false
            shift
            ;;
        --radius-list)
            RADIUS_LIST="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-pkd            Skip PKD Cycle Sampling, use existing pkd_cycles.npz"
            echo "  --skip-stats          Skip trajectory statistics step"
            echo "  --no-sweep            Disable ridge_radius_scale sweep"
            echo "  --radius-list STR     Comma-separated radius_scale list for sweep (default: ${RADIUS_LIST})"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# CONFIGURATION - MODIFY THESE (same style as run_analysis_2e5.sh)
# =============================================================================

# Source routes (from collection output)
SOURCE_ROUTES="/root/backup/kinematics/experiments/run_random_seeds_2e5/data/routes.npz"

# Output experiment name
EXP_NAME="run_random_seeds_2e5_analysis_cca_ac80_normB—v1-50"

# Model checkpoint (same as collection)
MODEL_CKPT="/root/logs/ppo/meta-rl-maze-dense-long-n1280meta-40gpu1/model.tar"

# Base output directory
BASE_OUT_DIR="/root/backup/kinematics/experiments"

# Device
DEVICE="cuda:0"

# =============================================================================
# PKD CYCLE SAMPLER PARAMETERS
# =============================================================================

NUM_H0=20
WARMUP_PERIODS=8
SAMPLE_PERIODS=2
AC_MATCH_THRESH=0.8
SEED=42

# Length filtering
MIN_LENGTH="3"
MAX_LENGTH="50"

# =============================================================================
# CCA PARAMETERS (shared)
# =============================================================================

NUM_MODES=10
FILTER_OUTLIERS="true"
TEST_FRAC=0.2

# =============================================================================
# Update B (global normalization) parameters
# =============================================================================

GRID_UNIT_ESTIMATOR="axis_mode"      # axis_mode | median_euclid
GLOBAL_SCALE_QUANTILE="0.95"
GLOBAL_TARGET_RADIUS="9.0"
GRID_STEP_TOL="0.1"

# Ridge embedding knobs (important for ridge diversity)
RIDGE_AGGREGATE="max"                # max | sum
RIDGE_NORMALIZE_PATH="false"         # false | true
RIDGE_GRID_SIZE="21"

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
echo "║      META-RL ALIGNMENT ANALYSIS PIPELINE — NormB Ablations   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Experiment:    ${EXP_NAME}"
echo "Source routes: ${SOURCE_ROUTES}"
echo "Model:         ${MODEL_CKPT}"
echo "Device:        ${DEVICE}"
echo "Started:       $(date)"
echo ""
echo "Output: ${EXP_DIR}/"
echo "  ├── data/     - cycles and routes"
echo "  ├── figures/  - CCA plots (multiple variants)"
echo "  └── logs/     - pipeline log"
echo ""

# Verify source routes exist
if [ ! -f "${SOURCE_ROUTES}" ]; then
    echo "[ERROR] Source routes not found: ${SOURCE_ROUTES}"
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
# STEP 2: CCA Alignment Analysis (variants)
# =============================================================================

echo "╭──────────────────────────────────────────────────────────────╮"
echo "│ STEP 2: CCA Alignment Analysis (Variants)                    │"
echo "╰──────────────────────────────────────────────────────────────╯"
echo ""

CCA_START=$(date +%s)

CCA_BASE_ARGS="--num_modes=${NUM_MODES} --test_frac=${TEST_FRAC} --grid_step_tol=${GRID_STEP_TOL} --ridge_grid_size=${RIDGE_GRID_SIZE} --ridge_aggregate=${RIDGE_AGGREGATE}"
[ "${FILTER_OUTLIERS}" = "true" ] && CCA_BASE_ARGS="${CCA_BASE_ARGS} --filter_outliers"
[ "${RIDGE_NORMALIZE_PATH}" = "true" ] && CCA_BASE_ARGS="${CCA_BASE_ARGS} --ridge_normalize_path"

run_cca_variant () {
    local label="$1"
    shift
    local out_dir="${FIGURES_DIR}/${label}"
    mkdir -p "${out_dir}"
    echo ""
    echo "----------------------------------------"
    echo "CCA VARIANT: ${label}"
    echo "Output: ${out_dir}"
    echo "Args: ${CCA_BASE_ARGS} $*"
    echo "----------------------------------------"
    python -W ignore analysis/cca_alignment.py \
        --cycles_npz="${CYCLES_NPZ}" \
        --routes_npz="${SOURCE_ROUTES}" \
        --out_dir="${out_dir}" \
        ${CCA_BASE_ARGS} \
        "$@"
}

# (A) Baseline (original behavior normalization)
run_cca_variant "A_baseline_per_episode" --ridge_norm=per_episode

# (B) Update-B: global normalization + scaling diagnostics
run_cca_variant "B_global_${GRID_UNIT_ESTIMATOR}_q${GLOBAL_SCALE_QUANTILE}_R${GLOBAL_TARGET_RADIUS}" \
    --ridge_norm=global \
    --grid_unit_estimator="${GRID_UNIT_ESTIMATOR}" \
    --global_scale_quantile="${GLOBAL_SCALE_QUANTILE}" \
    --global_target_radius="${GLOBAL_TARGET_RADIUS}" \
    --save_scale_diagnostics

# (C) Sweep ridge_radius_scale under global normalization (key for ridge diversity)
if [ "${DO_SWEEP}" = true ]; then
    IFS=',' read -r -a RADII <<< "${RADIUS_LIST}"
    for r in "${RADII[@]}"; do
        r_tag="$(echo "${r}" | sed 's/\./p/g')"
        run_cca_variant "C_global_${GRID_UNIT_ESTIMATOR}_r${r_tag}" \
            --ridge_norm=global \
            --grid_unit_estimator="${GRID_UNIT_ESTIMATOR}" \
            --global_scale_quantile="${GLOBAL_SCALE_QUANTILE}" \
            --global_target_radius="${GLOBAL_TARGET_RADIUS}" \
            --ridge_radius_scale="${r}"
    done
fi

CCA_END=$(date +%s)
CCA_TIME=$((CCA_END - CCA_START))

echo ""
echo "[STEP 2 COMPLETE] CCA variants: ${CCA_TIME}s"
echo ""

# =============================================================================
# STEP 3: Trajectory Statistics
# =============================================================================

STATS_TIME=0

if [ "${SKIP_STATS}" = true ]; then
    echo "╭──────────────────────────────────────────────────────────────╮"
    echo "│ STEP 3: Trajectory Statistics [SKIPPED]                      │"
    echo "╰──────────────────────────────────────────────────────────────╯"
    echo ""
else
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
fi

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
echo "  CCA Variants:  ${CCA_TIME}s"
if [ "${SKIP_STATS}" = true ]; then
    echo "  Statistics:    skipped"
else
    echo "  Statistics:    ${STATS_TIME}s"
fi
echo "  Total:         ${TOTAL_TIME}s"
echo ""
echo "Output: ${EXP_DIR}/"
echo "  data/"
echo "    ├── routes.npz (symlink)"
echo "    └── pkd_cycles.npz"
echo "  figures/"
echo "    ├── A_baseline_per_episode/"
echo "    ├── B_global_${GRID_UNIT_ESTIMATOR}_q${GLOBAL_SCALE_QUANTILE}_R${GLOBAL_TARGET_RADIUS}/"
if [ "${DO_SWEEP}" = true ]; then
    echo "    └── C_global_${GRID_UNIT_ESTIMATOR}_r*/"
fi
echo "  logs/"
echo "    └── analysis.log"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"


