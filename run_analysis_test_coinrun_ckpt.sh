#!/bin/bash
# =============================================================================
# Test: Analysis Pipeline with Partial Routes from Checkpoint
#
# This test validates that we can:
# 1. Export partial routes from checkpoint
# 2. Run PKD cycle sampler on the exported routes
# 3. (Optional) Run CCA alignment
#
# Usage:
#   ./run_analysis_test_coinrun_ckpt.sh              # Full pipeline (export + PKD)
#   ./run_analysis_test_coinrun_ckpt.sh --skip-export # Skip export, use existing routes
# =============================================================================

set -e

# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================

SKIP_EXPORT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-export)
            SKIP_EXPORT=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-export   Skip checkpoint export, use existing routes_partial.npz"
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
# CONFIGURATION
# =============================================================================

EXP_NAME="test_coinrun_ckpt"

MODEL_CKPT="/root/logs/ppo/meta-rl-coinrun-easy-step1024-n1k-trial10-gpu0=lr2e4/saved/model_step_100007936.tar"

BASE_OUT_DIR="/root/backup/kinematics/experiments"

# Analysis experiment name
ANALYSIS_EXP_NAME="${EXP_NAME}_analysis"

# Device
DEVICE="cuda:0"

# =============================================================================
# DERIVED PATHS
# =============================================================================

EXP_DIR="${BASE_OUT_DIR}/${EXP_NAME}"
DATA_DIR="${EXP_DIR}/data"
CKPT_DIR="${DATA_DIR}/ckpt"
ROUTES_PARTIAL_NPZ="${DATA_DIR}/routes_partial.npz"

ANALYSIS_EXP_DIR="${BASE_OUT_DIR}/${ANALYSIS_EXP_NAME}"
ANALYSIS_DATA_DIR="${ANALYSIS_EXP_DIR}/data"
ANALYSIS_FIGURES_DIR="${ANALYSIS_EXP_DIR}/figures"
ANALYSIS_LOGS_DIR="${ANALYSIS_EXP_DIR}/logs"

CYCLES_NPZ="${ANALYSIS_DATA_DIR}/pkd_cycles.npz"
LOG_FILE="${ANALYSIS_LOGS_DIR}/analysis.log"

# =============================================================================
# PKD CYCLE SAMPLER PARAMETERS (small for testing)
# =============================================================================

NUM_H0=5                # Small number for quick test
WARMUP_PERIODS=2        # Small warmup
SAMPLE_PERIODS=1        # Small sample
AC_MATCH_THRESH=0.5     # Action consistency threshold
SEED=42                 # Random seed

# Length filtering
MIN_LENGTH="5"          # Minimum sequence length
MAX_LENGTH="256"        # Maximum sequence length

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

. /root/miniconda3/etc/profile.d/conda.sh
conda activate ede
cd /root/backup/kinematics

mkdir -p "${ANALYSIS_DATA_DIR}" "${ANALYSIS_FIGURES_DIR}" "${ANALYSIS_LOGS_DIR}"

# Log to both console and file
exec > >(tee -a "${LOG_FILE}") 2>&1

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║    TEST: ANALYSIS PIPELINE WITH PARTIAL ROUTES               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Source checkpoint: ${CKPT_DIR}"
echo "Analysis experiment: ${ANALYSIS_EXP_NAME}"
echo "Model: ${MODEL_CKPT}"
echo "Device: ${DEVICE}"
echo "Started: $(date)"
echo ""

# Verify checkpoint exists
if [ ! -f "${CKPT_DIR}/manifest.json" ]; then
    echo "[ERROR] Checkpoint not found: ${CKPT_DIR}"
    echo "Please run run_collect_test_coinrun_ckpt.sh first."
    exit 1
fi

# =============================================================================
# STEP 1: Export routes from checkpoint
# =============================================================================

EXPORT_TIME=0

if [ "${SKIP_EXPORT}" = true ]; then
    echo "╭──────────────────────────────────────────────────────────────╮"
    echo "│ STEP 1: Export Routes from Checkpoint [SKIPPED]              │"
    echo "╰──────────────────────────────────────────────────────────────╯"
    echo ""

    if [ ! -f "${ROUTES_PARTIAL_NPZ}" ]; then
        echo "[ERROR] Routes file not found: ${ROUTES_PARTIAL_NPZ}"
        echo "Run without --skip-export first."
        exit 1
    fi
    echo "[INFO] Using existing: ${ROUTES_PARTIAL_NPZ}"
    echo ""
else
    echo "╭──────────────────────────────────────────────────────────────╮"
    echo "│ STEP 1: Export Routes from Checkpoint                        │"
    echo "╰──────────────────────────────────────────────────────────────╯"
    echo ""

    EXPORT_START=$(date +%s)

    python -W ignore eval/routes_ckpt_tools.py build \
        --ckpt_dir "${CKPT_DIR}" \
        --out_npz "${ROUTES_PARTIAL_NPZ}"

    EXPORT_END=$(date +%s)
    EXPORT_TIME=$((EXPORT_END - EXPORT_START))

    echo ""
    echo "[STEP 1 COMPLETE] Export: ${EXPORT_TIME}s"
    echo ""
fi

# Verify exported file
if [ ! -f "${ROUTES_PARTIAL_NPZ}" ]; then
    echo "[ERROR] Routes file not found: ${ROUTES_PARTIAL_NPZ}"
    exit 1
fi

# Show routes info
echo "Routes info:"
python -W ignore -c "
import numpy as np
data = np.load('${ROUTES_PARTIAL_NPZ}', allow_pickle=True)
seeds = data['routes_seed']
ep_lens = data['routes_ep_len']
success = data['routes_success'] if 'routes_success' in data.files else None
print(f'  Trajectories: {len(seeds)}')
print(f'  Unique seeds: {len(np.unique(seeds))}')
print(f'  Ep lengths:   min={ep_lens.min()}, max={ep_lens.max()}, mean={ep_lens.mean():.1f}')
if success is not None:
    print(f'  Success rate: {100.0*success.mean():.1f}%')
"
echo ""

# Link exported routes to analysis data dir
if [ ! -f "${ANALYSIS_DATA_DIR}/routes.npz" ]; then
    ln -s "${ROUTES_PARTIAL_NPZ}" "${ANALYSIS_DATA_DIR}/routes.npz" 2>/dev/null || \
        cp "${ROUTES_PARTIAL_NPZ}" "${ANALYSIS_DATA_DIR}/routes.npz"
fi

# =============================================================================
# STEP 2: PKD Cycle Sampling
# =============================================================================

echo "╭──────────────────────────────────────────────────────────────╮"
echo "│ STEP 2: PKD Cycle Sampling                                   │"
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

python -W ignore analysis/pkd_cycle_sampler.py \
    --model_ckpt="${MODEL_CKPT}" \
    --routes_npz="${ROUTES_PARTIAL_NPZ}" \
    --out_npz="${CYCLES_NPZ}" \
    --device="${DEVICE}" \
    --num_h0=${NUM_H0} \
    --warmup_periods=${WARMUP_PERIODS} \
    --sample_periods=${SAMPLE_PERIODS} \
    --ac_match_thresh=${AC_MATCH_THRESH} \
    --seed=${SEED} \
    --min_length=${MIN_LENGTH} \
    --max_length=${MAX_LENGTH} \
    --max_routes=20 \
    --batch_size=4

PKD_END=$(date +%s)
PKD_TIME=$((PKD_END - PKD_START))

echo ""
echo "[STEP 2 COMPLETE] PKD sampling: ${PKD_TIME}s"
echo ""

# =============================================================================
# SUMMARY
# =============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    TEST COMPLETE                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Export time:    ${EXPORT_TIME}s"
echo "PKD time:       ${PKD_TIME}s"
echo "Total time:     $((EXPORT_TIME + PKD_TIME))s"
echo ""
echo "Output files:"
echo "  ├── Routes:   ${ROUTES_PARTIAL_NPZ}"
echo "  └── Cycles:   ${CYCLES_NPZ}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✓ Test passed! The checkpoint/export/PKD pipeline works correctly."
echo ""

