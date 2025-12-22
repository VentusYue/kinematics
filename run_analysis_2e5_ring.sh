#!/bin/bash
# =============================================================================
# Meta-RL Behavior-Neural Alignment Analysis Pipeline (Ring Topology)
#
# - Step 1: PKD Cycle Sampling (optional)
# - Step 2: CCA Alignment Analysis (Ring/Angle Focused)
# - Step 3: Trajectory Statistics
#
# Usage:
#   ./run_analysis_2e5_ring.sh              # Run full pipeline
#   ./run_analysis_2e5_ring.sh --skip-pkd   # Skip PKD, use existing pkd_cycles.npz
#
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
            echo "Options:"
            echo "  --skip-pkd    Skip PKD Cycle Sampling"
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

SOURCE_ROUTES="${SOURCE_ROUTES:-/root/backup/kinematics/experiments/run_random_seeds_2e5/data/routes.npz}"
EXP_NAME="${EXP_NAME:-run_random_seeds_2e5_analysis_cca_ac80_debug_ring}"
MODEL_CKPT="${MODEL_CKPT:-/root/logs/ppo/meta-rl-maze-dense-long-n1280meta-40gpu1/model.tar}"
BASE_OUT_DIR="${BASE_OUT_DIR:-/root/backup/kinematics/experiments}"
DEVICE="${DEVICE:-cuda:0}"
CCA_SCRIPT="analysis/cca_alignment_ring.py"

# PKD Parameters
NUM_H0=20
WARMUP_PERIODS=8
SAMPLE_PERIODS=2
AC_MATCH_THRESH=0.8
SEED=42
MIN_LENGTH="5"
MAX_LENGTH="30"

# CCA Ring Parameters
PCA_DIM=50
FILTER_IQR="true"
COLOR_BY="angle"

# =============================================================================
# DERIVED PATHS
# =============================================================================

EXP_DIR="${BASE_OUT_DIR}/${EXP_NAME}"
DATA_DIR="${EXP_DIR}/data"
# Save ring figures in a separate folder to preserve original results
FIGURES_DIR="${EXP_DIR}/figures_ring" 
LOGS_DIR="${EXP_DIR}/logs"

CYCLES_NPZ="${DATA_DIR}/pkd_cycles.npz"
LOG_FILE="${LOGS_DIR}/analysis_ring.log"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

. /root/miniconda3/etc/profile.d/conda.sh
conda activate ede
cd /root/backup/kinematics

mkdir -p "${DATA_DIR}" "${FIGURES_DIR}" "${LOGS_DIR}"

# Link source routes if missing
if [ ! -f "${DATA_DIR}/routes.npz" ]; then
    ln -s "${SOURCE_ROUTES}" "${DATA_DIR}/routes.npz" 2>/dev/null || cp "${SOURCE_ROUTES}" "${DATA_DIR}/routes.npz"
fi

exec > >(tee -a "${LOG_FILE}") 2>&1

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║      META-RL ALIGNMENT ANALYSIS PIPELINE (RING TOPOLOGY)     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Experiment:    ${EXP_NAME}"
echo "Source routes: ${SOURCE_ROUTES}"
echo "Script:        ${CCA_SCRIPT}"
echo "Output:        ${FIGURES_DIR}/"
echo ""

# =============================================================================
# STEP 1: PKD CYCLE SAMPLING
# =============================================================================

PKD_TIME=0

if [ "${SKIP_PKD}" = true ]; then
    echo "[STEP 1] PKD Sampling skipped. Using existing: ${CYCLES_NPZ}"
    if [ ! -f "${CYCLES_NPZ}" ]; then
        echo "[ERROR] Cycles file not found!"
        exit 1
    fi
else
    echo "[STEP 1] Running PKD Cycle Sampling..."
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
    echo "PKD Sampling complete in ${PKD_TIME}s"
fi

# =============================================================================
# STEP 2: CCA ALIGNMENT (RING ANALYSIS)
# =============================================================================

echo ""
echo "[STEP 2] Running CCA Alignment (Ring/Angle Analysis)..."
CCA_START=$(date +%s)

CCA_ARGS="--pca_dim=${PCA_DIM} --color_by=${COLOR_BY}"
[ "${FILTER_IQR}" = "true" ] && CCA_ARGS="${CCA_ARGS} --filter_iqr"

python -W ignore "${CCA_SCRIPT}" \
    --cycles_npz="${CYCLES_NPZ}" \
    --routes_npz="${SOURCE_ROUTES}" \
    --out_dir="${FIGURES_DIR}" \
    ${CCA_ARGS}

CCA_END=$(date +%s)
CCA_TIME=$((CCA_END - CCA_START))
echo "CCA Analysis complete in ${CCA_TIME}s"

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "Analysis Finished!"
echo "Figures saved to: ${FIGURES_DIR}"
echo "  - fig5_by_length.png"
echo "  - fig5_by_angle.png"
echo "  - fig5_by_disp.png"
echo "  - fig5_3d_length.png"
echo "  - cca_spectrum.png"
echo ""

