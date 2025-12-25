#!/bin/bash
# =============================================================================
# Meta-RL Route Collection (Maze) with Checkpointing - OPTIMIZED collector
#
# Reference:
# - run_collect_10k_gpuopt_step170196992.sh (model + params)
# - eval/collect_meta_routes_optimized.py (persistent env workers + optimized data path)
#
# Features:
# - Crash-safe checkpoint shards
# - Resumable via --resume
# - Optional --fresh to restart from scratch
#
# Usage:
#   ./run_collect_10k_maze_gpuopt_step170196992_ckpt_optimized.sh
#   ./run_collect_10k_maze_gpuopt_step170196992_ckpt_optimized.sh --fresh
# =============================================================================

set -e

# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================

FRESH_START=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --fresh)
            FRESH_START=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --fresh       Delete existing checkpoint and start fresh"
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
# CONFIGURATION (matches run_collect_10k_gpuopt_step170196992.sh)
# =============================================================================

EXP_NAME="run_optimized_100k_maze_collect"
MODEL_CKPT="/root/logs/ppo/meta-rl-maze-easy-n10k-trial10-dense-gpu-opt/model_step_170196992.tar"
BASE_OUT_DIR="/root/backup/kinematics/experiments"

# =============================================================================
# COLLECTION PARAMETERS
# =============================================================================

NUM_TASKS=100000            # Target number of SUCCESSFUL trajectories
NUM_PROCESSES=128          # Parallel environments per batch
DEVICE="cuda:0"            # <-- requested: use gpu:0
ADAPT_EPISODES=6           # Episodes for adaptation (sampling)
RECORD_EPISODES=2          # Episodes after adaptation (typically deterministic)
SEED_OFFSET=0              # Starting seed

MAX_STEPS=512              # Max steps per episode
MAX_EP_LEN=256             # Max episode length to keep (also used by selector)

DISTRIBUTION_MODE="easy"   # "easy" or "hard"
REQUIRE_SUCCESS=1          # 1 = only keep successful trajectories

# SPEED OPTIMIZATION PARAMS
BATCH_COMPLETION=0.9       # Don't wait for slowest 10% of envs
TASK_TIMEOUT=0             # Total steps per task (0 = auto)
EARLY_ABORT=1              # Abort tasks with no success after adapt phase

# PERF FLAGS (supported by collect_meta_routes_optimized.py)
USE_COMPILE=1
USE_AMP=1

# CHECKPOINT PARAMS
CKPT_SHARD_SIZE=200         # Flush every N routes
CKPT_FLUSH_SECS=120        # Flush at least every 2 minutes

# =============================================================================
# DERIVED PATHS
# =============================================================================

EXP_DIR="${BASE_OUT_DIR}/${EXP_NAME}"
DATA_DIR="${EXP_DIR}/data"
LOGS_DIR="${EXP_DIR}/logs"
CKPT_DIR="${DATA_DIR}/ckpt"
ROUTES_NPZ="${DATA_DIR}/routes.npz"
LOG_FILE="${LOGS_DIR}/collection.log"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

. /root/miniconda3/etc/profile.d/conda.sh
conda activate ede
cd /root/backup/kinematics

if [ "${FRESH_START}" = true ] && [ -d "${CKPT_DIR}" ]; then
    echo "[WARNING] Deleting existing checkpoint: ${CKPT_DIR}"
    rm -rf "${CKPT_DIR}"
fi

mkdir -p "${DATA_DIR}" "${LOGS_DIR}"

# Suppress gym/numpy deprecation warnings
export PYTHONWARNINGS="ignore::DeprecationWarning:gym"

# Log to both console and file
exec > >(tee -a "${LOG_FILE}") 2>&1

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   META-RL ROUTE COLLECTION (Maze) WITH CHECKPOINTING (OPT)   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Experiment:    ${EXP_NAME}"
echo "Collector:     eval/collect_meta_routes_optimized.py"
echo "Model:         ${MODEL_CKPT}"
echo "Checkpoint:    ${CKPT_DIR}"
echo "Output:        ${ROUTES_NPZ}"
echo "Started:       $(date)"
echo ""
echo "Parameters:"
echo "  ├── Env:                  maze"
echo "  ├── Device:               ${DEVICE}"
echo "  ├── Target trajectories:  ${NUM_TASKS}"
echo "  ├── Batch size:           ${NUM_PROCESSES}"
echo "  ├── Adapt episodes:       ${ADAPT_EPISODES}"
echo "  ├── Record episodes:      ${RECORD_EPISODES}"
echo "  ├── Max steps/episode:    ${MAX_STEPS}"
echo "  ├── Max episode length:   ${MAX_EP_LEN}"
echo "  ├── Require success:      ${REQUIRE_SUCCESS}"
echo "  └── Distribution mode:    ${DISTRIBUTION_MODE}"
echo ""
echo "Perf:"
echo "  ├── torch.compile:        ${USE_COMPILE}"
echo "  └── AMP:                  ${USE_AMP}"
echo ""
echo "Checkpoint settings:"
echo "  ├── Shard size:           ${CKPT_SHARD_SIZE} routes"
echo "  └── Flush interval:       ${CKPT_FLUSH_SECS} seconds"
echo ""
echo "Speed optimizations:"
echo "  ├── Batch completion:     ${BATCH_COMPLETION} (don't wait for slowest)"
echo "  ├── Task timeout:         ${TASK_TIMEOUT} (0=auto)"
echo "  └── Early abort:          ${EARLY_ABORT} (abort failed tasks)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# =============================================================================
# RUN COLLECTION
# =============================================================================

COLLECT_START=$(date +%s)

python -W ignore eval/collect_meta_routes_optimized.py \
    --env_name maze \
    --model_ckpt="${MODEL_CKPT}" \
    --out_npz="${ROUTES_NPZ}" \
    --device="${DEVICE}" \
    --num_tasks=${NUM_TASKS} \
    --num_processes=${NUM_PROCESSES} \
    --adapt_episodes=${ADAPT_EPISODES} \
    --record_episodes=${RECORD_EPISODES} \
    --seed_offset=${SEED_OFFSET} \
    --max_steps=${MAX_STEPS} \
    --max_ep_len=${MAX_EP_LEN} \
    --distribution_mode=${DISTRIBUTION_MODE} \
    --require_success=${REQUIRE_SUCCESS} \
    --batch_completion_threshold=${BATCH_COMPLETION} \
    --task_timeout=${TASK_TIMEOUT} \
    --early_abort=${EARLY_ABORT} \
    --xy_fail_policy="warn_only" \
    --use_compile=${USE_COMPILE} \
    --use_amp=${USE_AMP} \
    --ckpt_dir="${CKPT_DIR}" \
    --resume \
    --ckpt_shard_size=${CKPT_SHARD_SIZE} \
    --ckpt_flush_secs=${CKPT_FLUSH_SECS}

COLLECT_END=$(date +%s)
COLLECT_TIME=$((COLLECT_END - COLLECT_START))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    COLLECTION COMPLETE                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Time:          ${COLLECT_TIME}s"
echo "Output:        ${ROUTES_NPZ}"
echo "Checkpoint:    ${CKPT_DIR}"
echo "Finished:      $(date)"
echo ""

# =============================================================================
# CHECKPOINT INFO
# =============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Checkpoint Info:"
echo ""

python -W ignore eval/routes_ckpt_tools.py info --ckpt_dir "${CKPT_DIR}"


