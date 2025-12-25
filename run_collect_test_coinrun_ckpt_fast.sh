#!/bin/bash
# =============================================================================
# Test: Route Collection with Checkpointing (CoinRun) - FAST collector
#
# Mirrors: run_collect_test_coinrun_ckpt.sh
# Uses:    eval/collect_meta_routes_fast.py
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

EXP_NAME="test_coinrun_ckpt_fast"

MODEL_CKPT="/root/logs/ppo/meta-rl-coinrun-easy-step1024-n1k-trial10-gpu0=lr2e4/saved/model_step_100007936.tar"

BASE_OUT_DIR="/root/backup/kinematics/experiments"

# =============================================================================
# COLLECTION PARAMETERS (small for testing)
# =============================================================================

NUM_TASKS=20                 # Small number for quick test
NUM_PROCESSES=8              # Small batch size for test
DEVICE="cuda:0"              # GPU device
ADAPT_EPISODES=6             # Episodes for adaptation
RECORD_EPISODES=2            # Episodes after adaptation
SEED_OFFSET=0                # Starting seed

MAX_STEPS=64                 # Max steps per episode
MAX_EP_LEN=32                # Max episode length to keep

DISTRIBUTION_MODE="easy"     # "easy" or "hard"
REQUIRE_SUCCESS=1            # 1 = only keep successful trajectories

# SPEED OPTIMIZATION PARAMS
BATCH_COMPLETION=0.9         # Don't wait for slowest 10% of envs
TASK_TIMEOUT=0               # Total steps per task (0 = auto)
EARLY_ABORT=1                # Abort tasks with no success after adapt phase

# CHECKPOINT PARAMS
CKPT_SHARD_SIZE=5            # Flush every 5 routes (small for testing)
CKPT_FLUSH_SECS=10           # Flush every 10 seconds (short for testing)

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

mkdir -p "${DATA_DIR}" "${LOGS_DIR}"

export PYTHONWARNINGS="ignore::DeprecationWarning:gym"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║    TEST: META-RL ROUTE COLLECTION WITH CHECKPOINTING (FAST)  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Experiment:    ${EXP_NAME}"
echo "Model:         ${MODEL_CKPT}"
echo "Checkpoint:    ${CKPT_DIR}"
echo "Output:        ${ROUTES_NPZ}"
echo "Started:       $(date)"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

COLLECT_START=$(date +%s)

python -W ignore eval/collect_meta_routes_fast.py \
    --env_name coinrun \
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

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Checkpoint Info:"
echo ""

python -W ignore eval/routes_ckpt_tools.py info --ckpt_dir "${CKPT_DIR}"


