#!/bin/bash
# =============================================================================
# Meta-RL Route Collection (Optimized with Filtering) - Chaser
#
# Based on: /root/backup/kinematics/run_collect.sh
# Updated for:
# - Env: chaser
# - Checkpoint: model_step_114425856.tar
# - Longer max steps / episode length for chaser
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

EXP_NAME="run_random_seeds_chaser_10k_step114425856"
MODEL_CKPT="/root/logs/ppo/meta-rl-chaser-easy-step1024-n1k-trial10-gpu1=lr2e4/saved/model_step_114425856.tar"
BASE_OUT_DIR="/root/backup/kinematics/experiments"

# =============================================================================
# COLLECTION PARAMETERS
# =============================================================================

ENV_NAME="chaser"
NUM_TASKS=10000            # Target number of SUCCESSFUL trajectories
NUM_PROCESSES=128          # Parallel environments per batch
DEVICE="cuda:1"            # GPU device (cuda:0, cuda:1, etc.)
ADAPT_EPISODES=6           # Episodes for adaptation (sampling)
RECORD_EPISODES=2          # Episodes after adaptation (typically deterministic)
SEED_OFFSET=0              # Starting seed

# Longer episodes for chaser
MAX_STEPS=1024             # Max steps per episode
MAX_EP_LEN=1024            # Max episode length to keep (also used by selector)

DISTRIBUTION_MODE="easy"   # "easy" or "hard"
REQUIRE_SUCCESS=1          # 1 = only keep successful trajectories

# SPEED OPTIMIZATION PARAMS
BATCH_COMPLETION=0.9       # Don't wait for slowest 10% of envs
TASK_TIMEOUT=0             # Total steps per task (0 = auto)
EARLY_ABORT=1              # Abort tasks with no success after adapt phase

# =============================================================================
# DERIVED PATHS
# =============================================================================

EXP_DIR="${BASE_OUT_DIR}/${EXP_NAME}"
DATA_DIR="${EXP_DIR}/data"
LOGS_DIR="${EXP_DIR}/logs"
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
echo "║         META-RL ROUTE COLLECTION (Optimized)                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Env:           ${ENV_NAME}"
echo "Experiment:    ${EXP_NAME}"
echo "Model:         ${MODEL_CKPT}"
echo "Output:        ${ROUTES_NPZ}"
echo "Started:       $(date)"
echo ""
echo "Parameters:"
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

python -W ignore eval/collect_meta_routes.py \
    --env_name "${ENV_NAME}" \
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
    --xy_fail_policy="warn_only"

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
echo "Finished:      $(date)"
echo ""


