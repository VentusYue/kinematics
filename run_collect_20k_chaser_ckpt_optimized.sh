#!/bin/bash
# =============================================================================
# Meta-RL Route Collection (Chaser) with Checkpointing - OPTIMIZED collector
#
# Based on:
# - run_collect_20k_coinrun_ckpt_optimized.sh
# - eval/collect_meta_routes_optimized.py
#
# Notes for Chaser:
# - Procgen gym exposes `prev_level_complete` (not `level_complete`) in info; the
#   optimized collector maps this into success tracking (see collector code).
# - For Chaser, reward can be positive without finishing the level (orb pickups),
#   so we recommend `--success_policy=prefer_level_complete` to ensure "success"
#   means full level completion when the signal exists.
#
# Usage:
#   ./run_collect_20k_chaser_ckpt_optimized.sh              # Start or resume collection
#   ./run_collect_20k_chaser_ckpt_optimized.sh --fresh      # Force fresh start (delete old checkpoint)
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
# CONFIGURATION
# =============================================================================

EXP_NAME="run_20k_chaser_ckpt_optimized"

MODEL_CKPT="/root/logs/ppo/meta-rl-chaser-easy-step1024-n1k-trial10-gpu1=lr2e4/saved/model_step_128319488.tar"

BASE_OUT_DIR="/root/backup/kinematics/experiments"

# =============================================================================
# COLLECTION PARAMETERS
# =============================================================================

NUM_TASKS=20000            # Target number of SUCCESSFUL trajectories
NUM_PROCESSES=128          # Parallel environments per batch
DEVICE="cuda:0"            # cuda:0, cuda:1, cpu, etc.
ADAPT_EPISODES=6           # Episodes for adaptation (sampling)
RECORD_EPISODES=2          # Episodes after adaptation (typically deterministic)
SEED_OFFSET=0              # Starting seed

MAX_STEPS=512              # Max steps per episode
MAX_EP_LEN=256             # Max episode length to keep (also used by selector)

DISTRIBUTION_MODE="easy"   # "easy" or "hard"
REQUIRE_SUCCESS=1          # 1 = only keep successful trajectories

# Chaser-specific: reward can be positive without completion; prefer completion signal.
SUCCESS_POLICY="prefer_level_complete"  # legacy | prefer_level_complete

# SPEED OPTIMIZATION PARAMS
BATCH_COMPLETION=0.9       # Don't wait for slowest 10% of envs
TASK_TIMEOUT=0             # Total steps per task (0 = auto)
EARLY_ABORT=1              # Abort tasks with no success after adapt phase

# PERF FLAGS (supported by collect_meta_routes_optimized.py)
USE_COMPILE=1              # torch.compile (can increase speed; may increase startup time)
USE_AMP=1                  # AMP on CUDA

# CHECKPOINT PARAMS
CKPT_SHARD_SIZE=1000       # Flush every N routes (balance: safety vs file count)
CKPT_FLUSH_SECS=600        # Flush at least every 10 minutes

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

# Handle --fresh option
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
echo "║   META-RL ROUTE COLLECTION (Chaser) WITH CHECKPOINTING (OPT) ║"
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
echo "  ├── Env:                  chaser"
echo "  ├── Device:               ${DEVICE}"
echo "  ├── Target trajectories:  ${NUM_TASKS}"
echo "  ├── Batch size:           ${NUM_PROCESSES}"
echo "  ├── Adapt episodes:       ${ADAPT_EPISODES}"
echo "  ├── Record episodes:      ${RECORD_EPISODES}"
echo "  ├── Max steps/episode:    ${MAX_STEPS}"
echo "  ├── Max episode length:   ${MAX_EP_LEN}"
echo "  ├── Require success:      ${REQUIRE_SUCCESS}"
echo "  ├── Success policy:       ${SUCCESS_POLICY}"
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
    --env_name chaser \
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
    --success_policy="${SUCCESS_POLICY}" \
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

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo "  1. Run analysis: ./run_analysis_20k_chaser_ckpt_optimized.sh"
echo ""
echo "  Or if collection was interrupted, you can:"
echo "  2. Check progress:    python eval/routes_ckpt_tools.py info --ckpt_dir ${CKPT_DIR}"
echo "  3. Export partial:    python eval/routes_ckpt_tools.py build --ckpt_dir ${CKPT_DIR} --out_npz ${DATA_DIR}/routes_partial.npz"
echo "  4. Resume collection: ./run_collect_20k_chaser_ckpt_optimized.sh"
echo ""


