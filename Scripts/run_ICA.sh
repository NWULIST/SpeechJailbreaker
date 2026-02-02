#!/bin/bash
module load cuda/cuda-12.1.0-openmpi-4.1.4
export HF_HOME="/projects/e33046/.cache/"
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

PYTHON_SCRIPT="./Experiments/ica_exp.py"
MODEL_PATH="google/gemma-7b-it"
EVALUATION="default"
RUN_INDEX=2
ADD_EOS=False
FEW_SHOT_NUM=0

# GPU
GPU_MEMORY=60000
#NUM_GPU_SEARCH=7
#changing to 0 becuase we are only testing on one GPU
NUM_GPU_SEARCH=0
#NUM_TASKS=3 # Number of tasks to run in parallel

#changing to 2 tasks just to test
NUM_TASKS=2
MAX_PARALLEL=1


RETRY_DELAY=5
LOCK_DIR="/tmp/gpu_locks"

# Dataset paths
HARMFUL_DATASET="Dataset/harmful.csv"
TARGETS_DATASET="Dataset/harmful_targets.csv"
defence=""
guard=""

################################
# PARSE ARGUMENTS
###############################

while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --evaluation)
      EVALUATION="$2"
      shift 2
      ;;
    --add_eos)
      ADD_EOS="$2"
      shift 2
      ;;
    --eos_num)
      EOS_NUM="$2"
      shift 2
      ;;
    --gpu_memory)
      GPU_MEMORY="$2"
      shift 2
      ;;
    --num_gpu_search)
      NUM_GPU_SEARCH="$2"
      shift 2
      ;;
    --num_tasks)
      NUM_TASKS="$2"
      shift 2
      ;;
    --defence)
      defence="$2"
      shift 2
      ;;
    --few_shot_num)
      FEW_SHOT_NUM="$2"
      shift 2
      ;;
    --guard)
      guard="$2"
      shift 2
      ;;
    --harmful_dataset)
      HARMFUL_DATASET="$2"
      shift 2
      ;;
    --targets_dataset)
      TARGETS_DATASET="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done


if [ "$ADD_EOS" = "True" ]; then
    LOG_PATH="Logs/${MODEL_PATH}/ICA_eos-${RUN_INDEX}"
else
    LOG_PATH="Logs/${MODEL_PATH}/ICA-${RUN_INDEX}"
fi

RESULTS_CSV="${LOG_PATH}/results.csv"

# Create the log directory if it does not exist
mkdir -p "$LOG_PATH"
mkdir -p "$LOCK_DIR"

# Conditional flag for EARLY_STOP
EARLY_STOP_FLAG=""
if [ "$EARLY_STOP" = "True" ]; then
    EARLY_STOP_FLAG="--early_stop"
fi

###########################################
# CLEANUP ON INTERRUPT
###########################################
PIDS=()
cleanup() {
    echo "Keyboard interrupt detected. Cleaning up..."
    if [[ ${#PIDS[@]} -gt 0 ]]; then
        echo "Killing background jobs..."
        kill -9 "${PIDS[@]}" 2>/dev/null || true
    fi
    echo "Removing GPU lock files..."
    rm -f "$LOCK_DIR"/gpu_*.lock
    exit 1
}
trap cleanup SIGINT SIGTERM
###########################################
# GPU FUNCTIONS (COPY FROM JBC)
###########################################
get_available_gpus() { seq 0 $NUM_GPU_SEARCH; }

gpu_has_memory() {
    local gpu=$1
    local free_mem
    free_mem=$(nvidia-smi -i $gpu --query-gpu=memory.free --format=csv,noheader,nounits)
    [[ "$free_mem" -ge "$GPU_MEMORY" ]]
}

lock_gpu() {
    local gpu=$1
    (set -o noclobber; echo "$$" > "$LOCK_DIR/gpu_${gpu}.lock") 2>/dev/null
}

unlock_gpu() {
    rm -f "$LOCK_DIR/gpu_$1.lock"
}

###########################################
# ICA JOB FUNCTION
###########################################
run_job() {
    local idx=$1
    local shot=$2

    echo "Job prompt=$idx fs=$shot waiting for GPU..."
    local gpu=-1
    while true; do
        for g in $(get_available_gpus); do
            if lock_gpu "$g"; then
                if gpu_has_memory "$g"; then gpu=$g; break
                else unlock_gpu "$g"; fi
            fi
        done
        [[ $gpu -ge 0 ]] && break
        sleep $RETRY_DELAY
    done

    local log="${LOG_PATH}/prompt_${idx}.log"
    echo "Job prompt=$idx fs=$shot running on GPU $gpu"

    CUDA_VISIBLE_DEVICES=$gpu python -u "$PYTHON_SCRIPT" \
      --target_model "$MODEL_PATH" \
      --evaluation "$EVALUATION" \
      --few_shot_num "$FEW_SHOT_NUM" \
      --prompt_index "$idx" \
      --harmful_dataset "$HARMFUL_DATASET" \
      --targets_dataset "$TARGETS_DATASET" \
      --defence "$defence" \
      --guard "$guard" \
      &> "$log"

    # extract RESULT lines later if desired
    unlock_gpu "$gpu"
}

###########################################
# MAIN
###########################################
if [[ "$NUM_TASKS" -le 0 ]]; then
    echo "NUM_TASKS <= 0. Nothing to run."
    exit 0
fi

# Write CSV header
echo "job_index,total_score,total_count" > "$RESULTS_CSV"

echo "Launching $NUM_TASKS jobs with maximum $MAX_PARALLEL in parallel..."

for idx in $(seq 0 $((NUM_TASKS-1))); do
    run_job "$idx" "$FEW_SHOT_NUM" &
    PIDS+=($!)

    while [[ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]]; do
        sleep 1
    done
done

# Wait for all remaining jobs
wait

# Compute total ASR
total_score=$(awk -F, 'NR>1 {sum+=$2} END {print sum}' "$RESULTS_CSV")
total_count=$(awk -F, 'NR>1 {sum+=$3} END {print sum}' "$RESULTS_CSV")
total_ASR=$(awk -v s="$total_score" -v c="$total_count" 'BEGIN {print (c>0 ? s/c : 0)}')

echo "TOTAL_ASR,$total_ASR" >> "$RESULTS_CSV"
echo "All jobs completed. TOTAL_ASR=$total_ASR"