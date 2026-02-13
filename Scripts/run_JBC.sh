#!/usr/bin/env bash
set -e

###########################################
# MODULES & ENV
###########################################
module load cuda/cuda-12.1.0-openmpi-4.1.4
export HF_HOME="/projects/e33046/.cache/"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_LAUNCH_BLOCKING=1 #forces error reporting 

###########################################
# CONFIG
###########################################
#PYTHON_SCRIPT="./Experiments/jbc_exp.py"
PYTHON_SCRIPT="Experiments/jbc_exp.py"

MODEL_PATH="Qwen/Qwen2-Audio-7B-Instruct"
EVALUATION="strongreject"
RUN_INDEX=2
defence=""
guard=""
GPU_MEMORY=40000               # Minimum free memory per GPU in MiB
#NUM_GPU_SEARCH=7             # Highest GPU index to search

#change to 0 since I am using only one gpu 
NUM_GPU_SEARCH=0

#NUM_TASKS=4723                  # Total tasks to run

#start with 2 just to gauge that it works
NUM_TASKS=3

#MAX_PARALLEL=2                 # Maximum jobs to run simultaneously

#testing on one gpu 
MAX_PARALLEL=1

RETRY_DELAY=5
LOCK_DIR="/tmp/gpu_locks"

#add parsing so logging files are correctly labeled for the appropriate model
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
    --run_index)
      RUN_INDEX="$2"
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

LOG_PATH="Logs/${MODEL_PATH}/JBC-${RUN_INDEX}"
RESULTS_CSV="${LOG_PATH}/results.csv"

mkdir -p "$LOG_PATH"
mkdir -p "$LOCK_DIR"

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
# GPU FUNCTIONS
###########################################
get_available_gpus() {
    seq 0 $NUM_GPU_SEARCH
}

gpu_has_memory() {
    local gpu=$1
    local free_mem
    free_mem=$(nvidia-smi -i $gpu --query-gpu=memory.free --format=csv,noheader,nounits)

    echo "GPU $gpu free memory (before return): '$free_mem'"

    # Ensure it's a number
    if [[ ! "$free_mem" =~ ^[0-9]+$ ]]; then
        echo "GPU $gpu: invalid memory info, skipping..."
        return 1
    fi

    [[ "$free_mem" -ge "$GPU_MEMORY" ]]

}

lock_gpu() {
    local gpu=$1
    local lockfile="$LOCK_DIR/gpu_${gpu}.lock"
    ( set -o noclobber; echo "$$" > "$lockfile") 2>/dev/null
}

unlock_gpu() {
    local gpu=$1
    rm -f "$LOCK_DIR/gpu_${gpu}.lock"
}

###########################################
# RUN JOB
###########################################
run_job() {
    local idx=$1
    echo "Job $idx: waiting for a free GPU..."
    local gpu=-1

    while true; do
        for g in $(get_available_gpus); do
            if lock_gpu "$g"; then
                if gpu_has_memory "$g"; then
                    gpu=$g
                    break
                else
                    unlock_gpu "$g"
                fi
            fi
        done
        [[ $gpu -ge 0 ]] && break
        echo "Job $idx: no GPU available, retrying in $RETRY_DELAY seconds..."
        sleep $RETRY_DELAY
    done

    echo "Job $idx: running on GPU $gpu..."
    local log_file="${LOG_PATH}/job_${idx}.log"

    ###########################################
    # Capture all output, then extract RESULT from log
    ###########################################
    CUDA_VISIBLE_DEVICES=$gpu python -u "$PYTHON_SCRIPT" \
    --target_model "$MODEL_PATH" \
    --defence "$defence" \
    --evaluation "$EVALUATION" \
    --guard "$guard" \
    --index "$idx" \
    &> "$log_file"

    # Extract the most recent RESULT line
    result=$(grep '^RESULT:' "$log_file" | tail -1)

    # Parse result
    total_score=$(echo "$result" | cut -d':' -f2 | cut -d',' -f1)
    total_count=$(echo "$result" | cut -d':' -f2 | cut -d',' -f2)

    echo "$idx,$total_score,$total_count" >> "$RESULTS_CSV"

    echo "Job $idx finished, unlocking GPU $gpu"
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

for idx in $(seq 1 $NUM_TASKS); do
    run_job $idx &
    PIDS+=($!)

    # Wait if the number of running jobs reaches MAX_PARALLEL
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