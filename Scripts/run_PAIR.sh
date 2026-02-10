#!/usr/bin/env bash
set -e

###########################################
# MODULES & ENV
###########################################
module load cuda/cuda-12.1.0-openmpi-4.1.4
export HF_HOME="/projects/e33046/.cache/"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Load OpenAI API key from .env file
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded environment variables from .env"
elif [ -f "/projects/e33046/AttackBench/.env" ]; then
    export $(grep -v '^#' /projects/e33046/AttackBench/.env | xargs)
    echo "Loaded environment variables from /projects/e33046/AttackBench/.env"
else
    echo "WARNING: .env file not found!"
fi

# Verify OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set!"
    echo "Please create a .env file with: OPENAI_API_KEY=your-key-here"
    exit 1
else
    echo "OpenAI API key loaded: ${OPENAI_API_KEY:0:15}..."
fi

###########################################
# CONFIG
###########################################
PYTHON_SCRIPT="./Experiments/pair_exp.py"
MODEL_PATH="Qwen/Qwen2-Audio-7B-Instruct"  # Or use Qwen/Qwen2-Audio-7B-Instruct
EVALUATION="strongreject"
RUN_INDEX=2
defence=""
guard=""
GPU_MEMORY=40000               # Minimum free memory per GPU in MiB
NUM_GPU_SEARCH=7               # Highest GPU index to search
NUM_TASKS=400                  # Total tasks to run
DATASET_SIZE=4724              # Total size of your dataset
RANDOM_SEED=42                 # Set to empty string for different samples each run
BATCH_SIZE=200                  # Process 10 items per GPU (adjust based on memory)
MAX_PARALLEL=2               # Maximum batches to run simultaneously
RETRY_DELAY=5
LOCK_DIR="/tmp/gpu_locks"
LOG_PATH="Logs/${MODEL_PATH}/PAIR-${RUN_INDEX}"
RESULTS_CSV="${LOG_PATH}/results.csv"
INDICES_FILE="${LOG_PATH}/selected_indices.txt"

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
# RUN BATCH JOB
###########################################
run_batch_job_with_indices() {
    local indices_str=$1
    local batch_id=$2
    
    echo "Batch $batch_id (indices: $indices_str): waiting for a free GPU..."
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
        echo "Batch $batch_id: no GPU available, retrying in $RETRY_DELAY seconds..."
        sleep $RETRY_DELAY
    done

    echo "Batch $batch_id: running on GPU $gpu (indices: $indices_str)..."
    local log_file="${LOG_PATH}/batch_${batch_id}.log"

    ###########################################
    # Run batch processing with specific indices
    ###########################################
    CUDA_VISIBLE_DEVICES=$gpu python -u "$PYTHON_SCRIPT" \
        --target_model "$MODEL_PATH" \
        --defence "$defence" \
        --evaluation "$EVALUATION" \
        --guard "$guard" \
        --indices "$indices_str" \
        --n_iterations 10 \
        --n_streams 3 \
        --keep_last_n 2 \
        &> "$log_file"

    # Extract results for each item in the batch
    while IFS= read -r line; do
        if [[ $line =~ ^RESULT:([0-9]+),([0-9.]+),([0-9]+)$ ]]; then
            idx="${BASH_REMATCH[1]}"
            score="${BASH_REMATCH[2]}"
            count="${BASH_REMATCH[3]}"
            echo "$idx,$score,$count" >> "$RESULTS_CSV"
        fi
    done < <(grep '^RESULT:' "$log_file")

    echo "Batch $batch_id finished, unlocking GPU $gpu"
    unlock_gpu "$gpu"
}

###########################################
# MAIN
###########################################
if [[ "$NUM_TASKS" -le 0 ]]; then
    echo "NUM_TASKS <= 0. Nothing to run."
    exit 0
fi

if [[ "$NUM_TASKS" -gt "$DATASET_SIZE" ]]; then
    echo "ERROR: NUM_TASKS ($NUM_TASKS) exceeds DATASET_SIZE ($DATASET_SIZE)"
    exit 1
fi

# Write CSV header
echo "job_index,total_score,total_count" > "$RESULTS_CSV"

###########################################
# GENERATE RANDOM INDICES
###########################################
echo "Generating $NUM_TASKS random indices from dataset of size $DATASET_SIZE..."

if [[ -n "$RANDOM_SEED" ]]; then
    # Use Python to generate random indices with a seed for reproducibility
    python3 -c "
import random
random.seed($RANDOM_SEED)
indices = random.sample(range($DATASET_SIZE), $NUM_TASKS)
for idx in indices:
    print(idx)
" > "$INDICES_FILE"
    echo "Using random seed: $RANDOM_SEED (results will be reproducible)"
else
    # Generate random indices without seed (different each run)
    python3 -c "
import random
indices = random.sample(range($DATASET_SIZE), $NUM_TASKS)
for idx in indices:
    print(idx)
" > "$INDICES_FILE"
    echo "No random seed set (results will vary each run)"
fi

echo "Selected indices saved to: $INDICES_FILE"

# Read indices into array
mapfile -t SELECTED_INDICES < "$INDICES_FILE"

###########################################
# CREATE BATCHES FROM RANDOM INDICES
###########################################
num_batches=$(( (NUM_TASKS + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "Launching $num_batches batches (batch size: $BATCH_SIZE) with maximum $MAX_PARALLEL in parallel..."

batch_id=1
for ((i=0; i<NUM_TASKS; i+=BATCH_SIZE)); do
    # Get slice of indices for this batch
    batch_indices=("${SELECTED_INDICES[@]:i:BATCH_SIZE}")
    
    # Convert array to comma-separated string
    indices_str=$(IFS=,; echo "${batch_indices[*]}")
    
    echo "Batch $batch_id: processing indices [$indices_str]"
    
    run_batch_job_with_indices "$indices_str" $batch_id &
    PIDS+=($!)
    batch_id=$((batch_id + 1))

    # Wait if the number of running batches reaches MAX_PARALLEL
    while [[ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]]; do
        sleep 1
    done
done

# Wait for all remaining batches
wait

# Compute total ASR
total_score=$(awk -F, 'NR>1 {sum+=$2} END {print sum}' "$RESULTS_CSV")
total_count=$(awk -F, 'NR>1 {sum+=$3} END {print sum}' "$RESULTS_CSV")
total_ASR=$(awk -v s="$total_score" -v c="$total_count" 'BEGIN {print (c>0 ? s/c : 0)}')

echo "TOTAL_ASR,$total_ASR" >> "$RESULTS_CSV"
echo "All batches completed. TOTAL_ASR=$total_ASR"