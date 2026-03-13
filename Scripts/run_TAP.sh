#!/bin/bash
module load cuda/cuda-12.1.0-openmpi-4.1.4
export HF_HOME="/projects/e33046/.cache/"
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
#PYTHON_SCRIPT="../Experiments/tap_exp.py"
PYTHON_SCRIPT="Experiments/tap_exp.py"
MODEL_PATH="google/gemma-7b-it"
EVALUATION="default"
RUN_INDEX=2
ADD_EOS=False
EOS_NUM="10"
defence=""
guard=""
# GPU
GPU_MEMORY=40000
NUM_GPU_SEARCH=1
NUM_TASKS=100 # Number of tasks to run

#DATASET_SIZE=519              # Total size of your dataset
DATASET_SIZE=4724

RANDOM_SEED=42                 # Set to empty string for different samples each run
BATCH_SIZE=25                 # Process 10 items per GPU (adjust based on memory)
MAX_PARALLEL=2               # Maximum batches to run simultaneously
RETRY_DELAY=5

#For SmoothLLM Defense
NUM_COPIES=6   #default num_copies number

#LOCK_DIR="/tmp/gpu_locks"
#for dealing with stale lock issue
LOCK_DIR="/tmp/gpu_locks$(id -u)"

mkdir -p "$LOCK_DIR"

while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --guard)
      guard="$2"
      shift 2
      ;;
    --evaluation)
      EVALUATION="$2"
      shift 2
      ;;
    --gpu_memory)
      GPU_MEMORY="$2"
      shift 2
      ;;

    --num_copies)
      NUM_COPIES="$2"  
      shift 2
      ;;
    --num_gpu_search)
      NUM_GPU_SEARCH="$2"
      shift 2
      ;;
    --defence)
      defence="$2"
      shift 2
      ;;
    --num_tasks)
      NUM_TASKS="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done


LOG_PATH="Logs/${MODEL_PATH}/TAP"
RESULTS_PATH="Results/${MODEL_PATH}/TAP"

# Create the directories if it does not exist
mkdir -p "$LOG_PATH"
mkdir -p "$RESULTS_PATH"

run_identifier=$(date +"%Y-%m-%d_%H-%M")

# Output CSV file name
OUTPUT_FILE="${RESULTS_PATH}/TAP_${run_identifier}.csv"
# Create CSV header
echo "index,origin_question,iteration,self_id,parent_id,prompt,target_response,strongreject_score,success" > "$OUTPUT_FILE"

# Verify file creation
if [[ -f "$OUTPUT_FILE" ]]; then
    echo "CSV file '$OUTPUT_FILE' created successfully."
else
    echo "Error: Failed to create CSV file."
    exit 1
fi

INDICES_FILE="${LOG_PATH}/selected_indices_${run_identifier}.txt"


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
    echo "$free_mem"
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
    local log_file="${LOG_PATH}/${run_identifier}_${batch_id}.log"

    ###########################################
    # Run batch processing with specific indices
    ###########################################
    CUDA_VISIBLE_DEVICES=$gpu python -u "$PYTHON_SCRIPT" \
        --target_model "$MODEL_PATH" \
        --defence "$defence" \
        --evaluation "$EVALUATION" \
        --guard "$guard" \
        --indices "$indices_str" \
        --run_identifier "$run_identifier" \
        --batch_size "$BATCH_SIZE" \
        --num_copies "$NUM_COPIES" \
        &> "$log_file"



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

echo "All batches completed."
