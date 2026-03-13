#!/usr/bin/env bash
set -e

###########################################
# MODULES & ENV
###########################################
module load cuda/cuda-12.1.0-openmpi-4.1.4
export HF_HOME="/projects/e33046/.cache/"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

###########################################
# CONFIG
###########################################
#PYTHON_SCRIPT="./Experiments/jbc_exp.py"
PYTHON_SCRIPT="Experiments/jbc_exp.py"

MODEL_PATH="Qwen/Qwen2-Audio-7B-Instruct"
EVALUATION="strongreject"
RUN_INDEX="$(date +%Y-%m-%d_%H-%M-%S)_$RANDOM"
defence=""
guard=""
GPU_MEMORY=40000               # Minimum free memory per GPU in MiB
#NUM_GPU_SEARCH=7             # Highest GPU index to search

 
NUM_GPU_SEARCH=1

#start with 2 just to gauge that it works
#NUM_TASKS=2

#testing on one gpu 
MAX_PARALLEL=2

#BATCH_SIZE=1
DATASET_SIZE=4724

#For SmoothLLM Defense
NUM_COPIES=6   #default num_copies number

RETRY_DELAY=5
#original
#LOCK_DIR="/tmp/gpu_locks"

#for dealing with stale lock issue
LOCK_DIR="/tmp/gpu_locks$(id -u)"

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
    
    --batch_size)
      BATCH_SIZE="$2"
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

    --num_copies)
    NUM_COPIES="$2"  
    shift 2
    ;;

    --seed)
      RANDOM_SEED="$2"
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
    SEED_ARG=""
    if [[ -n "$RANDOM_SEED" ]]; then
    SEED_ARG="--seed $RANDOM_SEED"
    fi

    CUDA_VISIBLE_DEVICES=$gpu python -u "$PYTHON_SCRIPT" \
        --target_model "$MODEL_PATH" \
        --defence "$defence" \
        --evaluation "$EVALUATION" \
        --guard "$guard" \
        --indices "$indices_str" \
        --max-new-tokens 256 \
        --num_copies "$NUM_COPIES"\
        $SEED_ARG\
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
for ((i=0; i<$NUM_TASKS; i+=$BATCH_SIZE)); do
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

# Compute total ASR(per attempt)
total_score=$(awk -F, 'NR>1 {sum+=$2} END {print sum}' "$RESULTS_CSV")
total_count=$(awk -F, 'NR>1 {sum+=$3} END {print sum}' "$RESULTS_CSV")
total_ASR=$(awk -v s="$total_score" -v c="$total_count" 'BEGIN {print (c>0 ? s/c : 0)}')

#Compute total ASR(per prompt)
total_prompts=$(awk -F, 'NR>1 {count++} END {print count}' "$RESULTS_CSV")
successful_prompts=$(awk -F, 'NR>1 && $2>0 {count++} END {print count}' "$RESULTS_CSV")
prompt_ASR=$(awk -v s="$successful_prompts" -v p="$total_prompts" 'BEGIN {print (p>0 ? s/p : 0)}')

echo "PER_ATTEMPT_ASR,$total_ASR" >> "$RESULTS_CSV"
echo "All batches completed. PER_ATTEMPT_ASR=$total_ASR"
echo "PER_PROMPT_ASR,$prompt_ASR" >> "$RESULTS_CSV"
echo "All batches completed. PER_PROMPT_ASR=$prompt_ASR"