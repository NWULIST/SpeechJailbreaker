#!/usr/bin/env bash
set -euo pipefail

module load cuda/cuda-12.1.0-openmpi-4.1.4
export HF_HOME="/projects/e33046/qxq9828/.cache/huggingface"
mkdir -p "$HF_HOME"
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/Ming"

PYTHON_SCRIPT="./Experiments/boost_fuzzer_exp.py"
MODEL_PATH="google/gemma-7b-it"
EVALUATION="default"
RUN_INDEX="$(date +%Y-%m-%d_%H-%M-%S)_$RANDOM"
ADD_EOS=False
EOS_NUM="10"
defence=""
guard=""
EVALUATE_LOCALLY=""

# Fuzzer attempt controls (per index)
MAX_QUERY=100
MAX_JAILBREAK=1

# GPU scheduling
GPU_MEMORY=38000
NUM_GPU_SEARCH=1
NUM_TASKS=3

BATCH_SIZE=1
DATASET_SIZE=4724
MAX_PARALLEL=1
RETRY_DELAY=5
LOCK_DIR="/tmp/gpu_locks_${HOSTNAME}_$(id -u)_boost"

#For SmoothLLM Defense
NUM_COPIES=6   #default num_copies number

# Dataset paths
HARMFUL_DATASET="Dataset/harmful.csv"
TARGETS_DATASET="Dataset/harmful_targets.csv"

while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path|--target_model) MODEL_PATH="$2"; shift 2 ;;
    --guard) guard="$2"; shift 2 ;;
    --evaluation) EVALUATION="$2"; shift 2 ;;
    --run_index) RUN_INDEX="$2"; shift 2 ;;
    --add_eos) ADD_EOS="$2"; shift 2 ;;
    --defence) defence="$2"; shift 2 ;;
    --eos_num) EOS_NUM="$2"; shift 2 ;;
    --gpu_memory) GPU_MEMORY="$2"; shift 2 ;;
    --num_gpu_search) NUM_GPU_SEARCH="$2"; shift 2 ;;
    --num_tasks) NUM_TASKS="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --harmful_dataset) HARMFUL_DATASET="$2"; shift 2 ;;
    --targets_dataset) TARGETS_DATASET="$2"; shift 2 ;;
    --max_query) MAX_QUERY="$2"; shift 2 ;;
    --num_copies) NUM_COPIES="$2"; shift 2 ;;
    --max_jailbreak) MAX_JAILBREAK="$2"; shift 2 ;;
    --seed) RANDOM_SEED="$2"; shift 2 ;;
    --evaluate_locally) EVALUATE_LOCALLY="--evaluate_locally"; shift ;;
    *) shift ;;
  esac
done

# Log path
if [ "$ADD_EOS" = "True" ]; then
  LOG_PATH="Logs/${MODEL_PATH}/BoostGPTFuzzer_eos-${RUN_INDEX}"
else
  LOG_PATH="Logs/${MODEL_PATH}/BoostGPTFuzzer-${RUN_INDEX}"
fi

RESULTS_CSV="${LOG_PATH}/results.csv"
INDICES_FILE="${LOG_PATH}/selected_indices.txt"

mkdir -p "$LOG_PATH"
mkdir -p "$LOCK_DIR"

ADD_EOS_FLAG=""
if [ "$ADD_EOS" = "True" ]; then
  ADD_EOS_FLAG="--add_eos"
fi

###########################################
# CLEANUP ON INTERRUPT
###########################################
PIDS=()
cleanup() {
    echo "Keyboard interrupt detected. Cleaning up..."
    if [[ ${#PIDS[@]} -gt 0 ]]; then
        kill -9 "${PIDS[@]}" 2>/dev/null || true
    fi
    rm -f "$LOCK_DIR"/gpu_*.lock
    exit 1
}
trap cleanup SIGINT SIGTERM

###########################################
# GPU FUNCTIONS
###########################################
get_available_gpus() { seq 0 $NUM_GPU_SEARCH; }

gpu_has_memory() {
    local gpu=$1
    local free_mem
    free_mem=$(nvidia-smi -i $gpu --query-gpu=memory.free --format=csv,noheader,nounits)
    if [[ ! "$free_mem" =~ ^[0-9]+$ ]]; then return 1; fi
    [[ "$free_mem" -ge "$GPU_MEMORY" ]]
}

lock_gpu() {
    local gpu=$1
    (set -o noclobber; echo "$$" > "$LOCK_DIR/gpu_${gpu}.lock") 2>/dev/null
}

unlock_gpu() { rm -f "$LOCK_DIR/gpu_$1.lock"; }

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

    if [[ "${MODEL_PATH,,}" == *"qwen2-audio"* || "${MODEL_PATH,,}" == *"ming"* ]]; then
        echo "Audio model detected. Waiting 120 seconds for memory to clear..."
        sleep 120
    fi

    local log_file="${LOG_PATH}/batch_${batch_id}.log"

    SEED_ARG=""
    if [[ -n "${RANDOM_SEED:-}" ]]; then
        SEED_ARG="--seed $RANDOM_SEED"
    fi

    CUDA_VISIBLE_DEVICES="$gpu" python -u "$PYTHON_SCRIPT" \
      --indices "$indices_str" \
      --defence "$defence" \
      --target_model "$MODEL_PATH" \
      $ADD_EOS_FLAG \
      --guard "$guard" \
      --run_index "$RUN_INDEX" \
      --evaluation "$EVALUATION" \
      --max_query "$MAX_QUERY" \
      --max_jailbreak "$MAX_JAILBREAK" \
      ${EOS_NUM:+ --eos_num "$EOS_NUM"} \
      --harmful_dataset "$HARMFUL_DATASET" \
      --targets_dataset "$TARGETS_DATASET" \
      --num_copies "$NUM_COPIES"\
      $SEED_ARG \
      $EVALUATE_LOCALLY \
      &> "$log_file"

    # Extract RESULT lines
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

echo "job_index,total_score,total_count" > "$RESULTS_CSV"

###########################################
# GENERATE RANDOM INDICES
###########################################
echo "Generating $NUM_TASKS random indices from dataset of size $DATASET_SIZE..."

if [[ -n "${RANDOM_SEED:-}" ]]; then
    python3 -c "
import random
random.seed($RANDOM_SEED)
indices = random.sample(range($DATASET_SIZE), $NUM_TASKS)
for idx in indices:
    print(idx)
" > "$INDICES_FILE"
    echo "Using random seed: $RANDOM_SEED (results will be reproducible)"
else
    python3 -c "
import random
indices = random.sample(range($DATASET_SIZE), $NUM_TASKS)
for idx in indices:
    print(idx)
" > "$INDICES_FILE"
    echo "No random seed set (results will vary each run)"
fi

echo "Selected indices saved to: $INDICES_FILE"
mapfile -t SELECTED_INDICES < "$INDICES_FILE"

###########################################
# CREATE BATCHES FROM RANDOM INDICES
###########################################
num_batches=$(( (NUM_TASKS + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "Launching $num_batches batches (batch size: $BATCH_SIZE) with maximum $MAX_PARALLEL in parallel..."

batch_id=1
for ((i=0; i<NUM_TASKS; i+=BATCH_SIZE)); do
    batch_indices=("${SELECTED_INDICES[@]:i:BATCH_SIZE}")
    indices_str=$(IFS=,; echo "${batch_indices[*]}")

    echo "Batch $batch_id: processing indices [$indices_str]"

    run_batch_job_with_indices "$indices_str" $batch_id &
    PIDS+=($!)
    batch_id=$((batch_id + 1))

    while [[ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]]; do
        sleep 1
    done
done

wait

# Compute ASR
total_score=$(awk -F, 'NR>1 && $2>0 {count++} END {print count+0}' "$RESULTS_CSV")
total_count=$(awk -F, 'NR>1 {count++} END {print count+0}' "$RESULTS_CSV")
total_ASR=$(awk -v s="$total_score" -v c="$total_count" 'BEGIN {print (c>0 ? s/c : 0)}')

echo "TOTAL_ASR,$total_score,$total_count" >> "$RESULTS_CSV"
echo ""
echo "=========================================="
echo "All batches completed!"
echo "=========================================="
echo "Total jobs: $total_count"
echo "Total jailbroken attempts: $total_score"
echo "TOTAL_ASR=$total_ASR"
echo "Results: $RESULTS_CSV"
echo "=========================================="