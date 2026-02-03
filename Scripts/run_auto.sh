#!/usr/bin/env bash

module load cuda/cuda-12.1.0-openmpi-4.1.4
export HF_HOME="/projects/e33046/.cache/"

# Add project root to PYTHONPATH
#export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# # Get the project root (parent of Scripts directory)
# PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
# export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"
PYTHON_SCRIPT="./Experiments/autoattack_exp.py"
#MODEL_PATH="gpt-4o"
MODEL_PATH="Qwen/Qwen2-Audio-7B-Instruct"
EVALUATION="strongreject"
RUN_INDEX=0
defence=""
guard=""
# AutoAttack specific parameters
NORM="Linf"
EPS="0.3"
VERSION="standard"
DEVICE="cuda"

# GPU
GPU_MEMORY=40000
NUM_GPU_SEARCH=7
NUM_TASKS=3 # Number of tasks to run in parallel

# Dataset paths
# HARMFUL_DATASET="Dataset/harmful.csv"
# TARGETS_DATASET="Dataset/harmful_targets.csv"

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
    --run_index)
      RUN_INDEX="$2"
      shift 2
      ;;
    --defence)
      defence="$2"
      shift 2
      ;;
    --norm)
      NORM="$2"
      shift 2
      ;;
    --eps)
      EPS="$2"
      shift 2
      ;;
    --version)
      VERSION="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
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
    # --harmful_dataset)
    #   HARMFUL_DATASET="$2"
    #   shift 2
    #   ;;
    # --targets_dataset)
    #   TARGETS_DATASET="$2"
    #   shift 2
    #   ;;
    *)
      shift
      ;;
  esac
done

# Set the log path
LOG_PATH="Logs/${MODEL_PATH}/AutoAttack-${RUN_INDEX}"

echo "=========================================="
echo "AutoAttack Script Started"
echo "=========================================="
echo "Model Path: $MODEL_PATH"
echo "Evaluation: $EVALUATION"
echo "Run Index: $RUN_INDEX"
echo "Norm: $NORM"
echo "Epsilon: $EPS"
echo "Version: $VERSION"
echo "Number of Tasks: $NUM_TASKS"
echo "Log Path: $LOG_PATH"
echo "GPU Memory Required: $GPU_MEMORY MB"
echo "=========================================="
echo ""

# Create the log directory if it does not exist
mkdir -p "$LOG_PATH"
echo "[INFO] Created log directory: $LOG_PATH"

# Function to find the first available GPU
find_free_gpu() {
    # {0..$NUM_GPU_SEARCH}
    for i in $(seq 0 $NUM_GPU_SEARCH); do
        free_mem=$(nvidia-smi -i $i --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print $1}')
        if [[ "$free_mem" =~ ^[0-9]+$ ]] && [ "$free_mem" -ge $GPU_MEMORY ]; then
            echo $i
            return
        fi
    done

    echo "-1" # Return -1 if no suitable GPU is found
}

# Start the jobs with GPU assignment
echo "[INFO] Starting task scheduling..."
echo ""

for index in $(seq 0 $NUM_TASKS); do
    echo "----------------------------------------"
    echo "[TASK $index] Searching for available GPU..."

    FREE_GPU=-1
    retry_count=0

    # Keep looping until a free GPU is found
    while [ $FREE_GPU -eq -1 ]; do
        FREE_GPU=$(find_free_gpu)
        if [ $FREE_GPU -eq -1 ]; then
            retry_count=$((retry_count + 1))
            echo "[TASK $index] No GPU available (attempt $retry_count). Waiting 5 seconds..."
            sleep 5 # Wait for 5 seconds before trying to find a free GPU again
        fi
    done

    echo "[TASK $index] Found GPU $FREE_GPU with sufficient memory!"
    echo "[TASK $index] Launching task on GPU $FREE_GPU..."

    # Run the Python script on the free GPU
    (
        echo "[TASK $index] Task started on GPU $FREE_GPU at $(date '+%Y-%m-%d %H:%M:%S')"
        echo "CMD: CUDA_VISIBLE_DEVICES=$FREE_GPU python -u $PYTHON_SCRIPT --index $index --defence $defence --target_model $MODEL_PATH --norm $NORM --guard $guard --eps $EPS --version $VERSION --device $DEVICE --run_index $RUN_INDEX --evaluation $EVALUATION > ${LOG_PATH}/${index}.log 2>&1" >> ${LOG_PATH}/${index}.log

        echo "[TASK $index] Loading model and starting AutoAttack..."
        CUDA_VISIBLE_DEVICES=$FREE_GPU python -u "$PYTHON_SCRIPT" --index $index --defence $defence --target_model $MODEL_PATH --norm $NORM --guard $guard --eps $EPS --version $VERSION --device $DEVICE --run_index $RUN_INDEX --evaluation $EVALUATION 2>&1 | tee "${LOG_PATH}/${index}.log"

        echo "[TASK $index] Task finished on GPU $FREE_GPU at $(date '+%Y-%m-%d %H:%M:%S')"
    ) 

    TASK_PID=$!
    echo "[TASK $index] Background process PID: $TASK_PID"
    echo "[TASK $index] Log file: ${LOG_PATH}/${index}.log"
    echo "[TASK $index] Monitor with: tail -f ${LOG_PATH}/${index}.log"

    # Wait for 30 seconds to give the GPU some time to allocate memory
    if [ $index -lt $NUM_TASKS ]; then
        echo "[INFO] Waiting 30 seconds before launching next task..."
        sleep 30
        echo ""
    fi
done

echo ""
echo "=========================================="
echo "All tasks have been launched!"
echo "=========================================="
echo "Total tasks: $((NUM_TASKS + 1))"
echo "Waiting for all tasks to complete..."
echo "Monitor progress with:"
echo "  tail -f ${LOG_PATH}/*.log"
echo "=========================================="
echo ""

# Wait for all background jobs to finish
wait

echo ""
echo "=========================================="
echo "All tasks completed!"
echo "=========================================="
echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results saved in: $LOG_PATH"
echo "=========================================="
