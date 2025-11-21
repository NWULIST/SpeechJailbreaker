
#!/bin/bash
module load cuda/cuda-12.1.0-openmpi-4.1.4
export HF_HOME="/projects/e33046/.cache/"
# Add project root to PYTHONPATH
#export PYTHONPATH="${PYTHONPATH}:$(pwd)"
PYTHON_SCRIPT="./Experiments/tap_exp.py"
#MODEL_PATH="google/gemma-7b-it"
#MODEL_PATH="gpt-4o"
#MODEL_PATH="Qwen/Qwen2-Audio-7B-Instruct"
#MODEL_PATH="google/gemma-3n-E4B-it"
MODEL_PATH="google/gemma-3n-E2B-it"
#EVALUATION="default"
EVALUATION="strongreject"
RUN_INDEX=2


# GPU
GPU_MEMORY=40000
NUM_GPU_SEARCH=7
NUM_TASKS=1 # Number of tasks to run in parallel

# Dataset paths
# HARMFUL_DATASET="Dataset/harmful.csv"
# TARGETS_DATASET="Dataset/harmful_targets.csv"

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



LOG_PATH="Logs/${MODEL_PATH}/TAP-${RUN_INDEX}"



# Create the log directory if it does not exist
LOG_PATH="Logs/${MODEL_PATH}/TAP-${RUN_INDEX}"

echo "=========================================="
echo "TAP Attack Script Started"
echo "=========================================="
echo "Model Path: $MODEL_PATH"
echo "Evaluation: $EVALUATION"
echo "Run Index: $RUN_INDEX"
echo "Number of Tasks: $NUM_TASKS"
echo "Log Path: $LOG_PATH"
echo "GPU Memory Required: $GPU_MEMORY MB"
echo "=========================================="
echo ""

mkdir -p "$LOG_PATH"
echo "[INFO] Created log directory: $LOG_PATH"


# Conditional flag for EARLY_STOP
EARLY_STOP_FLAG=""
if [ "$EARLY_STOP" = "True" ]; then
    EARLY_STOP_FLAG="--early_stop"
fi

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
for index in $(seq 0 $NUM_TASKS); do

    FREE_GPU=-1

    # Keep looping until a free GPU is found
    while [ $FREE_GPU -eq -1 ]; do
        FREE_GPU=$(find_free_gpu)
        if [ $FREE_GPU -eq -1 ]; then
            sleep 5 # Wait for 5 seconds before trying to find a free GPU again
        fi
    done

    (
        echo "Task $index started on GPU $FREE_GPU."
        echo "CMD: CUDA_VISIBLE_DEVICES=$FREE_GPU python -u $PYTHON_SCRIPT --target_model $MODEL_PATH    --evaluation $EVALUATION  --index $index  > ${LOG_PATH}/${index}.log 2>&1" >> ${LOG_PATH}/${index}.log
        CUDA_VISIBLE_DEVICES=$FREE_GPU python -u "$PYTHON_SCRIPT"  --target_model $MODEL_PATH  --evaluation $EVALUATION --index $index 2>&1 | tee "${LOG_PATH}/${index}.log"
        echo "Task $index on GPU $FREE_GPU finished."
    ) &

    # Wait for 30 seconds to give the GPU some time to allocate memory
    sleep 30
done

# Wait for all background jobs to finish
wait
