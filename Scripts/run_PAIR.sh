#!/bin/bash
module load cuda/cuda-12.1.0-openmpi-4.1.4
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
PYTHON_SCRIPT="../Experiments/pair_exp.py"
MODEL_PATH="Qwen/Qwen2-Audio-7B-Instruct"
EVALUATION="default"
RUN_INDEX=2
ADD_EOS=False
EOS_NUM="10"
defence=""
guard=""
# GPU
GPU_MEMORY=40000
NUM_GPU_SEARCH=7
NUM_TASKS=1 # Number of tasks to run in parallel



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
    --defence)
      defence="$2"
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
    *)
      shift
      ;;
  esac
done


LOG_PATH="Logs/${MODEL_PATH}/PAIR-${RUN_INDEX}"
# Create the log directory if it does not exist
mkdir -p "$LOG_PATH"


# Start the jobs with GPU assignment
for index in $(seq 0 $NUM_TASKS); do


    (
        echo "Task $index started on GPU."
        echo "CMD: python -u $PYTHON_SCRIPT --target_model $MODEL_PATH --defence $defence --evaluation $EVALUATION --guard $guard --index $index" 
        echo "CMD: python -u $PYTHON_SCRIPT --target_model $MODEL_PATH --defence $defence --evaluation $EVALUATION --guard $guard --index $index  > ${LOG_PATH}/${index}.log 2>&1" >> ${LOG_PATH}/${index}.log
        python -u "$PYTHON_SCRIPT" --target_model $MODEL_PATH --defence $defence --evaluation $EVALUATION --guard $guard --index $index > "${LOG_PATH}/${index}.log" 2>&1
        echo "Task $index on GPU finished."
    ) 

    # Wait for 30 seconds to give the GPU some time to allocate memory
    sleep 60
done

# Wait for all background jobs to finish
wait