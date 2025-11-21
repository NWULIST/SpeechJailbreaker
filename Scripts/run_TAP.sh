#!/bin/bash
module load cuda/cuda-12.1.0-openmpi-4.1.4
export HF_HOME="/projects/e33046/.cache/"
# Add project root to PYTHONPATH
#export PYTHONPATH="${PYTHONPATH}:$(pwd)"
PYTHON_SCRIPT="/home/vaz5542/projects/AttackBench/Experiments/tap_exp.py"
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
