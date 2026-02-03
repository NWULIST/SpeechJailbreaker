#!/usr/bin/env bash
set -euo pipefail

module load cuda/cuda-12.1.0-openmpi-4.1.4
export HF_HOME="/projects/e33046/.cache/"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_SCRIPT="./Experiments/boost_fuzzer_exp.py"
MODEL_PATH="google/gemma-7b-it"
EVALUATION="default"
RUN_INDEX=2
ADD_EOS=False
EOS_NUM="10"
defence=""
guard=""

# Fuzzer attempt controls (per index)
MAX_QUERY=100
MAX_JAILBREAK=1

# GPU scheduling
GPU_MEMORY=40000
NUM_GPU_SEARCH=7
NUM_TASKS=3   # number of indices to run (0..NUM_TASKS-1)

# Dataset paths
HARMFUL_DATASET="Dataset/harmful.csv"
TARGETS_DATASET="Dataset/harmful_targets.csv"

while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --guard) guard="$2"; shift 2 ;;
    --evaluation) EVALUATION="$2"; shift 2 ;;
    --run_index) RUN_INDEX="$2"; shift 2 ;;
    --add_eos) ADD_EOS="$2"; shift 2 ;;
    --defence) defence="$2"; shift 2 ;;
    --eos_num) EOS_NUM="$2"; shift 2 ;;
    --gpu_memory) GPU_MEMORY="$2"; shift 2 ;;
    --num_gpu_search) NUM_GPU_SEARCH="$2"; shift 2 ;;
    --num_tasks) NUM_TASKS="$2"; shift 2 ;;
    --harmful_dataset) HARMFUL_DATASET="$2"; shift 2 ;;
    --targets_dataset) TARGETS_DATASET="$2"; shift 2 ;;
    --max_query) MAX_QUERY="$2"; shift 2 ;;
    --max_jailbreak) MAX_JAILBREAK="$2"; shift 2 ;;
    *) shift ;;
  esac
done

# Log path
if [ "$ADD_EOS" = "True" ]; then
  LOG_PATH="Logs/${MODEL_PATH}/GPTFuzzer_eos-${RUN_INDEX}"
else
  LOG_PATH="Logs/${MODEL_PATH}/GPTFuzzer-${RUN_INDEX}"
fi
mkdir -p "$LOG_PATH"

ADD_EOS_FLAG=""
if [ "$ADD_EOS" = "True" ]; then
  ADD_EOS_FLAG="--add_eos"
fi

# GPU lock directory (prevents multiple tasks claiming same GPU)
LOCKDIR="/tmp/${USER}_gpu_locks_boost_fuzzer"
mkdir -p "$LOCKDIR"

find_free_gpu() {
  for i in $(seq 0 $NUM_GPU_SEARCH); do
    free_mem=$(nvidia-smi -i "$i" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | awk '{print $1}')
    if [[ "$free_mem" =~ ^[0-9]+$ ]] && [ "$free_mem" -ge "$GPU_MEMORY" ]; then
      # Claim atomically
      if mkdir "$LOCKDIR/gpu_$i" 2>/dev/null; then
        echo "$i"
        return
      fi
    fi
  done
  echo "-1"
}

# Launch NUM_TASKS indices: 0..NUM_TASKS-1
for ((index=0; index<NUM_TASKS; index++)); do
  FREE_GPU=-1
  while [ "$FREE_GPU" -eq -1 ]; do
    FREE_GPU=$(find_free_gpu)
    if [ "$FREE_GPU" -eq -1 ]; then
      sleep 5
    fi
  done

  (
    # Ensure lock is released even if python dies
    cleanup() { rmdir "$LOCKDIR/gpu_$FREE_GPU" 2>/dev/null || true; }
    trap cleanup EXIT

    echo "Task $index started on GPU $FREE_GPU."
    {
      echo "Task $index started on GPU $FREE_GPU."
      echo "CMD: CUDA_VISIBLE_DEVICES=$FREE_GPU python -u $PYTHON_SCRIPT --index $index --defence $defence --target_model $MODEL_PATH $ADD_EOS_FLAG --guard $guard --run_index $RUN_INDEX --evaluation $EVALUATION --max_query $MAX_QUERY --max_jailbreak $MAX_JAILBREAK ${EOS_NUM:+ --eos_num $EOS_NUM} --harmful_dataset $HARMFUL_DATASET --targets_dataset $TARGETS_DATASET"
      echo "----------------------------------------"
    } >> "${LOG_PATH}/${index}.log"

    CUDA_VISIBLE_DEVICES="$FREE_GPU" python -u "$PYTHON_SCRIPT" \
      --index "$index" \
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
      >> "${LOG_PATH}/${index}.log" 2>&1

    echo "Task $index on GPU $FREE_GPU finished." >> "${LOG_PATH}/${index}.log"
  ) &

  sleep 30
done

wait
