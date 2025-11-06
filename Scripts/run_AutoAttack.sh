#!/usr/bin/env bash

# run_AutoAttack.sh - Run Audio AutoAttack on Speech LLM

# Default parameters
TARGET_MODEL=""  # Required
BASE_DIR="/projects/e33046/AudioJailbreak"
EPS=0.02
N_ITER=50
ATTACK_METHOD="pgd"
POPULATION_SIZE=10
MUTATION_RATE=0.3
DEVICE="auto"
EVALUATION="default"
MODEL_PATH="gpt-3.5-turbo"
RUN_INDEX=0
START_INDEX=0
END_INDEX=10
SEED=42
LOG_DIR="Logs"
RESULTS_DIR="Results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --target_model)
            TARGET_MODEL="$2"
            shift 2
            ;;
        --base_dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --eps)
            EPS="$2"
            shift 2
            ;;
        --n_iter)
            N_ITER="$2"
            shift 2
            ;;
        --attack_method)
            ATTACK_METHOD="$2"
            shift 2
            ;;
        --population_size)
            POPULATION_SIZE="$2"
            shift 2
            ;;
        --mutation_rate)
            MUTATION_RATE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --evaluation)
            EVALUATION="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --run_index)
            RUN_INDEX="$2"
            shift 2
            ;;
        --start_index)
            START_INDEX="$2"
            shift 2
            ;;
        --end_index)
            END_INDEX="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --target_model MODEL [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --target_model MODEL        Target Speech LLM model path"
            echo ""
            echo "Optional:"
            echo "  --base_dir PATH            Base directory for AudioJailbreak (default: /projects/e33046/AudioJailbreak)"
            echo "  --eps FLOAT                Perturbation budget (default: 0.02)"
            echo "  --n_iter INT               Number of iterations (default: 50)"
            echo "  --attack_method METHOD     Attack method: pgd|genetic (default: pgd)"
            echo "  --population_size INT      Population size for genetic (default: 10)"
            echo "  --mutation_rate FLOAT      Mutation rate for genetic (default: 0.3)"
            echo "  --device DEVICE            Device: auto|cuda|cpu (default: auto)"
            echo "  --evaluation METHOD        Evaluation: default|strongreject (default: default)"
            echo "  --model_path MODEL         Evaluator model (default: gpt-3.5-turbo)"
            echo "  --run_index INT            Run index (default: 0)"
            echo "  --start_index INT          Start index (default: 0)"
            echo "  --end_index INT            End index (default: 10)"
            echo "  --seed INT                 Random seed (default: 42)"
            exit 1
            ;;
    esac
done

# Check required parameters
if [ -z "$TARGET_MODEL" ]; then
    echo "Error: --target_model is required"
    echo "Example: $0 --target_model Qwen/Qwen2-Audio-7B-Instruct"
    exit 1
fi

# Create log and results directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/autoattack_audio_${TIMESTAMP}.log"

echo "============================================================" | tee -a "$LOG_FILE"
echo "Audio AutoAttack Runner for Speech LLM" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Target Model: $TARGET_MODEL" | tee -a "$LOG_FILE"
echo "Base Directory: $BASE_DIR" | tee -a "$LOG_FILE"
echo "Epsilon: $EPS" | tee -a "$LOG_FILE"
echo "Iterations: $N_ITER" | tee -a "$LOG_FILE"
echo "Attack Method: $ATTACK_METHOD" | tee -a "$LOG_FILE"
if [ "$ATTACK_METHOD" = "genetic" ]; then
    echo "Population Size: $POPULATION_SIZE" | tee -a "$LOG_FILE"
    echo "Mutation Rate: $MUTATION_RATE" | tee -a "$LOG_FILE"
fi
echo "Device: $DEVICE" | tee -a "$LOG_FILE"
echo "Evaluation: $EVALUATION" | tee -a "$LOG_FILE"
echo "Evaluator Model: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "Run Index: $RUN_INDEX" | tee -a "$LOG_FILE"
echo "Index Range: $START_INDEX - $END_INDEX" | tee -a "$LOG_FILE"
echo "Seed: $SEED" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run attacks
SUCCESS_COUNT=0
FAIL_COUNT=0

for ((INDEX=$START_INDEX; INDEX<$END_INDEX; INDEX++)); do
    echo "Running Audio AutoAttack on index $INDEX..." | tee -a "$LOG_FILE"
    
    python Experiments/autoattack_exp.py \
        --index "$INDEX" \
        --target_model "$TARGET_MODEL" \
        --base_dir "$BASE_DIR" \
        --eps "$EPS" \
        --n_iter "$N_ITER" \
        --attack_method "$ATTACK_METHOD" \
        --population_size "$POPULATION_SIZE" \
        --mutation_rate "$MUTATION_RATE" \
        --device "$DEVICE" \
        --evaluation "$EVALUATION" \
        --model_path "$MODEL_PATH" \
        --run_index "$RUN_INDEX" \
        --seed "$SEED" \
        --save_results \
        2>&1 | tee -a "$LOG_FILE"
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "  Success for index $INDEX" | tee -a "$LOG_FILE"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "  Failed for index $INDEX" | tee -a "$LOG_FILE"
    fi
    
    echo "" | tee -a "$LOG_FILE"
done

# Summary
echo "============================================================" | tee -a "$LOG_FILE"
echo "Audio AutoAttack Summary" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Total Attacks: $((END_INDEX - START_INDEX))" | tee -a "$LOG_FILE"
echo "Successful: $SUCCESS_COUNT" | tee -a "$LOG_FILE"
echo "Failed: $FAIL_COUNT" | tee -a "$LOG_FILE"

if [ $((END_INDEX - START_INDEX)) -gt 0 ]; then
    SUCCESS_RATE=$(awk "BEGIN {printf \"%.2f\", $SUCCESS_COUNT / ($END_INDEX - $START_INDEX) * 100}")
    echo "Success Rate: ${SUCCESS_RATE}%" | tee -a "$LOG_FILE"
fi

echo "============================================================" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"

MODEL_NAME=$(basename "$TARGET_MODEL")
echo "Results saved to: $RESULTS_DIR/${MODEL_NAME}/AutoAttack/run_$RUN_INDEX/" | tee -a "$LOG_FILE"
