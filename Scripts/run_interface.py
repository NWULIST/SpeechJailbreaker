import argparse
import subprocess
import os
import sys

ATTACK_TO_SCRIPT = {
    'gcg': 'run_GCG.sh',
    'gptfuzzer': 'run_GPTFuzzer.sh',
    'ica': 'run_ICA.sh',
    'sure': 'run_SURE.sh',
}

def main():
    parser = argparse.ArgumentParser(description='Unified Python interface for running attack scripts.')
    parser.add_argument('--attack', required=True, choices=ATTACK_TO_SCRIPT.keys(), help='Attack method to run (gcg, gptfuzzer, ica, sure)')
    parser.add_argument('--model_path', default='google/gemma-7b-it', required=False, help='Model path to pass to the attack script')
    parser.add_argument('--evaluation', required=False, default='strongreject', help='Evaluation method to pass to the attack script (default or strongreject)')
    parser.add_argument('--eos_num', required=False, type=int, default=10, help='Number of EOS tokens to use for GCG (only passed if attack is gcg)')
    parser.add_argument('--gpu_memory', required=False, type=int, default=40000, help='Minimum GPU memory required in MiB (default: 40000)')
    parser.add_argument('--num_gpu_search', required=False, type=int, default=7, help='Number of GPUs to search (0-N, default: 7)')
    parser.add_argument('--num_tasks', required=False, type=int, default=2, help='Number of tasks to run in parallel (default: 0)')
    parser.add_argument('--harmful_dataset', required=False, default='Dataset/harmful.csv', help='Path to the harmful questions dataset')
    parser.add_argument('--targets_dataset', required=False, default='Dataset/harmful_targets.csv', help='Path to the harmful targets dataset')
    args = parser.parse_args()

    script_name = ATTACK_TO_SCRIPT[args.attack]
    script_path = os.path.join(os.path.dirname(__file__), script_name)

    # Build command arguments
    cmd = [script_path, "--evaluation", args.evaluation]
    if args.model_path:
        cmd.extend(["--model_path", args.model_path])
    if args.eos_num is not None:
        cmd.extend(["--eos_num", str(args.eos_num)])
    if args.gpu_memory is not None:
        cmd.extend(["--gpu_memory", str(args.gpu_memory)])
    if args.num_gpu_search is not None:
        cmd.extend(["--num_gpu_search", str(args.num_gpu_search)])
    if args.num_tasks is not None:
        cmd.extend(["--num_tasks", str(args.num_tasks)])
    if args.harmful_dataset:
        cmd.extend(["--harmful_dataset", args.harmful_dataset])
    if args.targets_dataset:
        cmd.extend(["--targets_dataset", args.targets_dataset])

    print(f"[INFO] Running {script_name} with EVALUATION={args.evaluation}")
    if args.model_path:
        print(f"[INFO] MODEL_PATH={args.model_path}")
    if args.gpu_memory is not None:
        print(f"[INFO] GPU_MEMORY={args.gpu_memory}")
    if args.num_gpu_search is not None:
        print(f"[INFO] NUM_GPU_SEARCH={args.num_gpu_search}")
    if args.num_tasks is not None:
        print(f"[INFO] NUM_TASKS={args.num_tasks}")
    if args.harmful_dataset:
        print(f"[INFO] HARMFUL_DATASET={args.harmful_dataset}")
    if args.targets_dataset:
        print(f"[INFO] TARGETS_DATASET={args.targets_dataset}")
    
    try:
        result = subprocess.run(cmd, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {script_name} failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == '__main__':
    main() 