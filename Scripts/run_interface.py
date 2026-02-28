#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
import os

from dotenv import load_dotenv
load_dotenv()

# Map short SPIRIT defence names to the prefixed names expected by attack scripts
SPIRIT_DEFENCE_MAP = {
    'bias':  'spirit_bias',
    'prune': 'spirit_prune',
    'patch': 'spirit_patch',
}

ATTACK_TO_SCRIPT = {
    'pgd': 'run_PGD.sh',
    'fuzzer': 'run_GPTFuzzer.sh',
    'ica': 'run_ICA.sh',
    'sure': 'run_SURE.sh',
    'reasoning': 'run_reasoning.sh',
    'jbc': 'run_JBC.sh',
    'boost_fuzzer': 'run_boost_GPTFuzzer.sh',
    'pair': 'run_PAIR.sh',
    'tap': 'run_TAP.sh',
    'autoattack': 'run_auto.sh',
}

def main():
    parser = argparse.ArgumentParser(description='Unified Python interface for running attack scripts.')
    parser.add_argument('--attack', required=True, default='tap', choices=ATTACK_TO_SCRIPT.keys(), help='Attack method to run')
    parser.add_argument('--defence', required=False, default='None', help='Defence method to run')
    parser.add_argument('--model_path', required=False, default='Qwen/Qwen2-Audio-7B-Instruct', help='Model path to pass to the attack script')
    parser.add_argument('--evaluation', required=False, default='strongreject', help='Evaluation method to pass to the attack script (default or strongreject)')
    parser.add_argument('--num_tasks', type=int, default=2, help='Number of tasks to run in parallel (default: 3)')
    parser.add_argument('--batch_size', type=int, default=1, help='number of prompts to run per batch')
    parser.add_argument('--guard', required=False, default=None, help='Guard model to run')
    parser.add_argument('--seed', type=int, required=False, default=None, help='seed input for reproducibility in batched runs')
    parser.add_argument('--few_shot_num',type=int, required=False, default=0, help='Number of example prompts to show to model')
    args = parser.parse_args()

    if args.few_shot_num > 0 and args.attack != "ica":
        print("--few_shot_num will be ignored fornon-ICA model")
        args.few_shot_num = 0

    # Resolve short SPIRIT defence names
    defence = args.defence
    if defence in SPIRIT_DEFENCE_MAP:
        defence = SPIRIT_DEFENCE_MAP[defence]

    script_name = ATTACK_TO_SCRIPT[args.attack]
    script_path = os.path.join(os.path.dirname(__file__), script_name)

    subprocess_input = [script_path, "--model_path", args.model_path, "--evaluation", args.evaluation, "--num_tasks", str(args.num_tasks), "--batch_size", str(args.batch_size), "--defence", str(defence), '--guard', str(args.guard)]

    #if few shots is enabled -> addd it to input to be passed on
    if args.attack == 'ica' and args.few_shot_num > 0:
        subprocess_input += ["--few_shot_num", str(args.few_shot_num)]
    
    if args.seed is not None:
        subprocess_input += ['--seed', str(args.seed)]


    print(f"[INFO] Running {script_name} under Defence {args.defence} with MODEL_PATH={args.model_path} and EVALUATION={args.evaluation}")
    try:
        result = subprocess.run(subprocess_input, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {script_name} failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == '__main__':
    main()