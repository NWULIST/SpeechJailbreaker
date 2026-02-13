#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
import os
os.environ["OPENAI_API_KEY"] = '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'

ATTACK_TO_SCRIPT = {
    'gcg': 'run_GCG.sh',
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
    parser.add_argument('--defence', required=False, default=None, help='Defence method to run')
    parser.add_argument('--model_path', required=False, default='Qwen/Qwen2-Audio-7B-Instruct', help='Model path to pass to the attack script')
    parser.add_argument('--evaluation', required=False, default='strongreject', help='Evaluation method to pass to the attack script (default or strongreject)')
    parser.add_argument('--num_tasks', type=int, default=2, help='Number of tasks to run in parallel (default: 3)')
    parser.add_argument('--guard', required=False, default=None, help='Guard model to run')
    parser.add_argument('--few_shot_num',type=int, required=False, default=0, help='Number of example prompts to show to model')
    args = parser.parse_args()

    if args.few_shot_num > 0 and args.attack != "ica":
        print("--few_shot_num will be ignored fornon-ICA model")
        args.few_shot_num = 0

    script_name = ATTACK_TO_SCRIPT[args.attack]
    script_path = os.path.join(os.path.dirname(__file__), script_name)

    # # Prepare environment variables
    # env = os.environ.copy()
    # env['MODEL_PATH'] = args.model_path
    # env['EVALUATION'] = args.evaluation

    subprocess_input = [script_path, "--model_path", args.model_path, "--evaluation", args.evaluation, "--num_tasks", str(args.num_tasks), "--defence", str(args.defence), '--guard', str(args.guard)]

    #if few shots is enabled -> addd it to input to be passed on
    if args.attack == 'ica' and args.few_shot_num > 0:
        subprocess_input += ["--few_shot_num", str(args.few_shot_num)]


    print(f"[INFO] Running {script_name} under Defence {args.defence} with MODEL_PATH={args.model_path} and EVALUATION={args.evaluation}")
    try:
        result = subprocess.run(subprocess_input, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {script_name} failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == '__main__':
    main() 
