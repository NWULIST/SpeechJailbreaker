import sys
import os
from dotenv import load_dotenv
load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')

# Add the path to the BOOST folder to sys.path
sys.path.append(os.path.abspath('../BOOST'))
import argparse
import random
import numpy as np
import torch
from BOOST.Attack_BOOST_GPTFuzzer.gptfuzz import fuzzer_attack
from fastchat.model import add_model_args
from BOOST.utils.constants import claude_key, gemini_key
from BOOST.utils.templates import get_eos


def set_random_seed(seed=42):
    """Set random seed for reproducibility across different libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fuzzing parameters')
    parser.add_argument('--index', type=int, default=None,
                        help='Single index (legacy, use --indices instead)')
    parser.add_argument('--indices', type=str, default=None,
                        help='Comma-separated list of indices to process')
    parser.add_argument('--model_path', type=str, default='gpt-4-1106-preview',
                        help='mutate model path')
    parser.add_argument('--defence', type=str, default='', help='defence file location')
    parser.add_argument('--guard', type=str, default=None, help='guard model')
    parser.add_argument('--target_model', type=str, default='google/gemma-7b-it',
                        help='The target model, openai model or open-sourced LLMs')
    parser.add_argument('--max_query', type=int, default=100,
                        help='The maximum number of queries')
    parser.add_argument('--max_jailbreak', type=int,
                        default=1, help='The maximum jailbreak number')
    parser.add_argument('--energy', type=int, default=1,
                        help='The energy of the fuzzing process')
    parser.add_argument('--seed_selection_strategy', type=str,
                        default='round_robin', help='The seed selection strategy')
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed_path", type=str,
                        default="./Dataset/fuzzer_seed.csv", help="The seed path")
    parser.add_argument("--add_eos", action='store_true')
    parser.add_argument("--eos_num", type=int, default=10, help="The number of eos tokens")
    parser.add_argument("--run_index", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--harmful_dataset', type=str, default='Dataset/harmful.csv',
                        help='Path to the harmful questions dataset')
    parser.add_argument('--targets_dataset', type=str, default='Dataset/harmful_targets.csv',
                        help='Path to the harmful targets dataset')
    parser.add_argument('--evaluation', type=str, default='default', choices=['default', 'strongreject'], help='Evaluation method for attack success')
    add_model_args(parser)

    args = parser.parse_args()
    set_random_seed(args.seed)
    args.openai_key = openai_key
    args.claude_key = claude_key
    args.gemini_key = gemini_key

    # Parse indices — support both --indices (batch) and --index (legacy)
    if args.indices:
        indices_list = [int(i) for i in args.indices.split(',')]
    elif args.index is not None:
        indices_list = [args.index]
    else:
        raise ValueError("Must provide either --indices or --index")

    # Loop over indices, calling fuzzer_attack for each
    for idx in indices_list:
        print(f"\n{'='*60}")
        print(f"Processing index {idx}")
        print(f"{'='*60}")
        
        args.index = idx
        try:
            fuzzer_attack(args)
            
            # Check result file for success
            save_path = f'./Results/{args.target_model}/GPTFuzzer-{args.run_index}/{idx}.csv'
            if args.add_eos:
                save_path = f'./Results/{args.target_model}/GPTFuzzer_eos-{args.run_index}/{idx}.csv'
            
            score = 0
            if os.path.isfile(save_path):
                import csv
                with open(save_path, 'r', encoding='utf-8', errors='replace') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    for row in reader:
                        try:
                            val = float(row[-1])
                            if val > 0:
                                score = 1
                                break
                        except (ValueError, IndexError):
                            pass
            
            print(f"\nRESULT:{idx},{score},1", flush=True)
        
        except Exception as e:
            print(f"ERROR processing index {idx}: {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            print(f"RESULT:{idx},0,0", flush=True)