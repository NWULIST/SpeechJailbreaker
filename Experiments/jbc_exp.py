import sys
import os
from dotenv import load_dotenv
load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')

sys.path.append(os.path.abspath('../BOOST'))
import argparse
import random
import numpy as np
import torch
from fastchat.model import add_model_args


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
    parser = argparse.ArgumentParser(description='JBC Attack - Batched Version')
    parser.add_argument('--defence', type=str, default='', help='defence file location')
    parser.add_argument('--guard', type=str, default=None, help='guard model')
    parser.add_argument('--start_index', type=int, default=None, help='Start index of batch (deprecated, use --indices)')
    parser.add_argument('--end_index', type=int, default=None, help='End index of batch (deprecated, use --indices)')
    parser.add_argument('--indices', type=str, default=None, help='Comma-separated list of indices to process')
    parser.add_argument('--model_path', type=str, default='gpt-3.5-turbo-0125',
                        help='mutate model path')
    parser.add_argument('--target_model', type=str, default='google/gemma-7b-it',
                        help='The target model')
    parser.add_argument('--few_shot_num', type=int, default=1, help='The number of few shot examples')
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--early_stop", action="store_true", help="early stop when the attack is successful")
    parser.add_argument("--eos_num", type=int, default=20, help="max number of eos tokens")
    parser.add_argument("--seed", type=int, default=42, help='Random seed for reproducibility')  
    parser.add_argument('--harmful_dataset', type=str, default='Dataset/harmful.csv',
                        help='Path to the harmful questions dataset')
    parser.add_argument('--targets_dataset', type=str, default='Dataset/harmful_targets.csv',
                        help='Path to the harmful targets dataset')
    parser.add_argument('--evaluation', type=str, default='default', 
                        choices=['default', 'strongreject'], 
                        help='Evaluation method')
    add_model_args(parser)

    args = parser.parse_args()
    set_random_seed(args.seed)
    args.openai_key = openai_key
    
    # Parse indices
    if args.indices:
        args.indices_list = [int(i) for i in args.indices.split(',')]
    elif args.start_index is not None and args.end_index is not None:
        # Backward compatibility with range-based approach
        args.indices_list = list(range(args.start_index, args.end_index + 1))
    else:
        raise ValueError("Must provide either --indices or both --start_index and --end_index")
    
    # Import here to avoid loading heavy imports until needed
    from BOOST.Attack_JBC.jbc import JBC_attack
    
    # Process batch of indices
    JBC_attack(args)