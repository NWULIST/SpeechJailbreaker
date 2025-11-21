import sys
import os
from dotenv import load_dotenv
load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')

# Add the path to the BOOST folder to sys.path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # Get the project root (parent of Experiments directory)
# project_root = os.path.dirname(current_dir)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
import argparse
import random
import numpy as np
import torch
from BOOST.Attack_Autoattack.autoattack_wrapper import autoattack_attack
from BOOST.utils.constants import claude_key, gemini_key


def set_random_seed(seed=42):
    """Set random seed for reproducibility across different libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AutoAttack parameters')

    # Basic parameters
    parser.add_augument('--defence', type=str, default='', help='defence file location')
    parser.add_argument('--index', type=int, default=0,
                        help='The index of the question')
    parser.add_argument('--model_path', type=str, default='gpt-4-1106-preview',
                        help='Evaluator model path')
    parser.add_argument('--target_model', type=str, default='google/gemma-7b-it',
                        help='The target model, openai model or open-sourced LLMs')

    # Attack parameters
    parser.add_argument('--norm', type=str, default='Linf',
                        choices=['Linf', 'L2', 'L1'],
                        help='Norm for adversarial perturbation')
    parser.add_argument('--eps', type=float, default=0.3,
                        help='Epsilon (perturbation budget)')
    parser.add_argument('--version', type=str, default='standard',
                        choices=['standard', 'plus', 'rand', 'custom'],
                        help='AutoAttack version')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda or cpu)')

    # Query and jailbreak limits
    parser.add_argument('--max_query', type=int, default=100,
                        help='The maximum number of queries')
    parser.add_argument('--max_jailbreak', type=int, default=1,
                        help='The maximum jailbreak number')

    # Experiment parameters
    parser.add_argument("--run_index", type=int, default=0,
                        help="Run index for organizing results")
    parser.add_argument("--seed", type=int, default=42,
                        help='Random seed for reproducibility')

    # Dataset parameters
    parser.add_argument('--harmful_dataset', type=str, default='Dataset/harmful.csv',
                        help='Path to the harmful questions dataset')
    parser.add_argument('--targets_dataset', type=str, default='Dataset/harmful_targets.csv',
                        help='Path to the harmful targets dataset')

    # Evaluation method
    parser.add_argument('--evaluation', type=str, default='default',
                        choices=['default', 'strongreject'],
                        help='Evaluation method for attack success: "default" or "strongreject"')
    # Base directory for audio files
    parser.add_argument('--base_dir', type=str, default='/projects/e33046/AudioJailbreak',
                        help='Base directory for audio files')
    args = parser.parse_args()

    # Set random seed
    set_random_seed(args.seed)

    # Set API keys
    args.openai_key = openai_key
    args.claude_key = claude_key
    args.gemini_key = gemini_key

    print("=" * 60)
    print("AutoAttack Experiment Configuration")
    print("=" * 60)
    print(f"Index: {args.index}")
    print(f"Target Model: {args.target_model}")
    print(f"Evaluator Model: {args.model_path}")
    print(f"Norm: {args.norm}")
    print(f"Epsilon: {args.eps}")
    print(f"Version: {args.version}")
    print(f"Max Queries: {args.max_query}")
    print(f"Max Jailbreak: {args.max_jailbreak}")
    print(f"Evaluation Method: {args.evaluation}")
    print(f"Run Index: {args.run_index}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    print()

    # Run attack
    autoattack_attack(args)
