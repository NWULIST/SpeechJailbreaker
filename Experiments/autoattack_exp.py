"""Experiment runner for the audio AutoAttack pipeline."""

# pylint: disable=import-error,wrong-import-position

import argparse
import os
import random
import sys

import numpy as np
import torch
from dotenv import load_dotenv

sys.path.append(os.path.abspath('../BOOST'))

from BOOST.Attack_AutoAttack import audio_autoattack as run_audio_autoattack

load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')


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


def autoattack_audio_attack(attack_args):
    """Delegate to the reusable audio_autoattack entry-point."""
    base_dir = getattr(attack_args, 'base_dir', '/projects/e33046/AudioJailbreak')
    return run_audio_autoattack(attack_args, base_dir=base_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio AutoAttack for Speech LLM')

    # Dataset parameters
    parser.add_argument('--index', type=int, default=0,
                        help='The index of the question in AudioJailbreak dataset')
    parser.add_argument('--base_dir', type=str, default='/projects/e33046/AudioJailbreak',
                        help='Base directory for AudioJailbreak audio files')

    # Model parameters
    parser.add_argument('--target_model', type=str, required=True,
                        help='Target Speech LLM model path')
    parser.add_argument('--model_path', type=str, default='gpt-3.5-turbo',
                        help='Evaluator model path (for default evaluation)')

    # Attack parameters
    parser.add_argument('--eps', type=float, default=0.02,
                        help='Epsilon (perturbation budget) for audio attack')
    parser.add_argument('--n_iter', type=int, default=50,
                        help='Number of attack iterations')
    parser.add_argument('--attack_method', type=str, default='pgd',
                        choices=['pgd', 'genetic'],
                        help='Attack method to use')

    # Genetic algorithm parameters (for genetic attack)
    parser.add_argument('--population_size', type=int, default=10,
                        help='Population size for genetic algorithm')
    parser.add_argument('--mutation_rate', type=float, default=0.3,
                        help='Mutation rate for genetic algorithm')

    # Device parameters
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Device to run on')

    # Evaluation and logging
    parser.add_argument('--evaluation', type=str, default='default',
                        choices=['default', 'strongreject'],
                        help='Evaluation method for attack success')
    parser.add_argument('--run_index', type=int, default=0,
                        help='Run index for logging and results organization')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--save_results', action='store_true', default=True,
                        help='Save attack results to file')

    args = parser.parse_args()

    # Set random seed
    set_random_seed(args.seed)

    print("=" * 80)
    print("Audio AutoAttack Experiment Configuration")
    print("=" * 80)
    print(f"Index: {args.index}")
    print(f"Target Model: {args.target_model}")
    print(f"Epsilon: {args.eps}")
    print(f"Iterations: {args.n_iter}")
    print(f"Attack Method: {args.attack_method}")
    print(f"Device: {args.device}")
    print(f"Evaluation: {args.evaluation}")
    print(f"Seed: {args.seed}")
    print("=" * 80)
    print()

    # Run attack
    args.openai_key = openai_key
    success = autoattack_audio_attack(args)

    # Exit with appropriate code
    sys.exit(0 if success else 1)
