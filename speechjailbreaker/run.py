"""
AttackBench CLI entry point. Invoked via `attackbench` or `python -m attackbench.run`.
"""

import argparse
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from attackbench.config import ATTACK_TO_SCRIPT, list_attacks, list_defenses, AttackConfig
from attackbench.api import run_attack


def main():
    parser = argparse.ArgumentParser(
        description="AttackBench: Unified interface for running jailbreak attack experiments."
    )
    parser.add_argument(
        "--attack",
        required=False,
        choices=list(ATTACK_TO_SCRIPT.keys()),
        help="Attack method to run (required unless --list-attacks/--list-defenses)",
    )
    parser.add_argument(
        "--defence",
        default="None",
        help="Defense method (e.g. None, smoothllm, spirit_bias, adashield)",
    )
    parser.add_argument(
        "--model_path",
        default="Qwen/Qwen2-Audio-7B-Instruct",
        help="Target model path",
    )
    parser.add_argument(
        "--evaluation",
        default="strongreject",
        choices=["default", "strongreject"],
        help="Evaluation method",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=2,
        help="Number of tasks to run",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size per run",
    )
    parser.add_argument(
        "--guard",
        default=None,
        help="Guard model path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--few_shot_num",
        type=int,
        default=0,
        help="Few-shot examples (ICA only)",
    )
    parser.add_argument(
        "--list-attacks",
        action="store_true",
        help="List supported attack methods and exit",
    )
    parser.add_argument(
        "--list-defenses",
        action="store_true",
        help="List supported defense methods and exit",
    )
    args = parser.parse_args()

    if args.list_attacks:
        print("Supported attacks:", ", ".join(list_attacks()))
        sys.exit(0)
    if args.list_defenses:
        print("Supported defenses:", ", ".join(list_defenses()))
        sys.exit(0)

    if not args.attack:
        parser.error("--attack is required (or use --list-attacks / --list-defenses)")

    if args.few_shot_num > 0 and args.attack != "ica":
        print("[INFO] --few_shot_num is ignored for non-ICA attacks")

    config = AttackConfig(
        attack=args.attack,
        model_path=args.model_path,
        defence=args.defence,
        evaluation=args.evaluation,
        num_tasks=args.num_tasks,
        batch_size=args.batch_size,
        guard=args.guard,
        seed=args.seed,
        few_shot_num=args.few_shot_num,
    )
    print(
        f"[INFO] Running {ATTACK_TO_SCRIPT[args.attack]} | "
        f"Defence={args.defence} | Model={args.model_path} | Eval={args.evaluation}"
    )
    exit_code = run_attack(config)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
