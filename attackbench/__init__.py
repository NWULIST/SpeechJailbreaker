"""
AttackBench: Framework for testing LLM robustness against jailbreak attacks.

Provides unified API for running attacks with different models, tasks, and defenses.
"""

from attackbench.config import (
    AttackConfig,
    AttackMethod,
    DefenseMethod,
    EvaluationMethod,
    list_attacks,
    list_defenses,
)
from attackbench.api import run_attack, run_attack_cli

__version__ = "0.1.0"
__all__ = [
    "run_attack",
    "run_attack_cli",
    "AttackConfig",
    "AttackMethod",
    "DefenseMethod",
    "EvaluationMethod",
    "list_attacks",
    "list_defenses",
]
