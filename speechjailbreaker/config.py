"""
AttackBench configuration: models, attacks, defenses, and test scenarios.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union


class AttackMethod(str, Enum):
    """Supported attack methods."""

    PGD = "pgd"
    FUZZER = "fuzzer"
    ICA = "ica"
    SURE = "sure"
    REASONING = "reasoning"
    JBC = "jbc"
    BOOST_FUZZER = "boost_fuzzer"
    PAIR = "pair"
    TAP = "tap"
    AUTOATTACK = "autoattack"


class DefenseMethod(str, Enum):
    """Supported defense methods (JSON prompts + special wrappers)."""

    NONE = "None"
    GUARD = "guard"
    # JSON-based defenses (Defense_prompt/*.json)
    ADASHIELD = "adashield"
    SELF_REMINDER = "self-reminder"
    ICD = "icd"
    # Wrapper-based defenses (no JSON)
    SMOOTHLLM = "smoothllm"
    SPIRIT_BIAS = "spirit_bias"
    SPIRIT_PRUNE = "spirit_prune"
    SPIRIT_PATCH = "spirit_patch"


class EvaluationMethod(str, Enum):
    """Evaluation methods for attack success."""

    DEFAULT = "default"
    STRONGREJECT = "strongreject"


# Short names for SPIRIT (user-friendly)
SPIRIT_SHORT_TO_FULL = {
    "bias": "spirit_bias",
    "prune": "spirit_prune",
    "patch": "spirit_patch",
}


@dataclass
class AttackConfig:
    """Configuration for running an attack experiment."""

    attack: Union[AttackMethod, str]
    model_path: str = "Qwen/Qwen2-Audio-7B-Instruct"
    defence: Union[str, DefenseMethod] = DefenseMethod.NONE
    evaluation: Union[EvaluationMethod, str] = EvaluationMethod.STRONGREJECT
    num_tasks: int = 2
    batch_size: int = 1
    guard: Optional[str] = None
    seed: Optional[int] = None
    few_shot_num: int = 0
    base_dir: Optional[str] = None
    scripts_dir: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.attack, AttackMethod):
            self.attack = self.attack.value
        if isinstance(self.defence, DefenseMethod):
            self.defence = self.defence.value
        if isinstance(self.evaluation, EvaluationMethod):
            self.evaluation = self.evaluation.value
        # Resolve SPIRIT short names
        if self.defence in SPIRIT_SHORT_TO_FULL:
            self.defence = SPIRIT_SHORT_TO_FULL[self.defence]


# Mapping from attack name to shell script
ATTACK_TO_SCRIPT = {
    "pgd": "run_PGD.sh",
    "fuzzer": "run_GPTFuzzer.sh",
    "ica": "run_ICA.sh",
    "sure": "run_SURE.sh",
    "reasoning": "run_reasoning.sh",
    "jbc": "run_JBC.sh",
    "boost_fuzzer": "run_boost_GPTFuzzer.sh",
    "pair": "run_PAIR.sh",
    "tap": "run_TAP.sh",
    "autoattack": "run_auto.sh",
}


def list_attacks() -> List[str]:
    """Return list of supported attack method names."""
    return list(ATTACK_TO_SCRIPT.keys())


def list_defenses() -> List[str]:
    """Return list of supported defense names (including short SPIRIT names)."""
    return [
        "None",
        "guard",
        "adashield",
        "self-reminder",
        "icd",
        "smoothllm",
        "spirit_bias",
        "spirit_prune",
        "spirit_patch",
        "bias",
        "prune",
        "patch",
    ]
