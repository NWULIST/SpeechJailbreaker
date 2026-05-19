"""
AttackBench API: programmatic interface for running attacks.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

from speechjailbreaker.config import (
    ATTACK_TO_SCRIPT,
    AttackConfig,
    SPIRIT_SHORT_TO_FULL,
)


def _resolve_project_root(scripts_dir: Optional[str] = None) -> Path:
    """Resolve project root (parent of Scripts/)."""
    if scripts_dir:
        p = Path(scripts_dir).resolve()
        if p.name == "Scripts":
            return p.parent
        return p
    # Assume we're in project root or Scripts/
    cwd = Path.cwd()
    if (cwd / "Scripts").exists():
        return cwd
    if (cwd / "speechjailbreaker").exists():
        return cwd
    # Fallback: parent of speechjailbreaker package
    mod = __file__
    pkg_dir = Path(mod).resolve().parent
    # speechjailbreaker/ is inside project root
    return pkg_dir.parent


def run_attack(config: AttackConfig) -> int:
    """
    Run an attack experiment.

    Args:
        config: AttackConfig with attack method, model, defense, etc.

    Returns:
        Exit code from the underlying script (0 = success).
    """
    root = _resolve_project_root(config.scripts_dir)
    scripts_dir = root / "Scripts"
    script_name = ATTACK_TO_SCRIPT.get(config.attack)
    if not script_name:
        raise ValueError(
            f"Unknown attack '{config.attack}'. "
            f"Supported: {list(ATTACK_TO_SCRIPT.keys())}"
        )
    script_path = scripts_dir / script_name
    if not script_path.exists():
        raise FileNotFoundError(
            f"Script not found: {script_path}. "
            "Ensure you run from the AttackBench project root."
        )

    defence = config.defence
    if defence in SPIRIT_SHORT_TO_FULL:
        defence = SPIRIT_SHORT_TO_FULL[defence]

    cmd = [
        str(script_path),
        "--model_path",
        config.model_path,
        "--evaluation",
        config.evaluation,
        "--num_tasks",
        str(config.num_tasks),
        "--batch_size",
        str(config.batch_size),
        "--defence",
        str(defence),
        "--guard",
        str(config.guard or ""),
    ]
    if config.attack == "ica" and config.few_shot_num > 0:
        cmd += ["--few_shot_num", str(config.few_shot_num)]
    if config.seed is not None:
        cmd += ["--seed", str(config.seed)]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + (os.environ.get("PYTHONPATH", "") and f":{os.environ['PYTHONPATH']}" or "")

    result = subprocess.run(cmd, cwd=str(root), env=env)
    return result.returncode


def run_attack_cli(
    attack: str,
    model_path: str = "Qwen/Qwen2-Audio-7B-Instruct",
    defence: str = "None",
    evaluation: str = "strongreject",
    num_tasks: int = 2,
    batch_size: int = 1,
    guard: Optional[str] = None,
    seed: Optional[int] = None,
    few_shot_num: int = 0,
) -> int:
    """
    CLI-friendly wrapper for run_attack.

    Args:
        attack: Attack method (e.g. 'ica', 'jbc', 'tap')
        model_path: Target model path
        defence: Defense method
        evaluation: 'default' or 'strongreject'
        num_tasks: Number of tasks
        batch_size: Batch size
        guard: Guard model path
        seed: Random seed
        few_shot_num: Few-shot examples (ICA only)

    Returns:
        Exit code.
    """
    config = AttackConfig(
        attack=attack,
        model_path=model_path,
        defence=defence,
        evaluation=evaluation,
        num_tasks=num_tasks,
        batch_size=batch_size,
        guard=guard,
        seed=seed,
        few_shot_num=few_shot_num,
    )
    return run_attack(config)
