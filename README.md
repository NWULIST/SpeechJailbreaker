# SpeechJailbreaker

SpeechJailbreaker is a research framework for evaluating how vulnerable speech/text-capable LLMs are to jailbreak attacks, and how well different defenses mitigate those attacks.

The repository provides:
- Multiple attack implementations (optimization, fuzzing, and prompt-based)
- A unified runner interface
- Optional defense wrappers/prompts
- Evaluation with `strongreject` or default scoring

## Features

- Unified command interface through `Scripts/run_interface.py`
- Configurable attack, model, defense, evaluation, and task count
- Support for common jailbreak attack families:
  - `pgd`
  - `fuzzer`
  - `ica`
  - `sure`
  - `reasoning`
  - `jbc`
  - `boost_fuzzer`
  - `pair`
  - `tap`
  - `autoattack`
- Defense options including:
  - `None`
  - `guard`
  - `adashield`
  - `self-reminder`
  - `icd`
  - `smoothllm`
  - `spirit_bias` / `spirit_prune` / `spirit_patch` (or short names `bias`, `prune`, `patch`)

## Project Layout

```text
SpeechJailbreaker/
├── BOOST/                  # Core attack implementations and utilities
├── Dataset/                # Prompt/target datasets
├── Defenses/               # Defense wrappers and implementations
├── Defense_prompt/         # Prompt-based defense templates
├── Experiments/            # Experiment scripts
├── Scripts/                # Shell entrypoints for each attack
├── speechjailbreaker/      # Python package modules
├── strongreject/           # StrongReject evaluation-related code
└── README.md
```

## Installation

### Option A) Install from PyPI

```bash
pip install speechjailbreaker
```

PyPI package: [speechjailbreaker](https://pypi.org/project/speechjailbreaker/)

After installing from PyPI, you can use the Python API directly:

```python
from speechjailbreaker import run_attack, AttackConfig

config = AttackConfig(
    attack="ica",
    model_path="Qwen/Qwen2-Audio-7B-Instruct",
    defence="smoothllm",
    evaluation="strongreject",
    num_tasks=2,
)
exit_code = run_attack(config)
print("Exit code:", exit_code)
```

### Option B) Install from source (recommended for development)

### 1) Clone

```bash
git clone https://github.com/NWULIST/SpeechJailbreaker.git
cd SpeechJailbreaker
```

### 2) Create environment

```bash
conda env create -f environment.yml
conda activate xllm_env
```

If `vllm` is not included by your environment file, install it manually:

```bash
pip install vllm
```

### 3) Install package in editable mode

```bash
pip install -e .
```

## Quick Start

List supported attacks/defenses:

```bash
python Scripts/run_interface.py --list-attacks
python Scripts/run_interface.py --list-defenses
```

Run an attack:

```bash
python Scripts/run_interface.py \
  --attack ica \
  --model_path Qwen/Qwen2-Audio-7B-Instruct \
  --defence smoothllm \
  --evaluation strongreject \
  --num_tasks 2
```

### Common arguments

- `--attack`: attack method name (required unless listing)
- `--model_path`: HuggingFace/local model path
- `--defence`: defense method (default: `None`)
- `--evaluation`: `default` or `strongreject`
- `--num_tasks`: number of tasks to run
- `--batch_size`: batch size per run
- `--guard`: guard model path (if needed by defense)
- `--seed`: random seed for reproducibility
- `--few_shot_num`: few-shot examples (`ica` only)

## Running Specific Scripts

You can call attack-specific shell runners directly, for example:

```bash
bash Scripts/run_GCG.sh
```

Other attack scripts follow the same pattern in `Scripts/` (for example `run_ICA.sh`, `run_SURE.sh`, `run_PAIR.sh`, etc.).

## Datasets

Example datasets shipped with the repository include:
- `Dataset/harmful.csv`
- `Dataset/harmful_targets.csv`
- `Dataset/Advbench.csv`

## Results and Logs

- Intermediate and final outputs are typically written under `Logs/` and/or `Results/`, depending on the script.
- For reproducibility, store the command, model version, defense setting, and random seed for each run.

## License

This project is licensed under the MIT License. See `LICENSE`.

## Disclaimer

This project is intended strictly for safety research and red-teaming evaluation. Do not use it for malicious activity or to attack production systems.
