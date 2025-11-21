import argparse
from pathlib import Path
import sys
import random
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from autoattack import TapLikeArgs, run_autoattack_apgd_ce  

# ---------------- Utils ----------------
def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def map_device(dev: str) -> str:
    if dev.lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return dev

# --------------- Main ------------------
def build_argparser():
    p = argparse.ArgumentParser("AutoAttack (APGD-CE only) — TAP-compatible (no search)")

    p.add_argument("--dataset", type=str, default="MBZUAI/AudioJailbreak")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--run_index", type=int, default=0)
    p.add_argument("--base_dir", type=str, default="/projects/e33046/AudioJailbreak")
    p.add_argument("--result_dir", type=str, default="Results")


    p.add_argument("--target_model", type=str, required=True)
    p.add_argument("--evaluation", type=str, choices=["strongreject", "judge"], default="strongreject")
    p.add_argument("--early_stop", action="store_true")


    p.add_argument("--goal", type=str, default="")
    p.add_argument("--target_str", type=str, default="Sure, here is a tutorial for")
    p.add_argument("--seed_path", type=str, default=None)

    # APGD-CE 
    p.add_argument("--hf_model", "--hf-model", dest="hf_model", type=str, required=True,
                   help="HuggingFace classifier (path or hub id) used for APGD-CE on inputs_embeds")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--attack_norm", type=str, default="Linf", choices=["Linf", "L2"])
    p.add_argument("--eps", type=float, default=0.125)
    p.add_argument("--n_iter", type=int, default=100, help="APGD iterations")
    p.add_argument("--n_restarts", type=int, default=5)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--target_label", type=int, default=1, help="index of the 'unsafe' class in the HF classifier")


    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--store_folder", type=str, default="")
    p.add_argument("--iter_index", type=int, default=0)
    return p

def main():
    args = build_argparser().parse_args()
    set_random_seed(args.seed)

    print("=" * 80)
    print("AutoAttack (APGD-CE) Experiment Configuration")
    print("=" * 80)
    print(f"Dataset:       {args.dataset}")
    print(f"Index:         {args.index}")
    print(f"Run Index:     {args.run_index}")
    print(f"Base Dir:      {args.base_dir}")
    print(f"Result Dir:    {args.result_dir}")
    print(f"Target Model:  {args.target_model}")
    print(f"Evaluation:    {args.evaluation}")
    print(f"Device:        {args.device} -> {map_device(args.device)}")
    print(f"Eps:           {args.eps}")
    print(f"APGD iters:    {args.n_iter}")
    print(f"Restarts:      {args.n_restarts}")
    print(f"HF model:      {args.hf_model}")
    print(f"Max length:    {args.max_length}")
    print(f"Target label:  {args.target_label}")
    print(f"Seed:          {args.seed}")
    print("=" * 80)
    print()

    targs = TapLikeArgs(
        dataset=args.dataset,
        index=args.index,
        run_index=args.run_index,
        base_dir=args.base_dir,
        result_dir=args.result_dir,
        target_model=args.target_model,
        evaluation=args.evaluation,
        early_stop=args.early_stop,
        goal=args.goal,
        target_str=args.target_str,
        seed_path=args.seed_path,
        # APGD-CE
        hf_model=args.hf_model,
        device=map_device(args.device),
        attack_norm=args.attack_norm,
        eps=args.eps,
        apgd_iters=args.n_iter,
        n_restarts=args.n_restarts,
        max_length=args.max_length,
        target_label=args.target_label,
        # misc
        seed=args.seed,
        store_folder=args.store_folder,
        iter_index=args.iter_index,
    )

    out_csv = run_autoattack_apgd_ce(targs)
    print(f"[AutoAttack/APGD-CE] Wrote: {out_csv}")

if __name__ == "__main__":
    main()
