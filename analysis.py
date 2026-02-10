#!/usr/bin/env python3
"""
Complete script to extract and analyze PAIR attack results
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import json
import re


def extract_results_from_log(log_file):
    """Extract RESULT lines from a single log file."""
    results = []
    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                match = re.match(r"^RESULT:(\d+),([0-9.]+),(\d+)", line.strip())
                if match:
                    index = int(match.group(1))
                    score = float(match.group(2))
                    count = int(match.group(3))
                    results.append((index, score, count))
    except Exception as e:
        print(f"Warning: Error reading {log_file}: {e}")
    return results


def extract_all_results(log_dir):
    """Extract results from all log files in a directory."""
    log_dir = Path(log_dir)

    if not log_dir.exists():
        raise FileNotFoundError(f"Directory not found: {log_dir}")

    # Find all log files
    log_files = list(log_dir.glob("*.log")) + list(log_dir.glob("**/*.log"))

    if not log_files:
        raise ValueError(f"No log files found in {log_dir}")

    print(f"Found {len(log_files)} log files")

    # Extract results from all files
    all_results = []
    for log_file in sorted(log_files):
        results = extract_results_from_log(log_file)
        all_results.extend(results)

    if not all_results:
        raise ValueError("No RESULT lines found in any log files")

    # Create DataFrame
    df = pd.DataFrame(all_results, columns=["index", "total_score", "total_count"])
    df["ASR"] = df.apply(
        lambda row: row["total_score"] / row["total_count"]
        if row["total_count"] > 0
        else 0.0,
        axis=1,
    )

    # Remove duplicates and sort
    df = df.drop_duplicates(subset=["index"], keep="first")
    df = df.sort_values("index").reset_index(drop=True)

    return df


def calculate_asr(df):
    """Calculate ASR statistics from DataFrame."""
    if len(df) == 0:
        return {}

    total_score = df["total_score"].sum()
    total_count = df["total_count"].sum()
    total_asr = total_score / total_count if total_count > 0 else 0.0

    samples_with_success = (df["total_score"] > 0).sum()
    sample_success_rate = samples_with_success / len(df)

    stats = {
        "total_asr": float(total_asr),
        "sample_success_rate": float(sample_success_rate),
        "num_samples": int(len(df)),
        "samples_with_success": int(samples_with_success),
        "total_attacks": int(total_count),
        "successful_attacks": float(total_score),
        "mean_asr_per_sample": float(df["ASR"].mean()),
        "median_asr_per_sample": float(df["ASR"].median()),
        "min_asr": float(df["ASR"].min()),
        "max_asr": float(df["ASR"].max()),
        "std_asr": float(df["ASR"].std()),
    }

    return stats


def print_results(stats):
    """Print ASR results in a readable format."""
    print("\n" + "=" * 60)
    print("ATTACK SUCCESS RATE (ASR) RESULTS")
    print("=" * 60)
    print(f"Total ASR:                    {stats['total_asr']:.2%}")
    print(f"Sample Success Rate:          {stats['sample_success_rate']:.2%}")
    print("-" * 60)
    print(f"Total Samples Tested:         {stats['num_samples']}")
    print(f"Samples with ≥1 Success:      {stats['samples_with_success']}")
    print(f"Total Attacks:                {stats['total_attacks']}")
    print(f"Successful Attacks:           {stats['successful_attacks']:.0f}")
    print("=" * 60)
    print(f"Mean ASR per sample:          {stats['mean_asr_per_sample']:.2%}")
    print(f"Median ASR per sample:        {stats['median_asr_per_sample']:.2%}")
    print(f"Min ASR:                      {stats['min_asr']:.2%}")
    print(f"Max ASR:                      {stats['max_asr']:.2%}")
    print(f"Std Dev:                      {stats['std_asr']:.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Extract and analyze PAIR attack results from log files"
    )
    parser.add_argument("--log_dir", type=str, help="Directory containing log files")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="pair_results.csv",
        help="Output CSV file (default: pair_results.csv)",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="pair_summary.json",
        help="Summary JSON file (default: pair_summary.json)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=10,
        help="Number of rows to preview (default: 10)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PAIR Attack Results Analysis")
    print("=" * 60)
    print(f"Log directory: {args.log_dir}\n")

    try:
        # Step 1: Extract results
        print("Step 1: Extracting results from log files...")
        print("-" * 60)
        df = extract_all_results(args.log_dir)

        # Save to CSV
        df.to_csv(args.output, index=False)
        print(f"✓ Extracted {len(df)} results")
        print(f"✓ Saved to: {args.output}\n")

        # Step 2: Calculate ASR
        print("Step 2: Calculating Attack Success Rate...")
        print("-" * 60)
        stats = calculate_asr(df)

        # Print results
        print_results(stats)

        # Save summary
        with open(args.summary, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\n✓ Summary saved to: {args.summary}")

        # Show preview
        if args.preview > 0:
            print(f"\nPreview (first {args.preview} rows):")
            print("=" * 60)
            print(df.head(args.preview).to_string(index=False))
            print("=" * 60)

        print("\n" + "=" * 60)
        print("Analysis Complete!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
