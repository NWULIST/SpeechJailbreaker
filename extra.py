#!/usr/bin/env python3
"""
Extract PAIR attack results from log files and create a CSV
"""

import re
import sys
import argparse
from pathlib import Path
import pandas as pd


def extract_results_from_log(log_file):
    """
    Extract RESULT lines from a single log file.
    
    Returns:
        list of tuples: [(index, score, count), ...]
    """
    results = []
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Match: RESULT:index,score,count
                match = re.match(r'^RESULT:(\d+),([0-9.]+),(\d+)', line.strip())
                if match:
                    index = int(match.group(1))
                    score = float(match.group(2))
                    count = int(match.group(3))
                    results.append((index, score, count))
    except Exception as e:
        print(f"Warning: Error reading {log_file}: {e}")
    
    return results


def extract_all_results(log_dir, output_csv):
    """
    Extract results from all log files in a directory.
    
    Args:
        log_dir: Path to directory containing log files
        output_csv: Path to output CSV file
    
    Returns:
        pandas DataFrame with results
    """
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
        print("Warning: No RESULT lines found in any log files")
        return pd.DataFrame(columns=['index', 'total_score', 'total_count', 'ASR'])
    
    # Create DataFrame
    df = pd.DataFrame(all_results, columns=['index', 'total_score', 'total_count'])
    
    # Calculate ASR for each sample
    df['ASR'] = df.apply(
        lambda row: row['total_score'] / row['total_count'] if row['total_count'] > 0 else 0.0,
        axis=1
    )
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['index'], keep='first')
    
    # Sort by index
    df = df.sort_values('index').reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    
    print(f"Extracted {len(df)} results")
    print(f"Saved to: {output_csv}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Extract PAIR attack results from log files to CSV'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        help='Directory containing log files'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='pair_results.csv',
        help='Output CSV file (default: pair_results.csv)'
    )
    parser.add_argument(
        '--preview',
        type=int,
        default=10,
        help='Number of rows to preview (default: 10, 0 for none)'
    )
    
    args = parser.parse_args()
    
    try:
        # Extract results
        df = extract_all_results(args.log_dir, args.output)
        
        # Show preview
        if args.preview > 0 and len(df) > 0:
            print(f"\nPreview (first {args.preview} rows):")
            print("="*80)
            print(df.head(args.preview).to_string(index=False))
            print("="*80)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

