import pandas as pd
import argparse
import sys

def calculate_asr(csv_path):
    # Load the dataset
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        sys.exit(1)

    # Ensure the required columns exist
    if 'success' not in df.columns:
        print("Error: The CSV does not contain a 'success' column.")
        sys.exit(1)
    if 'index' not in df.columns:
        print("Error: The CSV does not contain an 'index' column.")
        sys.exit(1)

    # Handle aborts/missing data: Fill NaN/NA values with False
    # This explicitly ensures empty rows/aborts count as failures.
    df['success'] = df['success'].fillna('False')

    # Convert to boolean. Only explicit 'True' strings become True.
    # 'NA', 'False', or aborts evaluate to False but still count as attempts.
    df['is_success'] = df['success'].astype(str).str.strip().str.lower() == 'true'

    # ---------------------------------------------------------
    # Level 1: ASR Grouped by Index (Any success = 1 for index)
    # ---------------------------------------------------------
    success_by_index = df.groupby('index')['is_success'].any()
    
    total_unique_indices = len(success_by_index)
    successful_indices_count = success_by_index.sum()
    
    asr_level_1 = (successful_indices_count / total_unique_indices) * 100 if total_unique_indices > 0 else 0

    # ---------------------------------------------------------
    # Level 2: ASR Based on All Attempts (Regardless of index)
    # ---------------------------------------------------------
    total_attempts = len(df)
    successful_attempts_count = df['is_success'].sum()
    
    asr_level_2 = (successful_attempts_count / total_attempts) * 100 if total_attempts > 0 else 0

    # ---------------------------------------------------------
    # Print Results
    # ---------------------------------------------------------
    print(f"Analyzing File: {csv_path}\n")
    print(f"--- Level 1 ASR (Grouped by Index) ---")
    print(f"Total Unique Indices: {total_unique_indices}")
    print(f"Successful Indices:   {successful_indices_count}")
    print(f"Level 1 ASR:          {asr_level_1:.2f}%\n")

    print(f"--- Level 2 ASR (All Attempts) ---")
    print(f"Total Attempts:       {total_attempts}")
    print(f"Successful Attempts:  {successful_attempts_count}")
    print(f"Level 2 ASR:          {asr_level_2:.2f}%")

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Calculate Attack Success Rate (ASR) from a CSV.")
    parser.add_argument("csv_file", help="Path to the CSV file containing the results.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Run the calculation
    calculate_asr(args.csv_file)