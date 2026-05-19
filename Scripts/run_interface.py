#!/usr/bin/env python3
"""
Unified entry point for running attack scripts.

Can be run directly:
  python Scripts/run_interface.py --attack ica --model_path Qwen/Qwen2-Audio-7B-Instruct --evaluation strongreject

Or via the installed speechjailbreaker CLI (after pip install -e .):
  speechjailbreaker --attack ica --model_path Qwen/Qwen2-Audio-7B-Instruct
"""

import sys
import os

# Ensure project root is on path when running this script directly
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Delegate to speechjailbreaker package
from speechjailbreaker.run import main

if __name__ == "__main__":
    main()