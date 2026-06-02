from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS = [
    "04_synthetic_unimodal_test.py",
    "02_projection_baselines.py",
    "03_alignment_ablation.py",
    "01_multi_seed_robustness.py",
    "05_umap_param_search.py",
]


def load_script(script_name: str):
    script_path = SCRIPT_DIR / script_name
    spec = importlib.util.spec_from_file_location(script_name[:-3], script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {script_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(force_rerun: bool = False):
    for script_name in SCRIPTS:
        print(f"\n=== Running {script_name} ===")
        module = load_script(script_name)
        if hasattr(module, "main"):
            module.main(force_rerun=force_rerun)
        else:
            print(f"No main() in {script_name}; skipping")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all experiments sequentially.")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore cache and re-run all experiments.")
    args = parser.parse_args()
    main(force_rerun=args.force_rerun)
