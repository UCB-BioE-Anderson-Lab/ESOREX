"""Regenerate all demonstration assets.

Usage:
    python experiments/demonstrations/run_all.py
    python experiments/demonstrations/run_all.py --demo tyrb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEMOS = {
    "tyrb":    "experiments.demonstrations.generate_tyrb",
    "fuctiii": "experiments.demonstrations.generate_fuctiii",
}


def run_demo(name: str) -> None:
    import importlib
    mod = importlib.import_module(DEMOS[name])
    mod.main()


def run_all(demos: list[str] | None = None) -> None:
    targets = demos if demos else list(DEMOS)
    for name in targets:
        print(f"\n{'='*60}")
        print(f"  demonstration: {name}")
        print(f"{'='*60}")
        run_demo(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate demonstration assets.")
    parser.add_argument("--demo", nargs="*", choices=list(DEMOS),
                        help="Run specific demonstration(s). Omit to run all.")
    args = parser.parse_args()
    run_all(args.demo)
