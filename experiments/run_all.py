"""Top-level runner, regenerate all experimental outputs.

Runs each category in order. Add new categories here as the pipeline grows.

Usage:
    python experiments/run_all.py              # everything
    python experiments/run_all.py --demonstrations
    python experiments/run_all.py --skip demonstrations
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CATEGORIES = ["demonstrations"]   # extend: "analyses", "benchmarks", "website", …


def run_category(name: str) -> None:
    if name == "demonstrations":
        from experiments.demonstrations.run_all import run_all as _run
        _run()
    else:
        raise ValueError(f"Unknown category: {name!r}")


def run_all(include: list[str] | None = None, skip: list[str] | None = None) -> None:
    targets = include if include else CATEGORIES
    if skip:
        targets = [c for c in targets if c not in skip]
    for cat in targets:
        run_category(cat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate all experimental outputs.")
    group = parser.add_mutually_exclusive_group()
    for cat in CATEGORIES:
        group.add_argument(f"--{cat}", action="store_true", dest=cat,
                           help=f"Run only the {cat!r} category.")
    parser.add_argument("--skip", nargs="*", choices=CATEGORIES,
                        help="Skip these categories.")
    args = parser.parse_args()

    include = [c for c in CATEGORIES if getattr(args, c, False)] or None
    run_all(include=include, skip=args.skip)
