"""
Runs the Lin-Kernighan TSP solver package as a module.

Usage:
    python -m lin_kernighan_tsp_solver [options]
"""

import argparse
from .main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lin-Kernighan TSP Solver")
    parser.add_argument("--sequential", action="store_true",
                        help="Use sequential processing instead of parallel")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: all CPU cores)")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="Time limit per instance in seconds")

    args = parser.parse_args()

    main(
        use_parallel=not args.sequential,
        max_workers=args.workers,
        time_limit=args.time_limit
    )
