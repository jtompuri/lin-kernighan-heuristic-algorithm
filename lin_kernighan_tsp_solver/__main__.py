"""
Runs the Lin-Kernighan TSP solver package as a module.

Usage:
    python -m lin_kernighan_tsp_solver [options]
"""

import argparse
from .main import main
from .config import STARTING_CYCLE_CONFIG

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m lin_kernighan_tsp_solver",
        description="Lin-Kernighan TSP Solver"
    )
    parser.add_argument("files", nargs="*",
                        help="TSP files to solve (optional). If not provided, processes all files in the TSP folder")
    parser.add_argument("--sequential", action="store_true",
                        help="Use sequential processing instead of parallel")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: all CPU cores)")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="Time limit per instance in seconds")
    parser.add_argument("--starting-cycle", type=str,
                        choices=STARTING_CYCLE_CONFIG["AVAILABLE_METHODS"],
                        default=None,
                        help=f"Starting cycle algorithm (default: {STARTING_CYCLE_CONFIG['DEFAULT_METHOD']})")
    parser.add_argument("--save-tours", action="store_true",
                        help="Save heuristic tours to solutions/ folder")
    parser.add_argument("--no-save-tours", action="store_true",
                        help="Do not save heuristic tours (overrides config)")
    
    # Numba optimization flags
    parser.add_argument("--enable-numba", action="store_true",
                        help="Enable Numba JIT optimizations for better performance on large problems")
    parser.add_argument("--disable-numba", action="store_true",
                        help="Disable Numba JIT optimizations (use original Python implementation)")
    parser.add_argument("--numba-threshold", type=int, default=None,
                        help="Minimum problem size to use Numba optimizations (default: 30)")

    args = parser.parse_args()

    # Determine tour saving preference
    save_tours = None
    if args.save_tours:
        save_tours = True
    elif args.no_save_tours:
        save_tours = False
    
    # Determine Numba preference
    numba_enabled = None
    if args.enable_numba:
        numba_enabled = True
    elif args.disable_numba:
        numba_enabled = False

    # If specific files are provided, use sequential processing for single files
    use_parallel = not args.sequential
    if args.files and len(args.files) == 1:
        use_parallel = False
        print("Single file specified, using sequential processing.")

    main(
        use_parallel=use_parallel,
        max_workers=args.workers,
        time_limit=args.time_limit,
        starting_cycle_method=args.starting_cycle,
        tsp_files=args.files if args.files else None,
        save_tours=save_tours,
        numba_enabled=numba_enabled,
        numba_threshold=args.numba_threshold
    )
