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
                        help="Use sequential processing instead of parallel (default: parallel)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: all CPU cores)")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="Time limit per instance in seconds (default: 5.0)")
    parser.add_argument("--starting-cycle", type=str,
                        choices=STARTING_CYCLE_CONFIG["AVAILABLE_METHODS"],
                        default=None,
                        help=f"Starting cycle algorithm (default: {STARTING_CYCLE_CONFIG['DEFAULT_METHOD']})")
    parser.add_argument("--no-save-tours", action="store_true",
                        help="Do not save heuristic tours (default: tours saved)")
    parser.add_argument("--save-plot", action="store_true",
                        help="Force saving plots to files instead of showing interactively (default: auto-detect)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable plot generation entirely (default: plotting enabled)")

    args = parser.parse_args()

    # Determine tour saving preference
    # Default to enabled unless explicitly disabled
    save_tours = None if not args.no_save_tours else False

    # Determine plotting preference
    # Default to enabled unless explicitly disabled
    plot_enabled = not args.no_plot
    force_save_plot = args.save_plot

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
        plot=plot_enabled,
        force_save_plot=force_save_plot
    )
