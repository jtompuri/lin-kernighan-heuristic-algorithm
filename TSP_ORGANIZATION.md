# TSP File Organization Structure

This document describes the reorganized folder structure for TSP files in the Lin-Kernighan TSP solver project.

## New Structure

```
documentation/                   # Project documentation
├── implementation_specification.md
├── requirements_specification.md
├── testing_specification.md
└── reports/                     # Weekly progress reports
    ├── weekly_report_1.md
    ├── weekly_report_2.md
    ├── weekly_report_3.md
    ├── weekly_report_4.md
    ├── weekly_report_5.md
    ├── weekly_report_6.md
    └── weekly_report_7.md

helpers/                         # Helper scripts for TSP problem management
├── create_tsp_problem.py        # Script to generate random TSP problems
├── exact_tsp_solver.py          # Brute-force exact TSP solver for small instances
└── simple_tsp_solver.py         # Simple TSP solver using nearest neighbor heuristic

problems/
├── my_tsp30.tsp                 # Custom TSP instances
├── random50.tsp
├── random500.tsp
├── stipples_high-key-2-2048px.tsp
├── tsplib95/                    # TSPLIB95 benchmark instances
│   ├── a280.tsp / a280.opt.tour
│   ├── berlin52.tsp / berlin52.opt.tour
│   ├── ch130.tsp / ch130.opt.tour
│   ├── ch150.tsp / ch150.opt.tour
│   ├── eil101.tsp / eil101.opt.tour
│   ├── eil51.tsp / eil51.opt.tour
│   ├── eil76.tsp / eil76.opt.tour
│   ├── kroA100.tsp / kroA100.opt.tour
│   ├── kroC100.tsp / kroC100.opt.tour
│   ├── kroD100.tsp / kroD100.opt.tour
│   ├── lin105.tsp / lin105.opt.tour
│   ├── pcb442.tsp / pcb442.opt.tour
│   ├── pr1002.tsp / pr1002.opt.tour
│   ├── pr2392.tsp / pr2392.opt.tour
│   ├── pr76.tsp / pr76.opt.tour
│   ├── rd100.tsp / rd100.opt.tour
│   ├── st70.tsp / st70.opt.tour
│   └── tsp225.tsp / tsp225.opt.tour
└── random/                      # Random test instances (used by tests)
    ├── rand4.tsp / rand4.opt.tour
    ├── rand5.tsp / rand5.opt.tour
    ├── rand6.tsp / rand6.opt.tour
    ├── rand7.tsp / rand7.opt.tour
    ├── rand8.tsp / rand8.opt.tour
    ├── rand9.tsp / rand9.opt.tour
    ├── rand10.tsp / rand10.opt.tour
    ├── rand11.tsp / rand11.opt.tour
    └── rand12.tsp / rand12.opt.tour
```

## Changes Made

### ✅ **Moved Folders:**
- `verifications/tsplib95/` → `problems/tsplib95/`
- `verifications/random/` → `problems/random/`
- `verifications/single/pcb442.*` → `problems/tsplib95/` (merged)
- `tsp/my_tsp30.tsp` → `problems/`

### ✅ **Removed Folders:**
- `verifications/` (empty after reorganization)
- `tsp/` (empty after reorganization)
- `verifications/single/` (not used by code, merged with tsplib95)

### ✅ **Updated Configuration:**
- **`lin_kernighan_tsp_solver/config.py`**: `TSP_FOLDER_PATH` updated to `problems/tsplib95/`
- **`tests/conftest.py`**: `VERIFICATION_RANDOM_PATH` updated to `problems/random/`
- **`tests/test_chained_lk.py`**: `VERIFICATION_RANDOM_PATH` updated to `problems/random/`
- **`helpers/simple_tsp_solver.py`**: `TSP_FOLDER_PATH` updated to `problems/tsplib95/`
- **`helpers/exact_tsp_solver.py`**: `OUTPUT_SUBDIRECTORY` updated to `problems/`
- **`README.md`**: Updated folder references, examples, and weekly report links

## Benefits

1. **Centralized Organization**: All TSP-related files are now under `problems/`
2. **Cleaner Structure**: Removed unused folders (`single/`, `tsp/`)
3. **Logical Grouping**: 
   - `tsplib95/` - Standard benchmark instances
   - `random/` - Test instances for automated testing
   - Root - Custom/generated instances and tools
4. **Maintained Functionality**: All existing tests and functionality work unchanged

## File Counts
- **Total TSP files**: 31
- **Total tour files**: 27
- **TSPLIB95 instances**: 18 (with optimal tours)
- **Random test instances**: 9 (with optimal tours)
- **Custom instances**: 4

## Usage Examples

```bash
# Process all TSPLIB95 instances (default behavior)
python -m lin_kernighan_tsp_solver

# Process specific instance
python -m lin_kernighan_tsp_solver problems/tsplib95/berlin52.tsp

# Generate new random instance using the helper script
python helpers/create_tsp_problem.py 100 my_random100.tsp

# Solve small instances exactly using brute-force
python helpers/exact_tsp_solver.py

# Run simple TSP solver on all TSPLIB95 instances
python helpers/simple_tsp_solver.py
```

The reorganization maintains backward compatibility while providing a cleaner, more logical file structure.
