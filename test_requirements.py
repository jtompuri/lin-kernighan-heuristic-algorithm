#!/usr/bin/env python3
"""
Test script to verify requirements compatibility.
Run this to ensure all dependencies work together correctly.
"""


def test_core_imports():
    """Test that all core dependencies can be imported."""
    print("Testing core imports...")
    
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
    
    import scipy
    print(f"✓ SciPy {scipy.__version__}")
    
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__}")
    
    try:
        import numba
        print(f"✓ Numba {numba.__version__}")
        numba_available = True
    except ImportError as e:
        print(f"✗ Numba import failed: {e}")
        numba_available = False
    
    return numba_available


def test_numba_compatibility():
    """Test Numba JIT compilation."""
    print("\nTesting Numba compatibility...")
    
    try:
        from numba import jit
        
        @jit(nopython=True)
        def test_function(x):
            return x * 2 + 1
        
        result = test_function(5)
        expected = 11
        
        if result == expected:
            print("✓ Numba JIT compilation successful")
            return True
        else:
            print(f"✗ Numba JIT test failed: got {result}, expected {expected}")
            return False
            
    except Exception as e:
        print(f"✗ Numba JIT test failed: {e}")
        return False


def test_tsp_solver_import():
    """Test that the TSP solver modules can be imported."""
    print("\nTesting TSP solver imports...")
    
    try:
        from lin_kernighan_tsp_solver.lk_algorithm_numba import get_numba_status
        status = get_numba_status()
        print(f"✓ TSP solver imports successful, Numba available: {status['available']}")
        return True
    except Exception as e:
        print(f"✗ TSP solver import failed: {e}")
        return False


if __name__ == "__main__":
    print("=== Requirements Compatibility Test ===")
    
    numba_imports = test_core_imports()
    numba_works = test_numba_compatibility() if numba_imports else False
    tsp_works = test_tsp_solver_import()
    
    print("\n=== Summary ===")
    if numba_works and tsp_works:
        print("✓ All requirements working correctly!")
        exit(0)
    else:
        print("✗ Some requirements have issues")
        exit(1)
