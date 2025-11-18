"""
Utility for testing ML implementations.
"""
import numpy as np
import torch
from typing import Any, Callable, Optional
import time


def assert_close(actual: Any, expected: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """
    Check if two values are close (handles numpy arrays and torch tensors).
    
    Args:
        actual: Actual output
        expected: Expected output
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        True if values are close, False otherwise
    """
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        return torch.allclose(actual, expected, rtol=rtol, atol=atol)
    elif isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
        return np.allclose(actual, expected, rtol=rtol, atol=atol)
    elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(actual - expected) <= atol + rtol * abs(expected)
    else:
        return actual == expected


class MLTestRunner:
    """Helper class to run test cases for ML implementations."""
    
    def __init__(self, implementation: Any):
        self.implementation = implementation
        self.passed = 0
        self.failed = 0
    
    def run_test(
        self, 
        test_func: Callable, 
        test_name: str = "",
        rtol: float = 1e-4,
        atol: float = 1e-6
    ) -> bool:
        """
        Run a single test case.
        
        Args:
            test_func: Function that takes the implementation and returns (result, expected)
            test_name: Name for the test case
            rtol: Relative tolerance for numerical comparison
            atol: Absolute tolerance for numerical comparison
            
        Returns:
            True if test passed, False otherwise
        """
        try:
            start_time = time.time()
            result, expected = test_func(self.implementation)
            end_time = time.time()
            
            passed = assert_close(result, expected, rtol=rtol, atol=atol)
            
            if passed:
                self.passed += 1
                print(f"✓ {test_name or 'Test'} passed ({(end_time - start_time)*1000:.2f}ms)")
            else:
                self.failed += 1
                print(f"✗ {test_name or 'Test'} failed")
                print(f"  Expected shape: {getattr(expected, 'shape', 'N/A')}")
                print(f"  Got shape: {getattr(result, 'shape', 'N/A')}")
                if hasattr(expected, 'shape') and hasattr(result, 'shape'):
                    if expected.shape == result.shape:
                        if isinstance(expected, torch.Tensor):
                            diff = torch.abs(result - expected).max()
                        else:
                            diff = np.abs(result - expected).max()
                        print(f"  Max difference: {diff}")
            
            return passed
            
        except Exception as e:
            self.failed += 1
            print(f"✗ {test_name or 'Test'} raised exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_tests(self, tests: list) -> None:
        """
        Run multiple test cases.
        
        Args:
            tests: List of (test_func, test_name) tuples
        """
        print(f"\nRunning {len(tests)} test case(s)...\n")
        
        for test in tests:
            if len(test) == 1:
                test_func = test[0]
                test_name = ""
            else:
                test_func, test_name = test
            
            self.run_test(test_func, test_name)
        
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print test summary."""
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        print(f"Tests passed: {self.passed}/{total}")
        if self.failed > 0:
            print(f"Tests failed: {self.failed}/{total}")
        print(f"{'='*50}\n")


def test_ml_implementation(implementation: Any, tests: list) -> None:
    """
    Convenience function to quickly test an ML implementation.
    
    Args:
        implementation: The implementation to test (could be a function, class, module, etc.)
        tests: List of (test_func, test_name) tuples
    """
    runner = MLTestRunner(implementation)
    runner.run_tests(tests)

