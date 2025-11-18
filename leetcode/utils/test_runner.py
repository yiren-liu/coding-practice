"""
Utility for running LeetCode-style test cases.
"""
from typing import Any, Callable, List, Tuple
import time


class TestRunner:
    """Helper class to run test cases for LeetCode problems."""
    
    def __init__(self, solution_func: Callable):
        self.solution_func = solution_func
        self.passed = 0
        self.failed = 0
    
    def run_test(self, inputs: Tuple, expected: Any, test_name: str = "") -> bool:
        """
        Run a single test case.
        
        Args:
            inputs: Tuple of arguments to pass to the solution function
            expected: Expected output
            test_name: Optional name for the test case
            
        Returns:
            True if test passed, False otherwise
        """
        try:
            start_time = time.time()
            result = self.solution_func(*inputs)
            end_time = time.time()
            
            passed = result == expected
            
            if passed:
                self.passed += 1
                print(f"✓ {test_name or 'Test'} passed ({(end_time - start_time)*1000:.2f}ms)")
            else:
                self.failed += 1
                print(f"✗ {test_name or 'Test'} failed")
                print(f"  Input: {inputs}")
                print(f"  Expected: {expected}")
                print(f"  Got: {result}")
            
            return passed
            
        except Exception as e:
            self.failed += 1
            print(f"✗ {test_name or 'Test'} raised exception: {e}")
            print(f"  Input: {inputs}")
            return False
    
    def run_tests(self, test_cases: List[Tuple[Tuple, Any, str]]) -> None:
        """
        Run multiple test cases.
        
        Args:
            test_cases: List of (inputs, expected, test_name) tuples
        """
        print(f"\nRunning {len(test_cases)} test case(s)...\n")
        
        for test_case in test_cases:
            if len(test_case) == 2:
                inputs, expected = test_case
                test_name = ""
            else:
                inputs, expected, test_name = test_case
            
            self.run_test(inputs, expected, test_name)
        
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print test summary."""
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        print(f"Tests passed: {self.passed}/{total}")
        if self.failed > 0:
            print(f"Tests failed: {self.failed}/{total}")
        print(f"{'='*50}\n")


def test_solution(solution_func: Callable, test_cases: List[Tuple]) -> None:
    """
    Convenience function to quickly test a solution.
    
    Args:
        solution_func: The solution function to test
        test_cases: List of (inputs, expected) or (inputs, expected, test_name) tuples
    """
    runner = TestRunner(solution_func)
    runner.run_tests(test_cases)

