"""
ML Implementation: [Name of the concept/algorithm]

Description:
[Brief description of what you're implementing]

References:
- [Papers, blog posts, or resources]
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class YourImplementation(nn.Module):
    """
    [Description of your implementation]
    
    Args:
        [List arguments and their descriptions]
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # TODO: Your layers/parameters here
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, output_dim]
        """
        # TODO: Your implementation here
        pass


# ============= Test Cases =============
if __name__ == "__main__":
    import sys
    import os
    # Navigate up to ml directory, then to utils
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    
    from utils.test_runner import test_ml_implementation
    
    # Test 1: Basic functionality
    def test_basic(impl):
        batch_size, seq_len, input_dim = 2, 4, 8
        x = torch.randn(batch_size, seq_len, input_dim)
        output = impl(x)
        
        # Expected shape
        expected_shape = (batch_size, seq_len, impl.output_dim)
        
        return output.shape, expected_shape
    
    # Test 2: Specific computation
    def test_computation(impl):
        # Create a simple test case where you know the expected output
        x = torch.ones(1, 1, impl.input_dim)
        output = impl(x)
        
        # Calculate expected output manually
        expected = torch.zeros(1, 1, impl.output_dim)  # Replace with actual expected
        
        return output, expected
    
    # Create instance
    model = YourImplementation(input_dim=8, output_dim=16)
    
    # Define tests
    tests = [
        (test_basic, "Basic shape test"),
        (test_computation, "Computation correctness"),
    ]
    
    # Run tests
    test_ml_implementation(model, tests)
