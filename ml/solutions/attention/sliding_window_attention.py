"""
ML Implementation: Sliding Window Attention

Description:
Implements sliding window attention, an efficient attention mechanism where each token 
only attends to a fixed-size window of nearby tokens. This reduces computational 
complexity from O(n²) to O(n*w) where w is the window size.

Used in models like:
- Longformer (Beltagy et al., 2020)
- BigBird (Zaheer et al., 2020)
- Local attention in Sparse Transformers

Key Concepts:
- Each query attends to a fixed window of keys (e.g., w tokens before and after)
- Reduces memory and computation for long sequences
- Still captures local context effectively
- Can be combined with global attention for long-range dependencies

Formula: 
- Attention(Q, K, V)_i = softmax(Q_i * K_{i-w:i+w}^T / sqrt(d_k)) * V_{i-w:i+w}
- where w is the window size (radius on each side)

References:
- Longformer: https://arxiv.org/abs/2004.05150
- BigBird: https://arxiv.org/abs/2007.14062
- Local Attention in Sparse Transformers: https://arxiv.org/abs/1904.10509
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


def sliding_window_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    window_size: int,
    dropout: Optional[nn.Dropout] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute sliding window attention.
    
    Args:
        query: Query tensor of shape [batch_size, num_heads, seq_len, d_k]
        key: Key tensor of shape [batch_size, num_heads, seq_len, d_k]
        value: Value tensor of shape [batch_size, num_heads, seq_len, d_v]
        window_size: Size of the attention window (radius on each side)
                    Total window = 2 * window_size + 1 (includes current position)
        dropout: Optional dropout layer
        
    Returns:
        output: Attention output of shape [batch_size, num_heads, seq_len, d_v]
        attention_weights: Attention weights of shape [batch_size, num_heads, seq_len, seq_len]
    """
    batch_size, num_heads, seq_len, d_k = query.shape
    device = query.device
    
    # Compute attention scores: QK^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    
    # Create sliding window mask (band diagonal matrix)
    # Each position i can attend to positions [i-window_size, i+window_size]
    # Use torch.triu and torch.tril to create a band around the diagonal
    mask = torch.ones(seq_len, seq_len, device=device)
    
    # torch.triu(mask, diagonal=k) keeps upper triangle starting from k-th diagonal
    # torch.tril(mask, diagonal=k) keeps lower triangle up to k-th diagonal
    # Combining them creates a band of width 2*window_size + 1
    mask = torch.triu(mask, diagonal=-window_size)  # Keep everything from -window_size diagonal and above
    mask = torch.tril(mask, diagonal=window_size)   # Keep everything up to +window_size diagonal
    
    # Reshape mask to [1, 1, seq_len, seq_len] for broadcasting
    mask = mask.unsqueeze(0).unsqueeze(0)
    
    # Apply mask (set positions outside window to large negative value)
    scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply dropout if provided
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    # Compute weighted sum of values
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention as a module.
    
    Args:
        window_size: Size of attention window (radius on each side)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, window_size: int, dropout: float = 0.1):
        super().__init__()
        self.window_size = window_size
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            query: [batch_size, num_heads, seq_len, d_k]
            key: [batch_size, num_heads, seq_len, d_k]
            value: [batch_size, num_heads, seq_len, d_v]
            
        Returns:
            output: [batch_size, num_heads, seq_len, d_v]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        return sliding_window_attention(query, key, value, self.window_size, self.dropout)


# ============= Test Cases =============
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from utils.test_runner import test_ml_implementation
    
    # Test 1: Output shape
    def test_output_shape(impl):
        batch_size, num_heads, seq_len, d_k = 2, 8, 16, 64
        
        query = torch.randn(batch_size, num_heads, seq_len, d_k)
        key = torch.randn(batch_size, num_heads, seq_len, d_k)
        value = torch.randn(batch_size, num_heads, seq_len, d_k)
        
        output, attn_weights = impl(query, key, value)
        
        expected_output_shape = (batch_size, num_heads, seq_len, d_k)
        expected_attn_shape = (batch_size, num_heads, seq_len, seq_len)
        
        result = (output.shape, attn_weights.shape)
        expected = (expected_output_shape, expected_attn_shape)
        
        return result, expected
    
    # Test 2: Attention weights sum to 1
    def test_attention_weights_sum(impl):
        batch_size, num_heads, seq_len, d_k = 2, 4, 10, 32
        
        query = torch.randn(batch_size, num_heads, seq_len, d_k)
        key = torch.randn(batch_size, num_heads, seq_len, d_k)
        value = torch.randn(batch_size, num_heads, seq_len, d_k)
        
        _, attn_weights = impl(query, key, value)
        
        # Sum along the last dimension (should be 1 for each query position)
        result = attn_weights.sum(dim=-1)
        expected = torch.ones(batch_size, num_heads, seq_len)
        
        return result, expected
    
    # Test 3: Window constraint is enforced
    def test_window_constraint(impl):
        batch_size, num_heads, seq_len, d_k = 1, 1, 12, 8
        window_size = impl.window_size  # Should be 2
        
        query = torch.randn(batch_size, num_heads, seq_len, d_k)
        key = torch.randn(batch_size, num_heads, seq_len, d_k)
        value = torch.randn(batch_size, num_heads, seq_len, d_k)
        
        _, attn_weights = impl(query, key, value)
        
        # Check middle position (index 6)
        # Should only attend to positions [4, 5, 6, 7, 8] (window_size=2)
        # All positions outside [4:9] should be near zero
        position = 6
        left_positions = attn_weights[0, 0, position, :position-window_size]
        right_positions = attn_weights[0, 0, position, position+window_size+1:]
        
        # Sum of attention outside window should be near zero
        result = left_positions.sum() + right_positions.sum()
        expected = torch.tensor(0.0)
        
        return result, expected
    
    # Test 4: Edge positions handle boundaries correctly
    def test_edge_positions(impl):
        batch_size, num_heads, seq_len, d_k = 1, 1, 10, 8
        
        query = torch.randn(batch_size, num_heads, seq_len, d_k)
        key = torch.randn(batch_size, num_heads, seq_len, d_k)
        value = torch.randn(batch_size, num_heads, seq_len, d_k)
        
        _, attn_weights = impl(query, key, value)
        
        # First position (index 0) should only attend to [0, 1, 2]
        # Check that positions beyond window are zero
        result = attn_weights[0, 0, 0, 3:].sum()
        expected = torch.tensor(0.0)
        
        return result, expected
    
    # Test 5: Last position handles boundaries correctly
    def test_last_position(impl):
        batch_size, num_heads, seq_len, d_k = 1, 1, 10, 8
        
        query = torch.randn(batch_size, num_heads, seq_len, d_k)
        key = torch.randn(batch_size, num_heads, seq_len, d_k)
        value = torch.randn(batch_size, num_heads, seq_len, d_k)
        
        _, attn_weights = impl(query, key, value)
        
        # Last position (index 9) should only attend to [7, 8, 9]
        # Check that positions before window are zero
        result = attn_weights[0, 0, -1, :-3].sum()
        expected = torch.tensor(0.0)
        
        return result, expected
    
    # Test 6: Attention is local (non-zero only in window)
    def test_locality(impl):
        batch_size, num_heads, seq_len, d_k = 1, 1, 20, 8
        window_size = impl.window_size
        
        query = torch.randn(batch_size, num_heads, seq_len, d_k)
        key = torch.randn(batch_size, num_heads, seq_len, d_k)
        value = torch.randn(batch_size, num_heads, seq_len, d_k)
        
        _, attn_weights = impl(query, key, value)
        
        # For a position in the middle, check that only window positions have attention
        position = seq_len // 2
        window_start = max(0, position - window_size)
        window_end = min(seq_len, position + window_size + 1)
        
        # Attention within window should sum to 1
        result = attn_weights[0, 0, position, window_start:window_end].sum()
        expected = torch.tensor(1.0)
        
        return result, expected
    
    # Test 7: Works with different window sizes
    def test_different_window_sizes(impl):
        batch_size, num_heads, seq_len, d_k = 1, 1, 8, 8
        
        # Test that the implementation respects its window_size
        query = torch.randn(batch_size, num_heads, seq_len, d_k)
        key = torch.randn(batch_size, num_heads, seq_len, d_k)
        value = torch.randn(batch_size, num_heads, seq_len, d_k)
        
        output, attn_weights = impl(query, key, value)
        
        # Output should have correct shape
        result = output.shape
        expected = (batch_size, num_heads, seq_len, d_k)
        
        return result, expected
    
    print("=" * 60)
    print("Sliding Window Attention Tests")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Window size: 2 (attends to 5 positions total)")
    print(f"  Complexity: O(n*w) vs O(n²) for full attention")
    print()
    
    # Create instance with window_size=2
    attention = SlidingWindowAttention(window_size=2, dropout=0.0)
    
    tests = [
        (test_output_shape, "Output shape test"),
        (test_attention_weights_sum, "Attention weights sum to 1"),
        (test_window_constraint, "Window constraint enforced"),
        (test_edge_positions, "Edge positions handle boundaries"),
        (test_last_position, "Last position handles boundaries"),
        (test_locality, "Attention is local"),
        (test_different_window_sizes, "Different window sizes work"),
    ]
    
    test_ml_implementation(attention, tests)
    
    print("\n" + "=" * 60)
    print("Implementation Tips:")
    print("=" * 60)
    print("""
1. Creating the Sliding Window Mask:
   - Use torch.tril() and torch.triu() to create band diagonal matrix
   - Example: mask = torch.triu(torch.tril(ones, diagonal=k), diagonal=-k)
   - This creates a band of 1s with width 2k+1 around the diagonal
   
2. Applying the Mask:
   - Compute scores = Q @ K^T / sqrt(d_k)
   - Apply mask: scores.masked_fill(mask == 0, -1e9)
   - This sets positions outside window to -inf before softmax
   
3. Edge Cases:
   - First tokens: window extends only to the right
   - Last tokens: window extends only to the left
   - The mask handles this automatically with boundary conditions
   
4. Efficiency Considerations (Optional):
   - Full attention with masking: Simple but computes unnecessary values
   - True sliding window: More complex indexing but saves computation
   - For long sequences, true sliding window is preferred
   
5. Comparison with Full Attention:
   - Full attention: Each token attends to ALL tokens - O(n²)
   - Sliding window: Each token attends to 2w+1 tokens - O(n*w)
   - Memory and compute savings are significant for long sequences
    """)


