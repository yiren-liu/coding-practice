"""
ML Implementation: Scaled Dot-Product Attention

Description:
Implements the core attention mechanism from "Attention Is All You Need" (Vaswani et al., 2017).
The attention function maps a query and a set of key-value pairs to an output.

Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

References:
- Paper: https://arxiv.org/abs/1706.03762
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.
    
    Args:
        query: Query tensor of shape [batch_size, num_heads, seq_len_q, d_k]
        key: Key tensor of shape [batch_size, num_heads, seq_len_k, d_k]
        value: Value tensor of shape [batch_size, num_heads, seq_len_v, d_v]
        mask: Optional mask tensor of shape [batch_size, 1, seq_len_q, seq_len_k]
        dropout: Optional dropout layer
        
    Returns:
        output: Attention output of shape [batch_size, num_heads, seq_len_q, d_v]
        attention_weights: Attention weights of shape [batch_size, num_heads, seq_len_q, seq_len_k]
    """
    d_k = query.size(-1)
    
    # Compute attention scores: QK^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    
    # Apply mask if provided (set masked positions to large negative value)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply dropout if provided
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    # Compute weighted sum of values
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention as a module.
    
    Args:
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            query: [batch_size, num_heads, seq_len_q, d_k]
            key: [batch_size, num_heads, seq_len_k, d_k]
            value: [batch_size, num_heads, seq_len_v, d_v]
            mask: Optional [batch_size, 1, seq_len_q, seq_len_k]
            
        Returns:
            output: [batch_size, num_heads, seq_len_q, d_v]
            attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        return scaled_dot_product_attention(query, key, value, mask, self.dropout)


# ============= Test Cases =============
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from utils.test_runner import test_ml_implementation
    
    # Test 1: Output shape
    def test_output_shape(impl):
        batch_size, num_heads, seq_len, d_k = 2, 8, 10, 64
        
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
        batch_size, num_heads, seq_len, d_k = 2, 4, 5, 32
        
        query = torch.randn(batch_size, num_heads, seq_len, d_k)
        key = torch.randn(batch_size, num_heads, seq_len, d_k)
        value = torch.randn(batch_size, num_heads, seq_len, d_k)
        
        _, attn_weights = impl(query, key, value)
        
        # Sum along the last dimension (should be 1 for each query position)
        result = attn_weights.sum(dim=-1)
        expected = torch.ones(batch_size, num_heads, seq_len)
        
        return result, expected
    
    # Test 3: Masking works correctly
    def test_masking(impl):
        batch_size, num_heads, seq_len, d_k = 1, 1, 4, 8
        
        query = torch.randn(batch_size, num_heads, seq_len, d_k)
        key = torch.randn(batch_size, num_heads, seq_len, d_k)
        value = torch.randn(batch_size, num_heads, seq_len, d_k)
        
        # Create a causal mask (lower triangular)
        mask = torch.tril(torch.ones(batch_size, 1, seq_len, seq_len))
        
        _, attn_weights = impl(query, key, value, mask)
        
        # Masked positions should have near-zero attention
        # Check upper triangular part (excluding diagonal) should be ~0
        result = attn_weights[0, 0, 0, 1:]  # First query's attention to future positions
        expected = torch.zeros(seq_len - 1)
        
        return result, expected
    
    # Test 4: Self-attention on identical sequences
    def test_self_attention(impl):
        batch_size, num_heads, seq_len, d_k = 1, 1, 3, 4
        
        # All queries, keys, values are the same
        x = torch.eye(seq_len, d_k).unsqueeze(0).unsqueeze(0)
        
        output, attn_weights = impl(x, x, x)
        
        # Each position should attend to all positions
        # The attention weights should be uniform without strong preferences
        result = attn_weights.sum(dim=-1)  # Should sum to 1
        expected = torch.ones(batch_size, num_heads, seq_len)
        
        return result, expected
    
    # Create instance
    attention = ScaledDotProductAttention(dropout=0.0)  # No dropout for deterministic tests
    
    tests = [
        (test_output_shape, "Output shape test"),
        (test_attention_weights_sum, "Attention weights sum to 1"),
        (test_masking, "Masking test"),
        (test_self_attention, "Self-attention test"),
    ]
    
    test_ml_implementation(attention, tests)

