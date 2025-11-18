"""
ML Implementation: Grouped Query Attention (GQA)

Description:
Implement Grouped Query Attention from scratch in PyTorch. GQA is a generalization
that interpolates between Multi-Head Attention (MHA) and Multi-Query Attention (MQA).
Query heads are divided into groups, and each group shares key and value projections.

Key Concepts:
- num_heads: Total number of query heads
- num_kv_heads: Number of key-value head groups
- Each KV head is shared by (num_heads // num_kv_heads) query heads
- When num_kv_heads = num_heads: equivalent to Multi-Head Attention
- When num_kv_heads = 1: equivalent to Multi-Query Attention

Key Requirements:
- Support for grouped attention with configurable num_kv_heads
- Proper dimension handling and broadcasting
- Causal masking support
- Efficient batched computation

Formula:
- GQA(Q, K, V) = Concat(head_1, ..., head_h)W^O
- where query heads are grouped, each group shares K and V projections

References:
- Paper: https://arxiv.org/abs/2305.13245 (GQA: Training Generalized Multi-Query Transformer)
- Used in: Llama 2, Mistral, Mixtral models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
import math


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention mechanism.
    
    Args:
        d_model: Dimension of the model (embedding dimension)
        num_heads: Number of query heads
        num_kv_heads: Number of key-value head groups (default: num_heads for MHA)
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear projections (default: True)
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # If num_kv_heads not specified, default to MHA behavior
        if num_kv_heads is None:
            num_kv_heads = num_heads
        
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads  # Dimension per head
        self.num_queries_per_kv = num_heads // num_kv_heads  # Queries per KV group
        
        # Query projection: projects to num_heads * d_k
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        
        # Key and Value projections: grouped projections
        # Projects to num_kv_heads * d_k
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k, bias=bias)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k, bias=bias)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Grouped Query Attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Optional attention mask of shape [batch_size, 1, seq_len, seq_len]
                  or [batch_size, 1, 1, seq_len] for causal masking.
                  Use 0 for positions to mask, 1 for positions to attend to.
            return_attention: If True, return attention weights along with output
            
        Returns:
            output: Tensor of shape [batch_size, seq_len, d_model]
            attention_weights (optional): Tensor of shape [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Project input to Q, K, V
        Q = self.W_q(x)  # [batch_size, seq_len, d_model]
        K = self.W_k(x)  # [batch_size, seq_len, num_kv_heads * d_k]
        V = self.W_v(x)  # [batch_size, seq_len, num_kv_heads * d_k]
        
        # 2. Reshape to multi-head format
        # Q: [batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # K, V: [batch_size, seq_len, num_kv_heads * d_k] -> [batch_size, num_kv_heads, seq_len, d_k]
        K = K.view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        
        # 3. Expand K and V to match Q's num_heads dimension
        # Each KV head is replicated num_queries_per_kv times
        # [batch_size, num_kv_heads, seq_len, d_k] -> [batch_size, num_heads, seq_len, d_k]
        K = K.repeat_interleave(self.num_queries_per_kv, dim=1)
        V = V.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # Now all have shape [batch_size, num_heads, seq_len, d_k]
        
        # 4. Compute scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores shape: [batch_size, num_heads, seq_len, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        
        # Apply dropout
        attn_weights = self.dropout(attn_weights)
        
        # Compute weighted sum of values
        attn_output = torch.matmul(attn_weights, V)  # [batch_size, num_heads, seq_len, d_k]
        
        # 5. Reshape back to [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # 6. Apply output projection
        output = self.W_o(attn_output)  # [batch_size, seq_len, d_model]
        
        if return_attention:
            return output, attn_weights
        else:
            return output
    
    @staticmethod
    def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create a causal mask for autoregressive attention.
        
        Args:
            seq_len: Sequence length
            device: Device to create the mask on
            
        Returns:
            Causal mask of shape [1, 1, seq_len, seq_len]
            Lower triangular matrix of 1s (can attend) and 0s (cannot attend)
        """
        # Create lower triangular mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        # Add batch and head dimensions
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        return mask


# ============= Test Cases =============
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    
    from utils.test_runner import test_ml_implementation
    
    # Test 1: Output shape
    def test_output_shape(gqa):
        batch_size, seq_len, d_model = 4, 16, 512
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = gqa(x)
        expected_shape = (batch_size, seq_len, d_model)
        
        return output.shape, expected_shape
    
    # Test 2: Different sequence lengths (generalization)
    def test_different_seq_lengths(gqa):
        batch_size, d_model = 2, 512
        
        # Test with different sequence lengths
        for seq_len in [1, 8, 32]:
            x = torch.randn(batch_size, seq_len, d_model)
            output = gqa(x)
            if output.shape != (batch_size, seq_len, d_model):
                return output.shape, (batch_size, seq_len, d_model)
        
        return True, True
    
    # Test 3: Causal mask creates lower triangular attention
    def test_causal_masking(gqa):
        batch_size, seq_len, d_model = 1, 8, 512
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Create causal mask
        causal_mask = GroupedQueryAttention.create_causal_mask(seq_len, x.device)
        
        # Get attention weights
        output, attn_weights = gqa(x, mask=causal_mask, return_attention=True)
        
        # Check that attention to future positions is near zero
        upper_tri = torch.triu(attn_weights[0, 0], diagonal=1)
        result = upper_tri.sum()
        expected = torch.tensor(0.0)
        
        return result, expected
    
    # Test 4: Attention weights sum to 1
    def test_attention_weights_sum(gqa):
        batch_size, seq_len, d_model = 2, 10, 512
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, attn_weights = gqa(x, return_attention=True)
        
        # Sum along last dimension (should be 1 for each query position)
        result = attn_weights.sum(dim=-1)
        expected = torch.ones(batch_size, gqa.num_heads, seq_len)
        
        return result, expected
    
    # Test 5: Batching works correctly
    def test_batching(gqa):
        seq_len, d_model = 10, 512
        
        # Process individually
        x1 = torch.randn(1, seq_len, d_model)
        x2 = torch.randn(1, seq_len, d_model)
        
        with torch.no_grad():
            out1 = gqa(x1)
            out2 = gqa(x2)
        
        # Process as batch
        x_batch = torch.cat([x1, x2], dim=0)
        with torch.no_grad():
            out_batch = gqa(x_batch)
        
        # Should be equivalent
        result = torch.cat([out1, out2], dim=0)
        expected = out_batch
        
        return result, expected
    
    # Test 6: Mask prevents attention to masked positions
    def test_masking_effectiveness(gqa):
        batch_size, seq_len, d_model = 1, 8, 512
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Create mask that blocks last 3 positions
        mask = torch.ones(batch_size, 1, seq_len, seq_len)
        mask[:, :, :, -3:] = 0  # Block last 3 positions
        
        _, attn_weights = gqa(x, mask=mask, return_attention=True)
        
        # Check that attention to masked positions is near zero
        result = attn_weights[0, 0, 0, -3:].sum()
        expected = torch.tensor(0.0)
        
        return result, expected
    
    # Test 7: Query and KV head dimensions
    def test_head_dimensions(gqa):
        # Test internal consistency
        result_queries_per_kv = gqa.num_queries_per_kv
        expected_queries_per_kv = gqa.num_heads // gqa.num_kv_heads
        
        result_total_dim = gqa.d_k * gqa.num_heads
        expected_total_dim = gqa.d_model
        
        return (result_queries_per_kv, result_total_dim), (expected_queries_per_kv, expected_total_dim)
    
    # Test 8: KV projection sizes
    def test_kv_projection_size(gqa):
        # K and V projections should be num_kv_heads * d_k
        result_k = gqa.W_k.out_features
        result_v = gqa.W_v.out_features
        expected = gqa.num_kv_heads * gqa.d_k
        
        return (result_k, result_v), (expected, expected)
    
    # Test 9: Equivalence to MQA when num_kv_heads=1
    def test_mqa_equivalence(gqa):
        # Create GQA with num_kv_heads=1 (should behave like MQA)
        gqa_as_mqa = GroupedQueryAttention(d_model=512, num_heads=8, num_kv_heads=1, dropout=0.0)
        
        result = (gqa_as_mqa.num_kv_heads, gqa_as_mqa.W_k.out_features)
        expected = (1, gqa_as_mqa.d_k)
        
        return result, expected
    
    # Test 10: Equivalence to MHA when num_kv_heads=num_heads
    def test_mha_equivalence(gqa):
        # Create GQA with num_kv_heads=num_heads (should behave like MHA)
        gqa_as_mha = GroupedQueryAttention(d_model=512, num_heads=8, num_kv_heads=8, dropout=0.0)
        
        result = (gqa_as_mha.num_kv_heads, gqa_as_mha.W_k.out_features)
        expected = (8, gqa_as_mha.d_model)
        
        return result, expected
    
    print("=" * 60)
    print("Grouped Query Attention Tests")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  d_model: 512")
    print(f"  num_heads: 8")
    print(f"  num_kv_heads: 2")
    print(f"  d_k (per head): 64")
    print(f"  Queries per KV group: 4")
    print()
    
    # Create instance with num_kv_heads=2 (4 queries per KV head)
    gqa = GroupedQueryAttention(d_model=512, num_heads=8, num_kv_heads=2, dropout=0.0)
    
    tests = [
        (test_output_shape, "Output shape test"),
        (test_different_seq_lengths, "Different sequence lengths"),
        (test_causal_masking, "Causal masking test"),
        (test_attention_weights_sum, "Attention weights sum to 1"),
        (test_batching, "Batching consistency"),
        (test_masking_effectiveness, "Masking effectiveness"),
        (test_head_dimensions, "Head dimensions correctness"),
        (test_kv_projection_size, "KV projection size"),
        (test_mqa_equivalence, "MQA equivalence (num_kv_heads=1)"),
        (test_mha_equivalence, "MHA equivalence (num_kv_heads=num_heads)"),
    ]
    
    test_ml_implementation(gqa, tests)
    
    print("\n" + "=" * 60)
    print("Implementation Tips:")
    print("=" * 60)
    print("""
1. Projection Layers:
   - W_q: Linear(d_model, d_model) - projects to num_heads * d_k
   - W_k: Linear(d_model, num_kv_heads * d_k) - grouped projections
   - W_v: Linear(d_model, num_kv_heads * d_k) - grouped projections
   
2. Reshaping:
   - Q: [batch, seq_len, d_model] -> [batch, num_heads, seq_len, d_k]
   - K: [batch, seq_len, num_kv_heads * d_k] -> [batch, num_kv_heads, seq_len, d_k]
   - V: [batch, seq_len, num_kv_heads * d_k] -> [batch, num_kv_heads, seq_len, d_k]
   
3. Expanding KV Heads:
   - Use repeat_interleave to replicate each KV head
   - K = K.repeat_interleave(num_queries_per_kv, dim=1)
   - Result: [batch, num_heads, seq_len, d_k]
   
4. Special Cases:
   - num_kv_heads = 1: Equivalent to Multi-Query Attention
   - num_kv_heads = num_heads: Equivalent to Multi-Head Attention
   
5. Benefits of GQA:
   - Balances quality and efficiency
   - Reduces KV cache size by (num_heads / num_kv_heads) factor
   - Used in Llama 2, Mistral, Mixtral models
    """)

