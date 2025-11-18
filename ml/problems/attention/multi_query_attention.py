"""
ML Implementation: Multi-Query Attention (MQA)

Description:
Implement Multi-Query Attention from scratch in PyTorch. MQA is a variant of 
multi-head attention where all heads share a single key and value projection,
but maintain separate query projections. This significantly reduces memory 
requirements and computation cost.

Key Differences from Multi-Head Attention:
- Multiple query heads (num_heads)
- Single shared key projection (d_k dimensions)
- Single shared value projection (d_v dimensions)
- Reduces KV cache size by num_heads factor
- Faster inference with minimal quality degradation

Key Requirements:
- Support for multiple query heads with shared K, V
- Proper dimension handling for broadcasting
- Causal masking support
- Efficient batched computation

Formula:
- MQA(Q, K, V) = Concat(head_1, ..., head_h)W^O
- where head_i = Attention(QW^Q_i, KW^K_shared, VW^V_shared)

References:
- Paper: https://arxiv.org/abs/1911.02150 (Fast Transformer Decoding)
- Used in: PaLM, Falcon, StarCoder models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
import math


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention mechanism with shared key and value projections.
    
    Args:
        d_model: Dimension of the model (embedding dimension)
        num_heads: Number of query heads
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear projections (default: True)
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # TODO: Initialize the following components:
        # 1. Query projection: Linear(d_model, d_model) - projects to num_heads * d_k
        # 2. Key projection: Linear(d_model, d_k) - single shared key projection
        # 3. Value projection: Linear(d_model, d_k) - single shared value projection
        # 4. Output projection: Linear(d_model, d_model)
        # 5. Dropout layer
        
        # Key difference: K and V project to d_k instead of d_model
        # This is what makes MQA more efficient!
        
        pass
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Multi-Query Attention.
        
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
        # TODO: Implement the forward pass:
        # 1. Project input to Q (with multiple heads), K (shared), V (shared)
        #    - Q: [batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, d_k]
        #    - K: [batch_size, seq_len, d_model] -> [batch_size, 1, seq_len, d_k]
        #    - V: [batch_size, seq_len, d_model] -> [batch_size, 1, seq_len, d_k]
        # 2. K and V need to be broadcasted to match Q's num_heads dimension
        # 3. Compute scaled dot-product attention
        # 4. Reshape back to [batch_size, seq_len, d_model]
        # 5. Apply output projection
        
        # Hint: K and V should be unsqueezed to add the head dimension
        # so they can broadcast across all query heads
        
        pass
    
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
        # TODO: Create a lower triangular mask
        # Hint: Use torch.tril()
        pass


# ============= Test Cases =============
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    
    from utils.test_runner import test_ml_implementation
    
    # Test 1: Output shape
    def test_output_shape(mqa):
        batch_size, seq_len, d_model = 4, 16, 512
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = mqa(x)
        expected_shape = (batch_size, seq_len, d_model)
        
        return output.shape, expected_shape
    
    # Test 2: Different sequence lengths (generalization)
    def test_different_seq_lengths(mqa):
        batch_size, d_model = 2, 512
        
        # Test with different sequence lengths
        for seq_len in [1, 8, 32]:
            x = torch.randn(batch_size, seq_len, d_model)
            output = mqa(x)
            if output.shape != (batch_size, seq_len, d_model):
                return output.shape, (batch_size, seq_len, d_model)
        
        return True, True
    
    # Test 3: Causal mask creates lower triangular attention
    def test_causal_masking(mqa):
        batch_size, seq_len, d_model = 1, 8, 512
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Create causal mask
        causal_mask = MultiQueryAttention.create_causal_mask(seq_len, x.device)
        
        # Get attention weights
        output, attn_weights = mqa(x, mask=causal_mask, return_attention=True)
        
        # Check that attention to future positions is near zero
        upper_tri = torch.triu(attn_weights[0, 0], diagonal=1)
        result = upper_tri.sum()
        expected = torch.tensor(0.0)
        
        return result, expected
    
    # Test 4: Attention weights sum to 1
    def test_attention_weights_sum(mqa):
        batch_size, seq_len, d_model = 2, 10, 512
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, attn_weights = mqa(x, return_attention=True)
        
        # Sum along last dimension (should be 1 for each query position)
        result = attn_weights.sum(dim=-1)
        expected = torch.ones(batch_size, mqa.num_heads, seq_len)
        
        return result, expected
    
    # Test 5: Batching works correctly
    def test_batching(mqa):
        seq_len, d_model = 10, 512
        
        # Process individually
        x1 = torch.randn(1, seq_len, d_model)
        x2 = torch.randn(1, seq_len, d_model)
        
        with torch.no_grad():
            out1 = mqa(x1)
            out2 = mqa(x2)
        
        # Process as batch
        x_batch = torch.cat([x1, x2], dim=0)
        with torch.no_grad():
            out_batch = mqa(x_batch)
        
        # Should be equivalent
        result = torch.cat([out1, out2], dim=0)
        expected = out_batch
        
        return result, expected
    
    # Test 6: Mask prevents attention to masked positions
    def test_masking_effectiveness(mqa):
        batch_size, seq_len, d_model = 1, 8, 512
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Create mask that blocks last 3 positions
        mask = torch.ones(batch_size, 1, seq_len, seq_len)
        mask[:, :, :, -3:] = 0  # Block last 3 positions
        
        _, attn_weights = mqa(x, mask=mask, return_attention=True)
        
        # Check that attention to masked positions is near zero
        result = attn_weights[0, 0, 0, -3:].sum()
        expected = torch.tensor(0.0)
        
        return result, expected
    
    # Test 7: Query heads dimension correctly
    def test_query_head_dimension(mqa):
        # This tests internal consistency
        result = mqa.d_k * mqa.num_heads
        expected = mqa.d_model
        
        return result, expected
    
    # Test 8: KV cache size efficiency (verify K and V projections are smaller)
    def test_kv_projection_size(mqa):
        # In MQA, K and V projections should be smaller (d_model -> d_k)
        # This is the key efficiency gain
        result_k = mqa.W_k.out_features
        result_v = mqa.W_v.out_features
        expected = mqa.d_k
        
        return (result_k, result_v), (expected, expected)
    
    print("=" * 60)
    print("Multi-Query Attention Tests")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  d_model: 512")
    print(f"  num_heads: 8")
    print(f"  d_k (per head): 64")
    print(f"  Key difference: Shared K, V projections (size {512//8})")
    print()
    
    # Create instance
    mqa = MultiQueryAttention(d_model=512, num_heads=8, dropout=0.0)
    
    tests = [
        (test_output_shape, "Output shape test"),
        (test_different_seq_lengths, "Different sequence lengths"),
        (test_causal_masking, "Causal masking test"),
        (test_attention_weights_sum, "Attention weights sum to 1"),
        (test_batching, "Batching consistency"),
        (test_masking_effectiveness, "Masking effectiveness"),
        (test_query_head_dimension, "Query head dimension correctness"),
        (test_kv_projection_size, "KV projection size efficiency"),
    ]
    
    test_ml_implementation(mqa, tests)
    
    print("\n" + "=" * 60)
    print("Implementation Tips:")
    print("=" * 60)
    print("""
1. Projection Layers:
   - W_q: Linear(d_model, d_model) - projects to num_heads * d_k
   - W_k: Linear(d_model, d_k) - SINGLE shared projection
   - W_v: Linear(d_model, d_k) - SINGLE shared projection
   
2. Reshaping:
   - Q: [batch, seq_len, d_model] -> [batch, num_heads, seq_len, d_k]
   - K: [batch, seq_len, d_k] -> [batch, 1, seq_len, d_k] (unsqueeze for broadcast)
   - V: [batch, seq_len, d_k] -> [batch, 1, seq_len, d_k] (unsqueeze for broadcast)
   
3. Attention Computation:
   - K and V broadcast across all query heads automatically
   - scores = Q @ K^T / sqrt(d_k)
   - Apply mask if provided
   - weights = softmax(scores, dim=-1)
   - output = weights @ V
   
4. Benefits of MQA:
   - Reduces KV cache size by num_heads factor
   - Faster inference with minimal accuracy loss
   - Used in PaLM, Falcon, StarCoder models
    """)

