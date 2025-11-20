"""
ML Implementation: Multi-Head Self-Attention

Description:
Implement Multi-Head Self-Attention from scratch in PyTorch, including proper 
batching and causal masking. This is a key component of the Transformer architecture.

Key Requirements:
- Support for multiple attention heads
- Proper input/output projections (Q, K, V, O)
- Causal masking for autoregressive models
- Efficient batched computation
- Correct dimension handling

Formula:
- MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
- where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)

References:
- Paper: https://arxiv.org/abs/1706.03762 (Section 3.2.2)
- The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    
    Args:
        d_model: Dimension of the model (embedding dimension)
        num_heads: Number of attention heads
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear projections (default: True)
    """
    
    def __init__(
        self, 
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        

        # params
        self.Wq = nn.Linear(*[self.d_model]*2, bias=bias)
        self.Wk = nn.Linear(*[self.d_model]*2, bias=bias)
        self.Wv = nn.Linear(*[self.d_model]*2, bias=bias)

        # MISTAKE!! don't forget this 
        self.Wo = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(p=dropout)



    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, s, d_model]
        # mask: [1, 1, s, s]
        bs, s, _ = x.shape
        d_head = self.d_model//self.num_heads

        # compute Q, K, V
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x) # [bs, s, d_model]

        # split d_model into num_heads
        Q = Q.view(bs, s, self.num_heads, d_head).transpose(1, 2) # [bs, num_heads, s, d_head]
        K = K.view(bs, s, self.num_heads, d_head).transpose(1, 2) # [bs, num_heads, s, d_head]
        V = V.view(bs, s, self.num_heads, d_head).transpose(1, 2) # [bs, num_heads, s, d_head]

        # perform SDPMM for each head
        ## SDP
        scores: torch.Tensor = Q @ K.transpose(-2, -1)
        scores = scores / torch.sqrt(torch.tensor(d_head))
        
        ## mask
        if mask is not None:
            scores.masked_fill_(mask==0, -torch.inf)
        
        ## softmax
        scores = F.softmax(scores, dim=-1)

        ## dropout
        scores = self.dropout(scores) # [bs, num_heads, s, s]

        ## calculate outputs
        outs: torch.Tensor = scores @ V # [bs, num_heads, s, d_head]

        # concate heads back together (only the outputs, not attn scores)
        outs = outs.transpose(1, 2).contiguous().view(bs, s, self.d_model) # [bs, s, d_model]

        outs = self.Wo(outs)

        if return_attention:
            return outs, scores
        else:
            return outs
        


    @staticmethod
    def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones([seq_len, seq_len]))
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask
        


# ============= Test Cases =============
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    
    from utils.test_runner import test_ml_implementation
    
    # Test 1: Output shape
    def test_output_shape(mha):
        batch_size, seq_len, d_model = 4, 16, 512
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = mha(x)
        expected_shape = (batch_size, seq_len, d_model)
        
        return output.shape, expected_shape
    
    # Test 2: Different sequence lengths (generalization)
    def test_different_seq_lengths(mha):
        batch_size, d_model = 2, 512
        
        # Test with different sequence lengths
        for seq_len in [1, 8, 32]:
            x = torch.randn(batch_size, seq_len, d_model)
            output = mha(x)
            if output.shape != (batch_size, seq_len, d_model):
                return output.shape, (batch_size, seq_len, d_model)
        
        return True, True
    
    # Test 3: Causal mask creates lower triangular attention
    def test_causal_masking(mha):
        batch_size, seq_len, d_model = 1, 8, 512
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Create causal mask
        causal_mask = MultiHeadAttention.create_causal_mask(seq_len, x.device)
        
        # Get attention weights
        output, attn_weights = mha(x, mask=causal_mask, return_attention=True)
        
        # Check that attention to future positions is near zero
        # Sum attention weights in upper triangular part (excluding diagonal)
        upper_tri = torch.triu(attn_weights[0, 0], diagonal=1)
        result = upper_tri.sum()
        expected = torch.tensor(0.0)
        
        return result, expected
    
    # Test 4: Attention weights sum to 1
    def test_attention_weights_sum(mha):
        batch_size, seq_len, d_model = 2, 10, 512
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, attn_weights = mha(x, return_attention=True)
        
        # Sum along last dimension (should be 1 for each query position)
        result = attn_weights.sum(dim=-1)
        expected = torch.ones(batch_size, mha.num_heads, seq_len)
        
        return result, expected
    
    # Test 5: Batching works correctly
    def test_batching(mha):
        seq_len, d_model = 10, 512
        
        # Process individually
        x1 = torch.randn(1, seq_len, d_model)
        x2 = torch.randn(1, seq_len, d_model)
        
        with torch.no_grad():
            out1 = mha(x1)
            out2 = mha(x2)
        
        # Process as batch
        x_batch = torch.cat([x1, x2], dim=0)
        with torch.no_grad():
            out_batch = mha(x_batch)
        
        # Should be equivalent
        result = torch.cat([out1, out2], dim=0)
        expected = out_batch
        
        return result, expected
    
    # Test 6: Mask prevents attention to masked positions
    def test_masking_effectiveness(mha):
        batch_size, seq_len, d_model = 1, 8, 512
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Create mask that blocks last 3 positions
        mask = torch.ones(batch_size, 1, seq_len, seq_len)
        mask[:, :, :, -3:] = 0  # Block last 3 positions
        
        _, attn_weights = mha(x, mask=mask, return_attention=True)
        
        # Check that attention to masked positions is near zero
        result = attn_weights[0, 0, 0, -3:].sum()
        expected = torch.tensor(0.0)
        
        return result, expected
    
    # Test 7: Multi-head splits dimension correctly
    def test_head_dimension(mha):
        # This tests internal consistency
        result = mha.d_k * mha.num_heads
        expected = mha.d_model
        
        return result, expected
    
    # Test 8: Output projection shape
    def test_output_projection(mha):
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, mha.d_model)
        
        output = mha(x)
        
        # Output should have same shape as input
        result = (output.shape == x.shape)
        expected = True
        
        return result, expected
    
    print("=" * 60)
    print("Multi-Head Attention Tests")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  d_model: 512")
    print(f"  num_heads: 8")
    print(f"  d_k (per head): 64")
    print()
    
    # Create instance
    mha = MultiHeadAttention(d_model=512, num_heads=8, dropout=0.0)
    
    tests = [
        (test_output_shape, "Output shape test"),
        (test_different_seq_lengths, "Different sequence lengths"),
        (test_causal_masking, "Causal masking test"),
        (test_attention_weights_sum, "Attention weights sum to 1"),
        (test_batching, "Batching consistency"),
        (test_masking_effectiveness, "Masking effectiveness"),
        (test_head_dimension, "Head dimension correctness"),
        (test_output_projection, "Output projection shape"),
    ]
    
    test_ml_implementation(mha, tests)
    
    print("\n" + "=" * 60)
    print("Implementation Tips:")
    print("=" * 60)
    print("""
1. Q, K, V Projections:
   - Option A: 3 separate Linear(d_model, d_model) layers
   - Option B: 1 Linear(d_model, 3*d_model) then split
   
2. Reshaping for Multi-Head:
   - Input: [batch, seq_len, d_model]
   - After projection: [batch, seq_len, num_heads, d_k]
   - Transpose to: [batch, num_heads, seq_len, d_k]
   
3. Scaled Dot-Product Attention:
   - scores = Q @ K^T / sqrt(d_k)
   - Apply mask: scores.masked_fill(mask == 0, -1e9)
   - weights = softmax(scores, dim=-1)
   - output = weights @ V
   
4. Output:
   - Transpose back to [batch, seq_len, num_heads, d_k]
   - Reshape to [batch, seq_len, d_model]
   - Apply output projection
    """)

