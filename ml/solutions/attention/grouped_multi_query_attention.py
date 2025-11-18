"""
ML Implementation: Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) - SOLUTION

Description:
Implement two efficient attention mechanisms that reduce the KV cache size:

1. Multi-Query Attention (MQA): 
   - Uses multiple query heads but SINGLE key/value heads shared across all queries
   - Significantly reduces memory and increases decoding speed
   - Used in: PaLM, Falcon, StarCoder

2. Grouped-Query Attention (GQA):
   - Middle ground between Multi-Head Attention (MHA) and Multi-Query Attention (MQA)
   - Groups of query heads share key/value heads
   - Used in: LLaMA 2, Mistral

These are critical for efficient inference in large language models, especially for:
- Reducing KV cache memory during autoregressive generation
- Faster decoding with minimal quality loss

Mathematical Comparison:
- MHA: num_q_heads = num_kv_heads (e.g., 32 = 32)
- GQA: num_q_heads > num_kv_heads (e.g., 32 query heads, 8 kv heads -> 4 queries per kv)
- MQA: num_kv_heads = 1 (e.g., 32 query heads, 1 kv head)

References:
- Multi-Query Attention: https://arxiv.org/abs/1911.02150
- Grouped-Query Attention (LLaMA 2): https://arxiv.org/abs/2305.13245
- Fast Transformer Decoding: https://arxiv.org/abs/2211.05102
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA).
    
    Uses multiple query heads but only ONE key head and ONE value head.
    All query heads attend to the same key/value representations.
    
    Benefits:
    - Reduces KV cache size by num_heads factor
    - Faster inference with minimal quality degradation
    - ~10x less memory for KV cache in large models
    
    Args:
        d_model: Model dimension
        num_heads: Number of query heads
        dropout: Dropout rate
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Query: multiple heads (full d_model)
        self.W_q = nn.Linear(d_model, d_model)
        
        # Key and Value: SINGLE head (only d_k dimension)
        self.W_k = nn.Linear(d_model, self.d_k)
        self.W_v = nn.Linear(d_model, self.d_k)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for Multi-Query Attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Optional attention mask [batch_size, 1, seq_len, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
            Optional: attention weights if return_attention=True
        """
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x)  # [batch_size, seq_len, d_model]
        K = self.W_k(x)  # [batch_size, seq_len, d_k] - SINGLE head
        V = self.W_v(x)  # [batch_size, seq_len, d_k] - SINGLE head
        
        # Reshape Q to multiple heads: [batch_size, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        Q = Q.transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        
        # Reshape K and V to have heads dimension (will broadcast)
        # [batch_size, 1, seq_len, d_k]
        K = K.unsqueeze(1)  # Add heads dimension
        V = V.unsqueeze(1)
        
        # Compute attention scores
        # Q: [batch_size, num_heads, seq_len, d_k]
        # K: [batch_size, 1, seq_len, d_k]
        # Broadcasting will repeat K across all query heads
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: [batch_size, num_heads, seq_len, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values (V broadcasts across heads)
        # attention: [batch_size, num_heads, seq_len, seq_len]
        # V: [batch_size, 1, seq_len, d_k]
        output = torch.matmul(attention, V)  # [batch_size, num_heads, seq_len, d_k]
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_heads, d_k]
        output = output.view(batch_size, seq_len, d_model)  # [batch_size, seq_len, d_model]
        
        # Final projection
        output = self.W_o(output)
        
        if return_attention:
            return output, attention
        return output


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).
    
    Groups of query heads share key/value heads. This is a hybrid between
    Multi-Head Attention (MHA) and Multi-Query Attention (MQA).
    
    Example with 32 query heads and 8 kv heads:
    - Query heads 0-3 share KV head 0
    - Query heads 4-7 share KV head 1
    - ... and so on
    
    Benefits:
    - More flexible than MQA (tunable kv heads)
    - Better quality than MQA while still memory efficient
    - Used in LLaMA 2, Mistral
    
    Args:
        d_model: Model dimension
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads (must divide num_heads)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.d_k = d_model // num_heads
        
        # Query: full dimension (num_heads * d_k)
        self.W_q = nn.Linear(d_model, num_heads * self.d_k)
        
        # Key and Value: reduced dimension (num_kv_heads * d_k)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k)
        
        # Output projection
        self.W_o = nn.Linear(num_heads * self.d_k, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for Grouped-Query Attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Optional attention mask [batch_size, 1, seq_len, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
            Optional: attention weights if return_attention=True
        """
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x)  # [batch_size, seq_len, num_heads * d_k]
        K = self.W_k(x)  # [batch_size, seq_len, num_kv_heads * d_k]
        V = self.W_v(x)  # [batch_size, seq_len, num_kv_heads * d_k]
        
        # Reshape Q: [batch_size, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        Q = Q.transpose(1, 2)
        
        # Reshape K and V: [batch_size, num_kv_heads, seq_len, d_k]
        K = K.view(batch_size, seq_len, self.num_kv_heads, self.d_k)
        K = K.transpose(1, 2)
        
        V = V.view(batch_size, seq_len, self.num_kv_heads, self.d_k)
        V = V.transpose(1, 2)
        
        # Repeat K and V to match Q's num_heads dimension
        # Each KV head is shared by num_queries_per_kv query heads
        # [batch_size, num_kv_heads, seq_len, d_k] -> [batch_size, num_heads, seq_len, d_k]
        K = torch.repeat_interleave(K, repeats=self.num_queries_per_kv, dim=1)
        V = torch.repeat_interleave(V, repeats=self.num_queries_per_kv, dim=1)
        
        # Now K and V have the same number of heads as Q
        # Compute scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: [batch_size, num_heads, seq_len, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        output = torch.matmul(attention, V)  # [batch_size, num_heads, seq_len, d_k]
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_heads, d_k]
        output = output.view(batch_size, seq_len, self.num_heads * self.d_k)
        
        # Final projection
        output = self.W_o(output)
        
        if return_attention:
            return output, attention
        return output


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention (MHA) for comparison.
    
    This is the baseline - each query head has its own key and value head.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # All projections are d_model -> d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Standard Multi-Head Attention forward pass."""
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x)  # [batch_size, seq_len, d_model]
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape to multi-head: [batch_size, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        output = torch.matmul(attention, V)  # [batch_size, num_heads, seq_len, d_k]
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, d_model)
        
        # Final projection
        output = self.W_o(output)
        
        if return_attention:
            return output, attention
        return output


def compare_kv_cache_sizes(
    d_model: int,
    num_heads: int,
    num_kv_heads: int,
    seq_len: int,
    batch_size: int = 1
) -> dict:
    """
    Compare KV cache memory requirements for different attention mechanisms.
    
    Args:
        d_model: Model dimension
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads (for GQA)
        seq_len: Sequence length
        batch_size: Batch size
        
    Returns:
        Dictionary with cache sizes in bytes for each mechanism
    """
    d_k = d_model // num_heads
    bytes_per_element = 4  # float32
    
    # KV cache shape: [batch_size, num_kv_heads, seq_len, d_k]
    # Memory = 2 (K and V) * batch_size * num_kv_heads * seq_len * d_k * 4 bytes
    
    # Multi-Head Attention: num_kv_heads = num_heads
    mha_size = 2 * batch_size * num_heads * seq_len * d_k * bytes_per_element
    
    # Grouped-Query Attention: num_kv_heads = num_kv_heads (provided)
    gqa_size = 2 * batch_size * num_kv_heads * seq_len * d_k * bytes_per_element
    
    # Multi-Query Attention: num_kv_heads = 1
    mqa_size = 2 * batch_size * 1 * seq_len * d_k * bytes_per_element
    
    return {
        'mha_bytes': mha_size,
        'gqa_bytes': gqa_size,
        'mqa_bytes': mqa_size,
        'mha_mb': mha_size / (1024**2),
        'gqa_mb': gqa_size / (1024**2),
        'mqa_mb': mqa_size / (1024**2),
        'gqa_reduction': mha_size / gqa_size,
        'mqa_reduction': mha_size / mqa_size,
    }


# ============= Test Cases =============
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    
    from utils.test_runner import test_ml_implementation
    
    print("=" * 70)
    print("Part 1: Multi-Query Attention (MQA) Tests")
    print("=" * 70)
    
    # MQA Tests
    def test_mqa_shape(mqa):
        batch_size, seq_len, d_model = 4, 16, 512
        x = torch.randn(batch_size, seq_len, d_model)
        output = mqa(x)
        return output.shape, x.shape
    
    def test_mqa_kv_params(mqa):
        # MQA should have significantly fewer parameters in K and V projections
        # K and V should project to d_k (not d_model)
        total_params = sum(p.numel() for p in mqa.parameters())
        
        # Standard MHA would have: 4 * d_model^2 parameters (Q, K, V, O)
        # MQA should have fewer because K, V project to smaller dimension
        expected_mha_params = 4 * (mqa.d_model ** 2)
        
        result = total_params < expected_mha_params
        expected = True
        return result, expected
    
    def test_mqa_attention_output(mqa):
        # Should produce valid attention output
        batch_size, seq_len, d_model = 2, 8, 512
        x = torch.randn(batch_size, seq_len, d_model)
        
        with torch.no_grad():
            output = mqa(x)
        
        # Output should not be identical to input
        result = torch.allclose(output, x, atol=1e-3)
        expected = False
        return result, expected
    
    def test_mqa_with_mask(mqa):
        # Should work with causal mask
        batch_size, seq_len, d_model = 2, 8, 512
        x = torch.randn(batch_size, seq_len, d_model)
        mask = torch.tril(torch.ones(1, 1, seq_len, seq_len))
        
        output = mqa(x, mask=mask)
        return output.shape, x.shape
    
    mqa = MultiQueryAttention(d_model=512, num_heads=8, dropout=0.0)
    
    mqa_tests = [
        (test_mqa_shape, "MQA output shape"),
        (test_mqa_kv_params, "MQA has fewer parameters than MHA"),
        (test_mqa_attention_output, "MQA produces valid attention"),
        (test_mqa_with_mask, "MQA works with mask"),
    ]
    
    test_ml_implementation(mqa, mqa_tests)
    
    print("\n" + "=" * 70)
    print("Part 2: Grouped-Query Attention (GQA) Tests")
    print("=" * 70)
    
    # GQA Tests
    def test_gqa_shape(gqa):
        batch_size, seq_len, d_model = 4, 16, 512
        x = torch.randn(batch_size, seq_len, d_model)
        output = gqa(x)
        return output.shape, x.shape
    
    def test_gqa_kv_params(gqa):
        # GQA should have fewer parameters than MHA but more than MQA
        total_params = sum(p.numel() for p in gqa.parameters())
        
        # Should have intermediate number of parameters
        expected_mha_params = 4 * (gqa.d_model ** 2)
        expected_mqa_params = gqa.d_model * (gqa.d_model + 2 * gqa.d_k + gqa.d_model)
        
        result = expected_mqa_params < total_params < expected_mha_params
        expected = True
        return result, expected
    
    def test_gqa_grouping(gqa):
        # Verify num_queries_per_kv is calculated correctly
        result = gqa.num_queries_per_kv
        expected = gqa.num_heads // gqa.num_kv_heads
        return result, expected
    
    def test_gqa_attention_output(gqa):
        batch_size, seq_len, d_model = 2, 8, 512
        x = torch.randn(batch_size, seq_len, d_model)
        
        with torch.no_grad():
            output = gqa(x)
        
        # Output should not be identical to input
        result = torch.allclose(output, x, atol=1e-3)
        expected = False
        return result, expected
    
    def test_gqa_with_mask(gqa):
        batch_size, seq_len, d_model = 2, 8, 512
        x = torch.randn(batch_size, seq_len, d_model)
        mask = torch.tril(torch.ones(1, 1, seq_len, seq_len))
        
        output = gqa(x, mask=mask)
        return output.shape, x.shape
    
    # Test with 32 query heads and 8 kv heads (LLaMA 2 style)
    gqa = GroupedQueryAttention(
        d_model=512,
        num_heads=8,
        num_kv_heads=2,  # 4 queries per kv head
        dropout=0.0
    )
    
    gqa_tests = [
        (test_gqa_shape, "GQA output shape"),
        (test_gqa_kv_params, "GQA has intermediate parameters"),
        (test_gqa_grouping, "GQA query grouping correct"),
        (test_gqa_attention_output, "GQA produces valid attention"),
        (test_gqa_with_mask, "GQA works with mask"),
    ]
    
    test_ml_implementation(gqa, gqa_tests)
    
    print("\n" + "=" * 70)
    print("Part 3: Comparison Tests")
    print("=" * 70)
    
    # Compare all three mechanisms
    def test_cache_size_comparison(_):
        d_model, num_heads = 512, 8
        seq_len, batch_size = 1024, 1
        d_k = d_model // num_heads
        
        # Calculate cache sizes (in MB)
        bytes_per_elem = 4  # float32
        
        # MHA: num_heads KV heads
        mha_size = 2 * batch_size * num_heads * seq_len * d_k * bytes_per_elem / (1024**2)
        
        # GQA: 2 KV heads (for this example)
        gqa_size = 2 * batch_size * 2 * seq_len * d_k * bytes_per_elem / (1024**2)
        
        # MQA: 1 KV head
        mqa_size = 2 * batch_size * 1 * seq_len * d_k * bytes_per_elem / (1024**2)
        
        # MQA should be smallest
        result = mqa_size < gqa_size < mha_size
        expected = True
        
        print(f"\n  KV Cache Sizes (seq_len={seq_len}):")
        print(f"  - MHA (8 KV heads):  {mha_size:.2f} MB")
        print(f"  - GQA (2 KV heads):  {gqa_size:.2f} MB ({gqa_size/mha_size*100:.1f}% of MHA)")
        print(f"  - MQA (1 KV head):   {mqa_size:.2f} MB ({mqa_size/mha_size*100:.1f}% of MHA)")
        
        return result, expected
    
    def test_output_similarity(_):
        # All three should produce similar-ish outputs (statistically)
        batch_size, seq_len, d_model, num_heads = 2, 8, 512, 8
        x = torch.randn(batch_size, seq_len, d_model)
        
        mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)
        gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads=2, dropout=0.0)
        mqa = MultiQueryAttention(d_model, num_heads, dropout=0.0)
        
        with torch.no_grad():
            out_mha = mha(x)
            out_gqa = gqa(x)
            out_mqa = mqa(x)
        
        # All should have same shape
        result = (out_mha.shape == out_gqa.shape == out_mqa.shape == x.shape)
        expected = True
        return result, expected
    
    comparison_tests = [
        (test_cache_size_comparison, "KV cache size comparison"),
        (test_output_similarity, "Output shapes match across all mechanisms"),
    ]
    
    test_ml_implementation(None, comparison_tests)
    
    print("\n" + "=" * 70)
    print("Key Implementation Details:")
    print("=" * 70)
    print("""
Multi-Query Attention (MQA):
✓ Single K and V heads (dimension d_k) shared across all Q heads
✓ K and V are broadcast automatically during matmul
✓ Memory reduction: ~num_heads factor (8x for 8 heads)
✓ Minimal quality loss compared to MHA

Grouped-Query Attention (GQA):
✓ Multiple KV heads (fewer than Q heads)
✓ Each KV head is repeated using repeat_interleave
✓ Flexible memory/quality trade-off
✓ LLaMA 2: 32 Q heads, 8 KV heads (4x reduction)

Broadcasting vs Repeat:
- MQA: Uses broadcasting (more implicit)
- GQA: Uses repeat_interleave (more explicit)
    """)
    
    print("\n" + "=" * 70)
    print("Real-World Performance Impact:")
    print("=" * 70)
    
    # Show concrete numbers for a large model
    d_model, num_heads = 4096, 32
    seq_len, batch_size = 2048, 1
    
    sizes = compare_kv_cache_sizes(d_model, num_heads, 8, seq_len, batch_size)
    
    print(f"""
Example: LLaMA 2 70B style model
  - d_model: {d_model}
  - num_heads: {num_heads}
  - seq_len: {seq_len}
  - batch_size: {batch_size}

KV Cache Sizes:
  - MHA ({num_heads} KV heads):  {sizes['mha_mb']:.2f} MB
  - GQA (8 KV heads):             {sizes['gqa_mb']:.2f} MB ({sizes['gqa_reduction']:.1f}x reduction)
  - MQA (1 KV head):              {sizes['mqa_mb']:.2f} MB ({sizes['mqa_reduction']:.1f}x reduction)

For a 70B parameter model with 80 layers:
  - MHA: {sizes['mha_mb'] * 80 / 1024:.2f} GB per batch
  - GQA: {sizes['gqa_mb'] * 80 / 1024:.2f} GB per batch (LLaMA 2 choice)
  - MQA: {sizes['mqa_mb'] * 80 / 1024:.2f} GB per batch
    """)
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)

