"""
ML Implementation: Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)

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
from typing import Optional, Tuple


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
        
        # TODO: Initialize projection layers
        # Query: d_model -> d_model (split into num_heads)
        # Key: d_model -> d_k (SINGLE head shared by all queries)
        # Value: d_model -> d_k (SINGLE head shared by all queries)
        # Output: d_model -> d_model
        # Hint: Use nn.Linear
        pass
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
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
        # TODO: Implement Multi-Query Attention
        # 1. Project input to Q, K, V
        #    - Q: [batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, d_k]
        #    - K: [batch_size, seq_len, d_model] -> [batch_size, 1, seq_len, d_k] (single head!)
        #    - V: [batch_size, seq_len, d_model] -> [batch_size, 1, seq_len, d_k] (single head!)
        # 
        # 2. K and V will be broadcast to match Q during attention computation
        # 
        # 3. Compute scaled dot-product attention:
        #    scores = (Q @ K^T) / sqrt(d_k)
        #    attention = softmax(scores)
        #    output = attention @ V
        # 
        # 4. Reshape and project output
        
        # Hint: Broadcasting will automatically handle the single KV head
        pass


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
        
        # TODO: Initialize projection layers
        # Query: d_model -> d_model (num_heads * d_k)
        # Key: d_model -> num_kv_heads * d_k
        # Value: d_model -> num_kv_heads * d_k
        # Output: d_model -> d_model
        pass
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
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
        # TODO: Implement Grouped-Query Attention
        # 1. Project input to Q, K, V
        #    - Q: [batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, d_k]
        #    - K: [batch_size, seq_len, num_kv_heads * d_k] -> [batch_size, num_kv_heads, seq_len, d_k]
        #    - V: [batch_size, seq_len, num_kv_heads * d_k] -> [batch_size, num_kv_heads, seq_len, d_k]
        # 
        # 2. Repeat K and V to match Q's num_heads dimension
        #    Each KV head should be repeated num_queries_per_kv times
        #    Use torch.repeat_interleave along the heads dimension
        # 
        # 3. Compute scaled dot-product attention (same as standard MHA)
        # 
        # 4. Reshape and project output
        
        # Hint: torch.repeat_interleave(K, repeats=self.num_queries_per_kv, dim=1)
        pass


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
        
        # TODO: Initialize projection layers
        # All projections are d_model -> d_model
        pass
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """Standard Multi-Head Attention forward pass."""
        # TODO: Implement standard MHA for comparison
        pass


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
    # TODO: Calculate KV cache sizes
    # KV cache stores keys and values for all tokens seen so far
    # Shape for each: [batch_size, num_kv_heads, seq_len, d_k]
    # where d_k = d_model // num_heads
    
    # Memory = 2 (K and V) * batch_size * num_kv_heads * seq_len * d_k * 4 (float32 bytes)
    
    # Calculate for:
    # 1. MHA: num_kv_heads = num_heads
    # 2. GQA: num_kv_heads = num_kv_heads (provided)
    # 3. MQA: num_kv_heads = 1
    
    pass


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
    print("Implementation Tips:")
    print("=" * 70)
    print("""
Multi-Query Attention (MQA):
- Q projection: d_model -> d_model (split into num_heads)
- K projection: d_model -> d_k (SINGLE head)
- V projection: d_model -> d_k (SINGLE head)
- K and V are broadcast during attention computation
- Memory savings: ~num_heads factor for KV cache

Grouped-Query Attention (GQA):
- Q projection: d_model -> d_model (num_heads * d_k)
- K projection: d_model -> num_kv_heads * d_k
- V projection: d_model -> num_kv_heads * d_k
- Repeat K and V using: torch.repeat_interleave(K, repeats=num_queries_per_kv, dim=1)
- Memory savings: num_heads/num_kv_heads factor

Key Implementation Notes:
1. Reshape Q: [B, L, d_model] -> [B, num_heads, L, d_k]
2. Reshape K, V based on mechanism
3. Use broadcasting (MQA) or repeat_interleave (GQA)
4. Standard attention computation
5. Concat heads and project back

Practical Usage:
- MHA: Best quality, most memory
- GQA: Good quality, medium memory (LLaMA 2 default)
- MQA: Good quality, least memory (fastest inference)
    """)
    
    print("\n" + "=" * 70)
    print("Real-World Model Examples:")
    print("=" * 70)
    print("""
Models using these mechanisms:

Multi-Query Attention (MQA):
- PaLM (Google): 540B parameters
- Falcon: 40B, 180B parameters
- StarCoder: 15B parameters

Grouped-Query Attention (GQA):
- LLaMA 2: 7B (num_kv_heads=8), 13B, 70B
- Mistral 7B: num_heads=32, num_kv_heads=8
- Code LLaMA: Various sizes

Trade-offs:
1. Quality: MHA ≥ GQA > MQA (gaps are small)
2. Speed: MQA > GQA > MHA (especially for long contexts)
3. Memory: MQA < GQA < MHA (critical for deployment)
    """)

