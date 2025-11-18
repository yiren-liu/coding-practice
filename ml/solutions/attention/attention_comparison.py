"""
Attention Mechanisms Comparison: MHA vs MQA vs GQA

This script demonstrates the differences between:
- Multi-Head Attention (MHA)
- Multi-Query Attention (MQA) 
- Grouped Query Attention (GQA)

Key differences:
1. MHA: Each head has separate Q, K, V projections
2. MQA: All heads share single K, V projections
3. GQA: Query heads are grouped, each group shares K, V projections
"""

import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from multi_query_attention import MultiQueryAttention
from grouped_query_attention import GroupedQueryAttention


def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def analyze_kv_cache_size(model, batch_size, seq_len):
    """
    Calculate KV cache size for inference.
    
    During autoregressive generation, we cache K and V for all previous tokens
    to avoid recomputation. The cache size is a key efficiency metric.
    """
    d_k = model.d_k
    
    if hasattr(model, 'num_kv_heads'):
        # GQA
        num_kv_heads = model.num_kv_heads
    elif hasattr(model, 'W_k') and model.W_k.out_features == d_k:
        # MQA (single K, V projection)
        num_kv_heads = 1
    else:
        # MHA (full K, V projections)
        num_kv_heads = model.num_heads
    
    # KV cache stores both K and V
    # Each has shape [batch_size, num_kv_heads, seq_len, d_k]
    kv_cache_size = 2 * batch_size * num_kv_heads * seq_len * d_k * 4  # 4 bytes per float32
    
    return kv_cache_size, num_kv_heads


def main():
    print("=" * 80)
    print("Attention Mechanisms Comparison")
    print("=" * 80)
    
    # Configuration
    d_model = 512
    num_heads = 8
    batch_size = 4
    seq_len = 1024
    
    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_k (per head): {d_model // num_heads}")
    print(f"  batch_size: {batch_size}")
    print(f"  sequence_length: {seq_len}")
    
    # Create models
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    mqa = MultiQueryAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    gqa_2 = GroupedQueryAttention(d_model=d_model, num_heads=num_heads, num_kv_heads=2, dropout=0.0)
    gqa_4 = GroupedQueryAttention(d_model=d_model, num_heads=num_heads, num_kv_heads=4, dropout=0.0)
    
    models = [
        ("Multi-Head Attention (MHA)", mha),
        ("Multi-Query Attention (MQA)", mqa),
        ("Grouped Query Attention (GQA, 2 groups)", gqa_2),
        ("Grouped Query Attention (GQA, 4 groups)", gqa_4),
    ]
    
    print("\n" + "=" * 80)
    print("Model Statistics")
    print("=" * 80)
    
    results = []
    
    for name, model in models:
        param_count = count_parameters(model)
        kv_cache_size, num_kv_heads = analyze_kv_cache_size(model, batch_size, seq_len)
        kv_cache_mb = kv_cache_size / (1024 * 1024)
        
        results.append({
            'name': name,
            'params': param_count,
            'num_kv_heads': num_kv_heads,
            'kv_cache_mb': kv_cache_mb
        })
    
    # Print table header
    print(f"\n{'Model':<45} {'Parameters':<15} {'KV Heads':<12} {'KV Cache (MB)':<15}")
    print("-" * 87)
    
    # Print results
    for result in results:
        print(f"{result['name']:<45} {result['params']:<15,} {result['num_kv_heads']:<12} {result['kv_cache_mb']:<15.2f}")
    
    # Calculate improvements
    print("\n" + "=" * 80)
    print("Efficiency Gains (compared to MHA)")
    print("=" * 80)
    
    mha_params = results[0]['params']
    mha_cache = results[0]['kv_cache_mb']
    
    for i, result in enumerate(results[1:], 1):
        param_reduction = (1 - result['params'] / mha_params) * 100
        cache_reduction = (1 - result['kv_cache_mb'] / mha_cache) * 100
        
        print(f"\n{result['name']}:")
        print(f"  Parameter reduction: {param_reduction:.1f}%")
        print(f"  KV cache reduction: {cache_reduction:.1f}%")
    
    # Test forward pass
    print("\n" + "=" * 80)
    print("Forward Pass Test")
    print("=" * 80)
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nInput shape: {list(x.shape)}")
    
    for name, model in models:
        with torch.no_grad():
            output = model(x)
        print(f"{name:<45} -> Output shape: {list(output.shape)}")
    
    # Attention pattern visualization info
    print("\n" + "=" * 80)
    print("Key Differences")
    print("=" * 80)
    
    print("""
1. Multi-Head Attention (MHA):
   - Each head has its own Q, K, V projections
   - Highest quality but most memory intensive
   - Used in: Original Transformer, BERT, GPT-2
   
2. Multi-Query Attention (MQA):
   - All heads share a single K, V projection
   - Significant memory savings for KV cache (8x reduction with 8 heads)
   - Minimal quality degradation
   - Used in: PaLM, Falcon, StarCoder
   
3. Grouped Query Attention (GQA):
   - Interpolates between MHA and MQA
   - Query heads divided into groups, each group shares K, V
   - Balances quality and efficiency
   - Used in: Llama 2, Mistral, Mixtral
   
Choice Guidelines:
- Use MHA when: Maximum quality is needed, memory is not a constraint
- Use MQA when: Inference speed/memory is critical, slight quality loss acceptable
- Use GQA when: Want to balance quality and efficiency (recommended for most cases)
    """)
    
    print("\n" + "=" * 80)
    print("References")
    print("=" * 80)
    print("""
- MHA: "Attention Is All You Need" (Vaswani et al., 2017)
  https://arxiv.org/abs/1706.03762

- MQA: "Fast Transformer Decoding: One Write-Head is All You Need" (Shazeer, 2019)
  https://arxiv.org/abs/1911.02150

- GQA: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
  https://arxiv.org/abs/2305.13245
    """)


if __name__ == "__main__":
    main()

