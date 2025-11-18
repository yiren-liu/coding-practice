"""
ML Implementation: Transformer Components (LayerNorm, RMSNorm, RoPE, Transformer Block)

Description:
Implement four key transformer components:
1. LayerNorm (Layer Normalization) - standard normalization technique
2. RMSNorm (Root Mean Square Layer Normalization) - simpler alternative to LayerNorm
3. Rotary Position Embeddings (RoPE) - position encoding in rotation space
4. Complete Transformer Block with pre-norm architecture

These are modern alternatives used in models like LLaMA, GPT-NeoX, and PaLM.

References:
- LayerNorm: https://arxiv.org/abs/1607.06450
- RMSNorm: https://arxiv.org/abs/1910.07467
- RoPE: https://arxiv.org/abs/2104.09864
- Pre-norm Transformers: https://arxiv.org/abs/2002.04745
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class LayerNorm(nn.Module):
    """
    Layer Normalization (standard version).
    
    Normalizes inputs to have zero mean and unit variance, then applies
    learnable affine transformation. Used in the original Transformer paper.
    
    Formula: y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
    
    Args:
        d_model: Dimension of the model
        eps: Small constant for numerical stability (default: 1e-6)
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # TODO: Initialize learnable parameters
        # gamma (scale) - initialized to ones
        # beta (shift) - initialized to zeros
        # Hint: Use nn.Parameter
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LayerNorm.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Normalized tensor of same shape
        """
        # TODO: Implement LayerNorm
        # 1. Compute mean and variance along last dimension
        # 2. Normalize: (x - mean) / sqrt(var + eps)
        # 3. Apply affine transformation: result * gamma + beta

        # TODO: which dim(s) are we doing the norm for?
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        x_normed = ((x - mean) / torch.sqrt(var - self.eps)) * self.gamma + self.beta
    
        return x_normed

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Simpler than LayerNorm - only does RMS normalization without mean centering.
    Used in LLaMA and other modern models for efficiency.
    
    Formula: y = x / RMS(x) * gamma
    where RMS(x) = sqrt(mean(x^2) + eps)
    
    Args:
        d_model: Dimension of the model
        eps: Small constant for numerical stability (default: 1e-6)
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # TODO: Initialize learnable scale parameter (gamma)
        # Hint: Use nn.Parameter with torch.ones
        self.gamma = nn.Parameter(torch.ones(d_model))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Normalized tensor of same shape
        """
        # TODO: Implement RMSNorm
        # 1. Compute RMS: sqrt(mean(x^2) + eps)
        # 2. Normalize: x / RMS
        # 3. Scale by learnable parameter

        rms = torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + self.eps)
        y = x / rms * self.gamma

        return y


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    
    Encodes position information by rotating queries and keys in complex space.
    This provides relative position information without explicit position embeddings.
    
    Used in: GPT-NeoX, PaLM, LLaMA, and many others.
    
    Args:
        d_model: Dimension per attention head (d_k)
        max_seq_len: Maximum sequence length to precompute (default: 2048)
        base: Base for frequency computation (default: 10000)
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        
        # TODO: Precompute rotation matrices
        # 1. Calculate frequencies: theta_i = base^(-2i/d) for i in [0, d/2)
        # 2. Create position indices: [0, 1, 2, ..., max_seq_len-1]
        # 3. Compute cos and sin for all positions and frequencies
        # 4. Register as buffers (not parameters)
        
        # Hint: Final shapes should be [1, max_seq_len, 1, d_model]
        # for easy broadcasting with [batch, seq_len, num_heads, d_k]

        self.theta = self.base
        

        pass
    
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Apply rotary position embeddings.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_heads, d_k]
            seq_len: Current sequence length (may be less than max_seq_len)
            
        Returns:
            Tensor with rotary embeddings applied, same shape as input
        """
        # TODO: Apply rotation
        # 1. Split x into first half and second half along d_k dimension
        # 2. Apply rotation: 
        #    x1_rot = x1 * cos - x2 * sin
        #    x2_rot = x1 * sin + x2 * cos
        # 3. Concatenate back together
        pass
    
    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Helper function to rotate half the dimensions.
        Splits x and swaps with negation.
        """
        # TODO: Implement rotation helper
        # Split x in half, swap them, negate the first half
        pass


class TransformerBlock(nn.Module):
    """
    Complete Transformer Block with pre-norm architecture.
    
    Structure (pre-norm):
    - x = x + MultiHeadAttention(RMSNorm(x))
    - x = x + FeedForward(RMSNorm(x))
    
    This is more stable than post-norm and used in modern transformers.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension (typically 4 * d_model)
        dropout: Dropout rate
        use_rope: Whether to use rotary position embeddings
        max_seq_len: Maximum sequence length for RoPE
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_rope: bool = False,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        
        # TODO: Initialize components:
        # 1. Two RMSNorm layers (for attention and feedforward)
        # 2. Multi-head attention (you can use a simplified version or import from previous problem)
        # 3. Feed-forward network (Linear -> Activation -> Linear)
        # 4. Dropout layers
        # 5. Optional: Rotary embeddings if use_rope=True
        
        # Feed-forward architecture:
        # - First linear: d_model -> d_ff
        # - Activation: GELU (or SwiGLU for extra credit!)
        # - Second linear: d_ff -> d_model
        pass
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor of same shape as input
        """
        # TODO: Implement pre-norm transformer block:
        # 1. Attention sub-layer: x = x + dropout(attention(norm(x)))
        # 2. Feed-forward sub-layer: x = x + dropout(feedforward(norm(x)))
        pass
    
    def _attention_block(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Helper for attention sub-layer with optional RoPE."""
        # TODO: Implement attention with optional RoPE
        pass
    
    def _feedforward_block(self, x: torch.Tensor) -> torch.Tensor:
        """Helper for feed-forward sub-layer."""
        # TODO: Implement feed-forward network
        pass


# ============= Test Cases =============
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    
    from utils.test_runner import test_ml_implementation
    
    print("=" * 60)
    print("Part 1: LayerNorm Tests")
    print("=" * 60)
    
    # LayerNorm Tests
    def test_layernorm_shape(norm):
        batch_size, seq_len, d_model = 4, 16, 512
        x = torch.randn(batch_size, seq_len, d_model)
        output = norm(x)
        return output.shape, x.shape
    
    def test_layernorm_zero_mean(norm):
        # After normalization, mean should be close to 0
        batch_size, seq_len, d_model = 2, 10, 512
        x = torch.randn(batch_size, seq_len, d_model) * 10 + 5  # Non-zero mean
        output = norm(x)
        
        result = output.mean(dim=-1, keepdim=True)
        expected = torch.zeros(batch_size, seq_len, 1)
        
        return result, expected
    
    def test_layernorm_unit_variance(norm):
        # After normalization, variance should be close to 1
        batch_size, seq_len, d_model = 2, 10, 512
        x = torch.randn(batch_size, seq_len, d_model) * 10
        output = norm(x)
        
        result = output.var(dim=-1, keepdim=True, unbiased=False)
        expected = torch.ones(batch_size, seq_len, 1)
        
        return result, expected
    
    def test_layernorm_learnable_params(norm):
        # Should have learnable gamma and beta
        has_weight = hasattr(norm, 'weight') or hasattr(norm, 'gamma')
        has_bias = hasattr(norm, 'bias') or hasattr(norm, 'beta')
        result = has_weight and has_bias
        expected = True
        return result, expected
    
    layernorm = LayerNorm(d_model=512)
    
    layernorm_tests = [
        (test_layernorm_shape, "LayerNorm output shape"),
        (test_layernorm_zero_mean, "LayerNorm mean ≈ 0"),
        (test_layernorm_unit_variance, "LayerNorm variance ≈ 1"),
        (test_layernorm_learnable_params, "LayerNorm has gamma and beta"),
    ]
    
    test_ml_implementation(layernorm, layernorm_tests)
    
    print("\n" + "=" * 60)
    print("Part 2: RMSNorm Tests")
    print("=" * 60)
    
    # RMSNorm Tests
    def test_rmsnorm_shape(norm):
        batch_size, seq_len, d_model = 4, 16, 512
        x = torch.randn(batch_size, seq_len, d_model)
        output = norm(x)
        return output.shape, x.shape
    
    def test_rmsnorm_mean_square(norm):
        # After normalization, mean of squares should be close to 1
        batch_size, seq_len, d_model = 2, 10, 512
        x = torch.randn(batch_size, seq_len, d_model) * 10  # Large values
        output = norm(x)
        
        result = (output ** 2).mean(dim=-1, keepdim=True)
        expected = torch.ones(batch_size, seq_len, 1)
        
        return result, expected
    
    def test_rmsnorm_learnable_scale(norm):
        # Should have learnable parameter
        result = hasattr(norm, 'weight') or hasattr(norm, 'scale') or hasattr(norm, 'gamma')
        expected = True
        return result, expected
    
    rmsnorm = RMSNorm(d_model=512)
    
    rmsnorm_tests = [
        (test_rmsnorm_shape, "RMSNorm output shape"),
        (test_rmsnorm_mean_square, "RMSNorm mean square ≈ 1"),
        (test_rmsnorm_learnable_scale, "RMSNorm has learnable scale"),
    ]
    
    test_ml_implementation(rmsnorm, rmsnorm_tests)
    
    print("\n" + "=" * 60)
    print("Part 3: Rotary Position Embeddings Tests")
    print("=" * 60)
    
    # RoPE Tests
    def test_rope_shape(rope):
        batch_size, seq_len, num_heads, d_k = 2, 16, 8, 64
        x = torch.randn(batch_size, seq_len, num_heads, d_k)
        output = rope(x, seq_len)
        return output.shape, x.shape
    
    def test_rope_relative_position(rope):
        # RoPE should encode relative positions
        # Two identical inputs at different positions should produce different outputs
        batch_size, num_heads, d_k = 1, 8, 64
        
        # Same input at different positions
        x1 = torch.ones(batch_size, 2, num_heads, d_k)
        output = rope(x1, 2)
        
        # First and second position should be different
        pos_0 = output[0, 0, 0, :]
        pos_1 = output[0, 1, 0, :]
        
        result = torch.allclose(pos_0, pos_1, atol=1e-5)
        expected = False  # Should be different
        
        return result, expected
    
    def test_rope_max_seq_len(rope):
        # Should work for sequences up to max_seq_len
        batch_size, num_heads, d_k = 1, 8, 64
        seq_len = min(rope.max_seq_len, 100)
        
        x = torch.randn(batch_size, seq_len, num_heads, d_k)
        output = rope(x, seq_len)
        
        return output.shape, x.shape
    
    rope = RotaryPositionEmbedding(d_model=64, max_seq_len=2048)
    
    rope_tests = [
        (test_rope_shape, "RoPE output shape"),
        (test_rope_relative_position, "RoPE encodes position"),
        (test_rope_max_seq_len, "RoPE handles long sequences"),
    ]
    
    test_ml_implementation(rope, rope_tests)
    
    print("\n" + "=" * 60)
    print("Part 4: Transformer Block Tests")
    print("=" * 60)
    
    # Transformer Block Tests
    def test_transformer_shape(block):
        batch_size, seq_len, d_model = 4, 16, 512
        x = torch.randn(batch_size, seq_len, d_model)
        output = block(x)
        return output.shape, x.shape
    
    def test_transformer_residual(block):
        # Output should be different from input (not identity)
        batch_size, seq_len, d_model = 2, 8, 512
        x = torch.randn(batch_size, seq_len, d_model)
        
        with torch.no_grad():
            output = block(x)
        
        # Should not be identical (residual connections add to modified values)
        result = torch.allclose(output, x, atol=1e-3)
        expected = False
        
        return result, expected
    
    def test_transformer_with_mask(block):
        # Should work with causal mask
        batch_size, seq_len, d_model = 2, 8, 512
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Create causal mask
        mask = torch.tril(torch.ones(1, 1, seq_len, seq_len))
        
        output = block(x, mask=mask)
        
        return output.shape, x.shape
    
    def test_transformer_different_seq_lens(block):
        # Should generalize to different sequence lengths
        batch_size, d_model = 2, 512
        
        for seq_len in [4, 8, 16]:
            x = torch.randn(batch_size, seq_len, d_model)
            output = block(x)
            if output.shape != x.shape:
                return output.shape, x.shape
        
        return True, True
    
    # Test both with and without RoPE
    transformer_no_rope = TransformerBlock(
        d_model=512,
        num_heads=8,
        d_ff=2048,
        dropout=0.0,
        use_rope=False
    )
    
    transformer_with_rope = TransformerBlock(
        d_model=512,
        num_heads=8,
        d_ff=2048,
        dropout=0.0,
        use_rope=True,
        max_seq_len=2048
    )
    
    print("\nTesting WITHOUT RoPE:")
    transformer_tests = [
        (test_transformer_shape, "Transformer output shape"),
        (test_transformer_residual, "Transformer applies transformations"),
        (test_transformer_with_mask, "Transformer works with mask"),
        (test_transformer_different_seq_lens, "Transformer handles variable lengths"),
    ]
    
    test_ml_implementation(transformer_no_rope, transformer_tests)
    
    print("\nTesting WITH RoPE:")
    test_ml_implementation(transformer_with_rope, transformer_tests)
    
    print("\n" + "=" * 60)
    print("Implementation Tips:")
    print("=" * 60)
    print("""
LayerNorm:
- mean = mean(x, dim=-1, keepdim=True)
- var = var(x, dim=-1, keepdim=True, unbiased=False)
- output = (x - mean) / sqrt(var + eps) * gamma + beta
- gamma and beta are learnable: nn.Parameter(torch.ones/zeros(d_model))

RMSNorm:
- RMS = sqrt(mean(x^2) + eps)
- output = (x / RMS) * gamma
- gamma is learnable: nn.Parameter(torch.ones(d_model))

Rotary Position Embeddings:
- theta[i] = base^(-2i/d) for i in range(d/2)
- freqs = position * theta
- Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]

Transformer Block (Pre-Norm):
- x = x + dropout(attention(norm1(x)))
- x = x + dropout(ffn(norm2(x)))
- FFN: Linear(d_model -> d_ff) -> GELU -> Linear(d_ff -> d_model)

Key Differences from Post-Norm:
- Pre-norm: Normalize BEFORE attention/FFN (more stable)
- Post-norm: Normalize AFTER (original transformer paper)
    """)

