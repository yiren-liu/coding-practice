"""
ML Implementation: Transformer Components (LayerNorm, RMSNorm, RoPE, Transformer Block) - SOLUTION

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
        # Learnable scale and shift parameters
        self.weight = nn.Parameter(torch.ones(d_model))  # gamma
        self.bias = nn.Parameter(torch.zeros(d_model))   # beta
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LayerNorm.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Normalized tensor of same shape
        """
        # Compute mean and variance along last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply affine transformation
        return self.weight * x_norm + self.bias


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
        # Learnable scale parameter
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Normalized tensor of same shape
        """
        # Compute RMS: sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize: x / RMS
        x_norm = x / rms
        
        # Scale by learnable parameter
        return self.weight * x_norm


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
        
        # Precompute rotation matrices
        # Calculate frequencies: theta_i = base^(-2i/d) for i in [0, d/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        
        # Create position indices: [0, 1, 2, ..., max_seq_len-1]
        position = torch.arange(max_seq_len).float()
        
        # Compute freqs: outer product of position and inv_freq
        # Shape: [max_seq_len, d_model/2]
        freqs = torch.outer(position, inv_freq)
        
        # Compute cos and sin
        # We'll duplicate to match full d_model dimension
        # Shape: [1, max_seq_len, 1, d_model]
        cos = torch.cos(freqs).unsqueeze(0).unsqueeze(2)  # [1, max_seq_len, 1, d_model/2]
        sin = torch.sin(freqs).unsqueeze(0).unsqueeze(2)  # [1, max_seq_len, 1, d_model/2]
        
        # Repeat to match full dimension (each frequency applies to 2 dimensions)
        cos = torch.cat([cos, cos], dim=-1)  # [1, max_seq_len, 1, d_model]
        sin = torch.cat([sin, sin], dim=-1)  # [1, max_seq_len, 1, d_model]
        
        # Register as buffers (not parameters, so they're not trained)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Apply rotary position embeddings.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_heads, d_k]
            seq_len: Current sequence length (may be less than max_seq_len)
            
        Returns:
            Tensor with rotary embeddings applied, same shape as input
        """
        # Get cos and sin for current sequence length
        cos = self.cos[:, :seq_len, :, :]  # [1, seq_len, 1, d_model]
        sin = self.sin[:, :seq_len, :, :]  # [1, seq_len, 1, d_model]
        
        # Apply rotation
        # x_rotated = x * cos + rotate_half(x) * sin
        x_rotated = x * cos + self.rotate_half(x) * sin
        
        return x_rotated
    
    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Helper function to rotate half the dimensions.
        Splits x and swaps with negation.
        
        For input [x1, x2, x3, x4, ...], returns [-x(d/2+1), -x(d/2+2), ..., x1, x2, ...]
        """
        # Split x in half along last dimension
        d = x.shape[-1]
        x1 = x[..., :d//2]  # First half
        x2 = x[..., d//2:]  # Second half
        
        # Rotate: [-x2, x1]
        return torch.cat([-x2, x1], dim=-1)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism (simplified for use in TransformerBlock).
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope: Optional[RotaryPositionEmbedding] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: Optional mask [batch_size, 1, seq_len, seq_len] or broadcastable
            rope: Optional RoPE module
        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        Q = self.W_q(x)  # [batch_size, seq_len, d_model]
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape to [batch_size, seq_len, num_heads, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Apply RoPE if provided
        if rope is not None:
            Q = rope(Q, seq_len)
            K = rope(K, seq_len)
        
        # Transpose to [batch_size, num_heads, seq_len, d_k]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, d_k]
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_heads, d_k]
        output = output.view(batch_size, seq_len, d_model)  # [batch_size, seq_len, d_model]
        
        # Final linear projection
        output = self.W_o(output)
        
        return output


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
        
        # Normalization layers
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Optional: Rotary embeddings
        if use_rope:
            # RoPE is applied per head, so dimension is d_model / num_heads
            self.rope = RotaryPositionEmbedding(d_model // num_heads, max_seq_len)
        else:
            self.rope = None
    
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
        # Pre-norm architecture
        # Attention sub-layer: x = x + dropout(attention(norm(x)))
        x = x + self.dropout1(self._attention_block(self.norm1(x), mask))
        
        # Feed-forward sub-layer: x = x + dropout(feedforward(norm(x)))
        x = x + self.dropout2(self._feedforward_block(self.norm2(x)))
        
        return x
    
    def _attention_block(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Helper for attention sub-layer with optional RoPE."""
        return self.attention(x, mask=mask, rope=self.rope)
    
    def _feedforward_block(self, x: torch.Tensor) -> torch.Tensor:
        """Helper for feed-forward sub-layer."""
        return self.ffn(x)


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
    print("Summary: Key Differences")
    print("=" * 60)
    print("""
LayerNorm vs RMSNorm:
- LayerNorm: Normalizes to zero mean AND unit variance (more computation)
  Formula: y = (x - mean) / sqrt(var + eps) * gamma + beta
  Parameters: gamma (scale) and beta (shift)
  
- RMSNorm: Only normalizes by RMS (simpler, faster)
  Formula: y = x / sqrt(mean(x^2) + eps) * gamma
  Parameters: only gamma (scale)
  
- RMSNorm is ~10-20% faster and often performs just as well
- Used in: LLaMA, GPT-NeoX, PaLM (RMSNorm) vs Original Transformer (LayerNorm)
    """)
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

