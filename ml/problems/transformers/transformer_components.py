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

from this import d
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
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
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
        # TODO: Initialize learnable scale parameter (gamma)
        # Hint: Use nn.Parameter with torch.ones
        self.weight = nn.Parameter(torch.ones(d_model))
    
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
        
        # TODO: Precompute rotation matrices
        # 1. Calculate frequencies: theta_i = base^(-2i/d) for i in [0, d/2)
        # 2. Create position indices: [0, 1, 2, ..., max_seq_len-1]
        # 3. Compute cos and sin for all positions and frequencies
        # 4. Register as buffers (not parameters)
        
        # Hint: Final shapes should be [1, max_seq_len, 1, d_model]
        # for easy broadcasting with [batch, seq_len, num_heads, d_k]
        self.theta = 1 / self.base ** (torch.arange(0, self.d_model, 2) / self.d_model)
        self.pos = torch.arange(0, self.max_seq_len)
        self.freq = torch.outer(self.pos, self.theta) # [seq_len, d/2]

        sin = torch.sin(self.freq).unsqueeze(0).unsqueeze(2) # [1, seq_len, 1, d/2]
        cos = torch.cos(self.freq).unsqueeze(0).unsqueeze(2) # [1, seq_len, 1, d/2]

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
        # TODO: Apply rotation
        # 1. Split x into first half and second half along d_k dimension
        # 2. Apply rotation: 
        #    x1_rot = x1 * cos - x2 * sin
        #    x2_rot = x1 * sin + x2 * cos
        # 3. Concatenate back together
        _, seq_len, _, _ = x.shape

        cos = self.cos[:, :seq_len, :, :]  # [1, seq_len, 1, d_model]
        sin = self.sin[:, :seq_len, :, :]  # [1, seq_len, 1, d_model]

        remb = x * cos + self.rotate_half(x) * sin

        return remb
    
    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Helper function to rotate half the dimensions.
        Splits x and swaps with negation.
        """
        # TODO: Implement rotation helper
        # Split x in half, swap them, negate the first half
        d_model = x.shape[-1]
        # MISTAKE: use ... instead of ::
        x1 = x[..., :d_model//2]
        x2 = x[..., d_model//2:]
        xx = torch.concat([-x2, x1], dim=-1)
        return xx

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
        bias: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Query, Key, Value projection layers
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        
        # Output projection layer
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
        Forward pass of Multi-Head Attention.
        
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
        K = self.W_k(x)  # [batch_size, seq_len, d_model]
        V = self.W_v(x)  # [batch_size, seq_len, d_model]
        
        # 2. Reshape to [batch_size, num_heads, seq_len, d_k]
        # First reshape to [batch_size, seq_len, num_heads, d_k]
        # Then transpose to [batch_size, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. Compute scaled dot-product attention
        # Calculate attention scores: Q @ K^T / sqrt(d_k)
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
        
        # 4. Reshape back to [batch_size, seq_len, d_model]
        # First transpose to [batch_size, seq_len, num_heads, d_k]
        # Then reshape/view to [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # 5. Apply output projection
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
        # This ensures that position i can only attend to positions <= i
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        # Add batch and head dimensions
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        return mask


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
        self.norm_attn = RMSNorm(self.d_model)
        self.norm_ff = RMSNorm(self.d_model)
        self.mha = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads)
        self.ff = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.d_model, self.d_model),
        )
        self.dropout = nn.Dropout(dropout)
        if use_rope:
            self.rope = RotaryPositionEmbedding(self.d_model // num_heads, max_seq_len)
        else:
            # Regular position embeddings are added at the input layer
            # (before transformer blocks), not inside the blocks
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
        # TODO: Implement pre-norm transformer block:
        # 1. Attention sub-layer: x = x + dropout(attention(norm(x)))
        # 2. Feed-forward sub-layer: x = x + dropout(feedforward(norm(x)))
        b, s, d = x.shape
        x = self.norm_attn(x)

        x = self._attention_block(x, mask)
        x += self.dropout(x)
    
        x += self.dropout(self.ff(self.norm_attn(x)))
        
        return x
    
    def _attention_block(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Helper for attention sub-layer with optional RoPE."""
        # TODO: Implement attention with optional RoPE
        b, s, d = x.shape
        if self.rope:
            x = x.view(b, s, self.num_heads, d // self.num_heads)
            x = self.rope(x, x.shape[-2]).view(b,s,d)
        return self.mha(x, mask)
    
    # def _feedforward_block(self, x: torch.Tensor) -> torch.Tensor:
    #     """Helper for feed-forward sub-layer."""
    #     # TODO: Implement feed-forward network
    #     pass


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

