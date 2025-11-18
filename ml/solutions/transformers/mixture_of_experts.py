"""
ML Implementation: Mixture of Experts (MoE) Layer - SOLUTION

Description:
Implement a Mixture of Experts layer with top-k routing. This is used in 
large-scale models like Switch Transformer, GShard, and GPT-4 (rumored).

Key Concepts:
- Multiple "expert" feed-forward networks
- Gating mechanism that routes tokens to top-k experts
- Load balancing to distribute tokens evenly
- Sparse computation: only k out of n experts are active per token

This allows scaling model capacity without proportionally increasing computation.

References:
- Switch Transformer: https://arxiv.org/abs/2101.03961
- GShard: https://arxiv.org/abs/2006.16668
- MoE Survey: https://arxiv.org/abs/2022.08301
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class Expert(nn.Module):
    """
    A single expert - a simple feed-forward network.
    
    Args:
        d_model: Input/output dimension
        d_ff: Hidden dimension (typically 4 * d_model)
        dropout: Dropout rate
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [num_tokens, d_model]
        Returns:
            Output tensor of shape [num_tokens, d_model]
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TopKGate(nn.Module):
    """
    Gating mechanism that selects top-k experts for each token.
    
    Args:
        d_model: Model dimension
        num_experts: Total number of experts
        top_k: Number of experts to route each token to
        dropout: Dropout rate for gating
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        assert top_k <= num_experts, "top_k must be <= num_experts"
        
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Gating network - maps input to expert logits
        self.gate = nn.Linear(d_model, num_experts)
    
    def forward(
        self,
        x: torch.Tensor,
        use_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute gating weights for top-k experts.
        
        Args:
            x: Input tensor of shape [batch_size * seq_len, d_model]
            use_aux_loss: Whether to compute auxiliary load balancing loss
            
        Returns:
            gate_weights: Sparse weights of shape [num_tokens, num_experts]
            selected_experts: Indices of selected experts [num_tokens, top_k]
            aux_loss: Optional auxiliary loss for load balancing
        """
        # Compute logits for all experts
        gate_logits = self.gate(x)  # [num_tokens, num_experts]
        
        # Select top-k experts per token
        top_k_logits, selected_experts = torch.topk(gate_logits, self.top_k, dim=-1)
        # top_k_logits: [num_tokens, top_k]
        # selected_experts: [num_tokens, top_k]
        
        # Compute softmax over selected experts only
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # [num_tokens, top_k]
        
        # Create sparse weight matrix
        num_tokens = x.shape[0]
        gate_weights = torch.zeros(num_tokens, self.num_experts, device=x.device, dtype=x.dtype)
        
        # Scatter the top-k weights to their positions
        gate_weights.scatter_(1, selected_experts, top_k_weights)
        
        # Compute load balancing loss if requested
        aux_loss = None
        if use_aux_loss:
            aux_loss = self.compute_load_balancing_loss(gate_logits, selected_experts)
        
        return gate_weights, selected_experts, aux_loss
    
    def compute_load_balancing_loss(
        self,
        gate_logits: torch.Tensor,
        selected_experts: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary loss to encourage balanced expert usage.
        
        This prevents the model from using only a few experts.
        
        Args:
            gate_logits: Raw logits [num_tokens, num_experts]
            selected_experts: Selected expert indices [num_tokens, top_k]
            
        Returns:
            Scalar loss value
        """
        # Compute fraction of tokens routed to each expert (f_i)
        # Create one-hot encoding of selected experts
        num_tokens = gate_logits.shape[0]
        expert_mask = torch.zeros_like(gate_logits)  # [num_tokens, num_experts]
        expert_mask.scatter_(1, selected_experts, 1.0)
        
        # f_i = fraction of tokens assigned to expert i
        tokens_per_expert = expert_mask.sum(dim=0)  # [num_experts]
        fraction_per_expert = tokens_per_expert / (num_tokens * self.top_k)
        
        # P_i = average probability assigned to expert i across all tokens
        # Use softmax probabilities
        gate_probs = F.softmax(gate_logits, dim=-1)  # [num_tokens, num_experts]
        mean_prob_per_expert = gate_probs.mean(dim=0)  # [num_experts]
        
        # Load balancing loss: num_experts * sum(f_i * P_i)
        # This encourages uniform distribution (minimized when all f_i and P_i are equal)
        loss = self.num_experts * torch.sum(fraction_per_expert * mean_prob_per_expert)
        
        return loss


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer with top-k routing.
    
    For each token:
    1. Gate selects top-k experts and computes weights
    2. Token is processed by each selected expert
    3. Expert outputs are combined using gate weights
    
    Args:
        d_model: Model dimension
        d_ff: Expert hidden dimension
        num_experts: Total number of experts
        top_k: Number of experts to use per token
        dropout: Dropout rate
        use_aux_loss: Whether to use load balancing loss
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        use_aux_loss: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_aux_loss = use_aux_loss
        
        # Create expert networks
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])
        
        # Create gating network
        self.gate = TopKGate(d_model, num_experts, top_k, dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            output: Tensor of shape [batch_size, seq_len, d_model]
            aux_loss: Optional load balancing loss
        """
        batch_size, seq_len, d_model = x.shape
        
        # Flatten input to [batch_size * seq_len, d_model]
        x_flat = x.view(-1, d_model)  # [num_tokens, d_model]
        num_tokens = x_flat.shape[0]
        
        # Get gate weights and selected experts
        gate_weights, selected_experts, aux_loss = self.gate(x_flat, self.use_aux_loss)
        # gate_weights: [num_tokens, num_experts]
        # selected_experts: [num_tokens, top_k]
        
        # Initialize output
        output = torch.zeros_like(x_flat)  # [num_tokens, d_model]
        
        # Process each token with its selected experts (naive implementation)
        for token_idx in range(num_tokens):
            token = x_flat[token_idx]  # [d_model]
            expert_indices = selected_experts[token_idx]  # [top_k]
            expert_weights = gate_weights[token_idx, expert_indices]  # [top_k]
            
            # Process token through selected experts and combine
            token_output = self._process_token_naive(token, expert_indices, expert_weights)
            output[token_idx] = token_output
        
        # Reshape back to [batch_size, seq_len, d_model]
        output = output.view(batch_size, seq_len, d_model)
        
        return output, aux_loss
    
    def _process_token_naive(
        self,
        token: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Process a single token through its selected experts (naive implementation).
        
        Args:
            token: Single token [d_model]
            expert_indices: Selected expert indices [top_k]
            expert_weights: Weights for selected experts [top_k]
            
        Returns:
            Weighted combination of expert outputs [d_model]
        """
        # Process token through each selected expert
        expert_outputs = []
        for expert_idx in expert_indices:
            expert = self.experts[expert_idx]
            # Add batch dimension for expert processing
            expert_output = expert(token.unsqueeze(0))  # [1, d_model]
            expert_outputs.append(expert_output.squeeze(0))  # [d_model]
        
        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs)  # [top_k, d_model]
        
        # Weighted combination
        output = torch.sum(expert_weights.unsqueeze(-1) * expert_outputs, dim=0)  # [d_model]
        
        return output


# ============= Test Cases =============
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    
    from utils.test_runner import test_ml_implementation
    
    print("=" * 60)
    print("Part 1: Expert Tests")
    print("=" * 60)
    
    # Expert Tests
    def test_expert_shape(expert):
        num_tokens, d_model = 10, 512
        x = torch.randn(num_tokens, d_model)
        output = expert(x)
        return output.shape, x.shape
    
    def test_expert_different(expert):
        # Expert should transform the input
        num_tokens, d_model = 5, 512
        x = torch.randn(num_tokens, d_model)
        
        with torch.no_grad():
            output = expert(x)
        
        result = torch.allclose(output, x, atol=1e-3)
        expected = False  # Should be different
        
        return result, expected
    
    expert = Expert(d_model=512, d_ff=2048, dropout=0.0)
    
    expert_tests = [
        (test_expert_shape, "Expert output shape"),
        (test_expert_different, "Expert transforms input"),
    ]
    
    test_ml_implementation(expert, expert_tests)
    
    print("\n" + "=" * 60)
    print("Part 2: TopK Gate Tests")
    print("=" * 60)
    
    # Gate Tests
    def test_gate_output_shapes(gate):
        num_tokens, d_model = 20, 512
        x = torch.randn(num_tokens, d_model)
        
        gate_weights, selected_experts, aux_loss = gate(x)
        
        # Check shapes
        weights_correct = gate_weights.shape == (num_tokens, gate.num_experts)
        experts_correct = selected_experts.shape == (num_tokens, gate.top_k)
        
        result = weights_correct and experts_correct
        expected = True
        
        return result, expected
    
    def test_gate_sparsity(gate):
        # Each token should have exactly top_k non-zero weights
        num_tokens, d_model = 20, 512
        x = torch.randn(num_tokens, d_model)
        
        gate_weights, _, _ = gate(x)
        
        # Count non-zero weights per token
        non_zero_per_token = (gate_weights > 1e-8).sum(dim=1)
        
        result = non_zero_per_token
        expected = torch.full((num_tokens,), gate.top_k)
        
        return result, expected
    
    def test_gate_weights_sum(gate):
        # Weights should sum to 1 for each token
        num_tokens, d_model = 20, 512
        x = torch.randn(num_tokens, d_model)
        
        gate_weights, _, _ = gate(x)
        
        result = gate_weights.sum(dim=1)
        expected = torch.ones(num_tokens)
        
        return result, expected
    
    def test_gate_selected_experts_range(gate):
        # Selected experts should be valid indices
        num_tokens, d_model = 20, 512
        x = torch.randn(num_tokens, d_model)
        
        _, selected_experts, _ = gate(x)
        
        result = (selected_experts >= 0).all() and (selected_experts < gate.num_experts).all()
        expected = True
        
        return result, expected
    
    gate = TopKGate(d_model=512, num_experts=8, top_k=2, dropout=0.0)
    
    gate_tests = [
        (test_gate_output_shapes, "Gate output shapes"),
        (test_gate_sparsity, "Gate sparsity (top-k selection)"),
        (test_gate_weights_sum, "Gate weights sum to 1"),
        (test_gate_selected_experts_range, "Selected experts are valid indices"),
    ]
    
    test_ml_implementation(gate, gate_tests)
    
    print("\n" + "=" * 60)
    print("Part 3: Mixture of Experts Tests")
    print("=" * 60)
    
    # MoE Tests
    def test_moe_shape(moe):
        batch_size, seq_len, d_model = 4, 16, 512
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, aux_loss = moe(x)
        
        return output.shape, x.shape
    
    def test_moe_aux_loss(moe):
        # Should return aux loss if enabled
        batch_size, seq_len, d_model = 4, 16, 512
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, aux_loss = moe(x)
        
        if moe.use_aux_loss:
            result = aux_loss is not None
            expected = True
        else:
            result = aux_loss is None
            expected = True
        
        return result, expected
    
    def test_moe_different_from_input(moe):
        # MoE should transform the input
        batch_size, seq_len, d_model = 2, 8, 512
        x = torch.randn(batch_size, seq_len, d_model)
        
        with torch.no_grad():
            output, _ = moe(x)
        
        result = torch.allclose(output, x, atol=1e-3)
        expected = False
        
        return result, expected
    
    def test_moe_variable_seq_length(moe):
        # Should handle different sequence lengths
        batch_size, d_model = 2, 512
        
        for seq_len in [4, 8, 16]:
            x = torch.randn(batch_size, seq_len, d_model)
            output, _ = moe(x)
            if output.shape != x.shape:
                return output.shape, x.shape
        
        return True, True
    
    def test_moe_expert_count(moe):
        # Should have the correct number of experts
        result = len(moe.experts) if hasattr(moe, 'experts') else 0
        expected = moe.num_experts
        
        return result, expected
    
    moe = MixtureOfExperts(
        d_model=512,
        d_ff=2048,
        num_experts=8,
        top_k=2,
        dropout=0.0,
        use_aux_loss=True
    )
    
    moe_tests = [
        (test_moe_shape, "MoE output shape"),
        (test_moe_aux_loss, "MoE auxiliary loss"),
        (test_moe_different_from_input, "MoE transforms input"),
        (test_moe_variable_seq_length, "MoE handles variable lengths"),
        (test_moe_expert_count, "MoE has correct number of experts"),
    ]
    
    test_ml_implementation(moe, moe_tests)
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

