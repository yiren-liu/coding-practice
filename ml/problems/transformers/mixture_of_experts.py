"""
ML Implementation: Mixture of Experts (MoE) Layer

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
        # TODO: Implement a simple FFN
        # Linear(d_model -> d_ff) -> Activation -> Dropout -> Linear(d_ff -> d_model)
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [num_tokens, d_model]
        Returns:
            Output tensor of shape [num_tokens, d_model]
        """
        # TODO: Implement forward pass
        pass


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
        
        # TODO: Initialize gating network
        # A simple linear layer that maps input to expert logits
        # Hint: Linear(d_model -> num_experts)
        pass
    
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
        # TODO: Implement gating logic:
        # 1. Compute logits for all experts
        # 2. Select top-k experts per token
        # 3. Compute softmax over selected experts
        # 4. Create sparse weight matrix
        # 5. Optionally compute load balancing loss
        
        # Hints:
        # - Use torch.topk() to select top-k experts
        # - Softmax should be over the top-k experts only
        # - Gate weights should sum to 1 for each token (over the k selected experts)
        pass
    
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
        # TODO: Implement load balancing loss
        # Common approach: encourage uniform distribution of tokens across experts
        # Loss = num_experts * sum(f_i * P_i) where:
        #   f_i = fraction of tokens routed to expert i
        #   P_i = average probability assigned to expert i
        pass


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
        
        # TODO: Initialize components:
        # 1. Create num_experts expert networks (use nn.ModuleList)
        # 2. Create gating network
        pass
    
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
        
        # TODO: Implement MoE forward pass:
        # 1. Reshape input to [batch_size * seq_len, d_model]
        # 2. Get gate weights and selected experts
        # 3. For each token, process with selected experts
        # 4. Combine expert outputs using gate weights
        # 5. Reshape back to [batch_size, seq_len, d_model]
        
        # Implementation approaches:
        # A. Naive: Loop over each token and its selected experts
        # B. Efficient: Batch process all tokens going to same expert
        # Start with A for correctness, optimize to B if time permits
        pass
    
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
        # TODO: Process token through each selected expert and combine
        pass


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
    print("Implementation Tips:")
    print("=" * 60)
    print("""
Expert:
- Simple FFN: Linear -> GELU/ReLU -> Dropout -> Linear
- Input/output dimension: d_model
- Hidden dimension: d_ff (typically 4 * d_model)

TopK Gate:
- Linear layer: d_model -> num_experts (produces logits)
- torch.topk(logits, k=top_k) to select experts
- softmax over selected experts only
- Create sparse weight tensor with selected weights

Mixture of Experts:
1. Flatten: [batch, seq_len, d_model] -> [batch*seq_len, d_model]
2. Gate: get weights and expert indices for each token
3. Route: send each token to its selected experts
4. Combine: weighted sum of expert outputs
5. Reshape: back to [batch, seq_len, d_model]

Load Balancing Loss:
- Prevents all tokens going to same few experts
- loss = num_experts * sum(f_i * P_i)
- f_i = fraction of tokens assigned to expert i
- P_i = mean probability of expert i across all tokens

Optimization Notes:
- Naive: Loop over tokens (easier to implement)
- Efficient: Batch all tokens for same expert together
- Trade-off: Start naive, optimize if needed
    """)
    
    print("\n" + "=" * 60)
    print("Challenge: Efficient Implementation")
    print("=" * 60)
    print("""
For an efficient implementation, consider:

1. Group tokens by expert:
   - Collect all tokens going to expert_0
   - Process them in one batch through expert_0
   - Repeat for all experts

2. Use scatter/gather operations:
   - torch.scatter() to distribute tokens
   - torch.gather() to collect results

3. Handle variable capacity:
   - What if too many tokens select the same expert?
   - Capacity factor: limit tokens per expert
   - Drop tokens that exceed capacity

This is how production MoE systems work (Switch Transformer, etc.)
    """)

