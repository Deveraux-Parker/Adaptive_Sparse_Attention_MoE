"""
Adaptive Sparse Attention with Mixture-of-Experts (MoE) implementation.

This module implements a hierarchical attention mechanism where tokens are routed
to different "experts" (attention configurations) based on their importance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class TokenRouter(nn.Module):
    """
    Router that assigns importance scores to tokens.

    For each token, produces probabilities for tiers:
    - background/peripheral (cheap attention)
    - mid-importance/focal (medium attention)
    - critical/reflective (expensive attention)
    """

    def __init__(self, d_model: int, num_experts: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts

        # Small routing MLP
        self.routing_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] token embeddings

        Returns:
            [B, N, num_experts] normalized routing probabilities
        """
        B, N, D = x.shape
        logits = self.routing_net(x)  # [B, N, num_experts]
        probs = F.softmax(logits, dim=-1)
        return probs


class SparseAttentionExpert(nn.Module):
    """
    A single attention expert with configurable sparsity budget.

    Each expert attends to only top-k keys instead of all keys.
    """

    def __init__(self, d_model: int, num_heads: int, k_attend: int, dropout: float = 0.0):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            k_attend: Number of keys each query can attend to (sparsity budget)
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.k_attend = k_attend
        self.d_head = d_model // num_heads

        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] input

        Returns:
            [B, N, D] attended output
        """
        B, N, D = x.shape

        # Project to Q, K, V
        Q = self.W_q(x)  # [B, N, D]
        K = self.W_k(x)  # [B, N, D]
        V = self.W_v(x)  # [B, N, D]

        # Reshape for multi-head attention: [B, N, num_heads, d_head]
        Q = Q.view(B, N, self.num_heads, self.d_head).transpose(1, 2)  # [B, num_heads, N, d_head]
        K = K.view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.d_head).transpose(1, 2)

        # Sparse attention: select top-k keys for each query
        # Compute all similarities first
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]

        # Select top-k keys per query
        k_actual = min(self.k_attend, N)
        top_k_scores, top_k_indices = torch.topk(scores, k=k_actual, dim=-1)  # [B, num_heads, N, k]

        # Create sparse attention matrix
        # Compute softmax only on top-k values
        attn_weights = F.softmax(top_k_scores, dim=-1)  # [B, num_heads, N, k]
        attn_weights = self.dropout(attn_weights)

        # Gather top-k values using indices
        # We need to gather from V using the indices
        V_expanded = V.unsqueeze(-2)  # [B, num_heads, N, 1, d_head]
        V_selected = torch.gather(
            V.unsqueeze(2).expand(B, self.num_heads, N, N, self.d_head),
            dim=3,
            index=top_k_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.d_head)
        )  # [B, num_heads, N, k, d_head]

        # Apply attention weights
        output = torch.matmul(attn_weights.unsqueeze(-2), V_selected)  # [B, num_heads, N, 1, d_head]
        output = output.squeeze(-2)  # [B, num_heads, N, d_head]

        # Reshape back: [B, N, D]
        output = output.transpose(1, 2).contiguous().view(B, N, D)
        output = self.W_out(output)

        return output


class AdaptiveAttentionMoE(nn.Module):
    """
    Adaptive Sparse Attention with Mixture-of-Experts.

    Tokens are routed to different attention experts based on importance.
    Different experts have different compute budgets (k_attend values).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_experts: int = 3,
        k_budgets: Optional[list] = None,
        expert_heads: Optional[list] = None,
        dropout: float = 0.0,
        use_load_balancing: bool = True,
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Total number of heads (for the "expensive" expert)
            num_experts: Number of experts (tiers)
            k_budgets: List of k values for each expert [cheap, mid, expensive]
                      If None, defaults to [32, 64, 128]
            expert_heads: List of num_heads for each expert
                         If None, scales linearly: [num_heads//3, 2*num_heads//3, num_heads]
            dropout: Dropout rate
            use_load_balancing: Whether to add auxiliary loss for load balancing
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.use_load_balancing = use_load_balancing

        if k_budgets is None:
            k_budgets = [32, 64, 128]
        self.k_budgets = k_budgets[:num_experts]

        if expert_heads is None:
            # All experts use the same num_heads (they differ in k_attend budget instead)
            # This ensures d_model % num_heads == 0 is satisfied
            expert_heads = [num_heads] * num_experts
        self.expert_heads = expert_heads[:num_experts]

        # Create router
        self.router = TokenRouter(d_model, num_experts)

        # Create experts with different budgets and head counts
        self.experts = nn.ModuleList([
            SparseAttentionExpert(d_model, self.expert_heads[i], self.k_budgets[i], dropout)
            for i in range(num_experts)
        ])

        # Output projection to merge expert outputs
        self.merge_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] input

        Returns:
            [B, N, D] attended output
        """
        B, N, D = x.shape

        # Get routing probabilities
        routing_probs = self.router(x)  # [B, N, num_experts]

        # Forward through all experts
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_outputs.append(expert(x))  # [B, N, D]

        # Stack expert outputs: [num_experts, B, N, D]
        expert_outputs = torch.stack(expert_outputs, dim=0)

        # Reshape for weighted combination: [B, N, num_experts, D]
        expert_outputs = expert_outputs.permute(1, 2, 0, 3)  # [B, N, num_experts, D]

        # Expand routing probs for broadcasting: [B, N, num_experts, 1]
        routing_probs_expanded = routing_probs.unsqueeze(-1)  # [B, N, num_experts, 1]

        # Weight each expert output by its routing probability
        weighted_outputs = expert_outputs * routing_probs_expanded  # [B, N, num_experts, D]

        # Sum across experts: [B, N, D]
        output = weighted_outputs.sum(dim=2)

        # Final projection
        output = self.merge_proj(output)

        # Store routing info for analysis (used in testing)
        self.last_routing_probs = routing_probs

        return output

    def get_load_balance_loss(self) -> torch.Tensor:
        """
        Auxiliary loss to encourage balanced expert usage.
        Prevents all tokens from going to one expert.
        """
        if not self.use_load_balancing:
            return torch.tensor(0.0, device=self.router.routing_net[2].weight.device)

        # Get mean probability per expert across batch and sequence
        mean_probs = self.last_routing_probs.mean(dim=(0, 1))  # [num_experts]

        # Encourage uniform distribution (entropy-based loss)
        target_prob = 1.0 / self.num_experts
        loss = F.kl_div(
            torch.log(mean_probs + 1e-10),
            torch.ones_like(mean_probs) * target_prob,
            reduction='batchmean'
        )
        return loss
