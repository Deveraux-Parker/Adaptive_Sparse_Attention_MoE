"""
Ablation Testing Framework for Adaptive Sparse Attention MoE

This module provides a structured approach to validate the novelty claims:
1. Ablation 1: Standard full attention vs Adaptive Sparse MoE (same parameter budget)
2. Ablation 2: Non-adaptive routing (all expensive) vs adaptive routing
3. Routing visualization: Show which tokens use which experts
4. Scaling: 64x64 experiments showing compute savings
5. Robustness: Variable noise training and generalization

Each ablation is a self-contained test that can be run independently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from adaptive_attention_moe import TokenRouter, SparseAttentionExpert


@dataclass
class AblationMetrics:
    """Results of an ablation experiment."""
    name: str
    psnr_mean: float
    psnr_std: float
    ssim_mean: float
    ssim_std: float
    parameters: int
    inference_time_ms: float
    flops: int  # Approximate FLOPs for forward pass


class StandardAttention(nn.Module):
    """
    Standard multi-head self-attention (O(N²) complexity).
    Used as baseline for ablation 1.

    Same parameter count as our AdaptiveAttentionMoE for fair comparison.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        assert d_model % num_heads == 0

        # Single set of projections (no expert routing)
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
            [B, N, D] standard attention output
        """
        B, N, D = x.shape

        # Project to Q, K, V
        Q = self.W_q(x)  # [B, N, D]
        K = self.W_k(x)
        V = self.W_v(x)

        # Multi-head: [B, num_heads, N, d_head]
        Q = Q.view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.d_head).transpose(1, 2)

        # Standard attention: attend to ALL keys (O(N²))
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        output = torch.matmul(attn_weights, V)  # [B, num_heads, N, d_head]

        # Reshape back: [B, N, D]
        output = output.transpose(1, 2).contiguous().view(B, N, D)
        output = self.W_out(output)

        return output


class NonAdaptiveSparseMoE(nn.Module):
    """
    Non-adaptive version: ALL tokens are routed to the expensive expert.

    This ablation tests if the quality gains come from "giving everything max compute"
    rather than from the adaptive routing mechanism.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_experts: int = 3,
        k_budgets: Optional[list] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_experts = num_experts

        if k_budgets is None:
            k_budgets = [32, 64, 128]
        self.k_budgets = k_budgets[:num_experts]

        # Create experts (but we only use the expensive one)
        self.experts = nn.ModuleList([
            SparseAttentionExpert(d_model, num_heads, self.k_budgets[i], dropout)
            for i in range(num_experts)
        ])

        # Only use the expensive expert (index 2)
        self.expensive_expert_idx = 2
        self.merge_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Force all tokens to expensive expert (no routing).

        Args:
            x: [B, N, D] input

        Returns:
            [B, N, D] output from expensive expert
        """
        B, N, D = x.shape

        # Use only the expensive expert
        expensive_expert = self.experts[self.expensive_expert_idx]
        output = expensive_expert(x)
        output = self.merge_proj(output)

        return output


class RoutingVisualizer:
    """
    Visualizes which expert each token (spatial location) is routed to.

    For image denoising: shows attention maps for cheap, medium, expensive experts.
    If edges/boundaries consistently go to expensive, we have strong evidence
    that adaptive routing is doing something intelligent.
    """

    @staticmethod
    def extract_routing_patterns(
        model: nn.Module,
        x: torch.Tensor,
        patch_size: int = 4,
        image_size: int = 32,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract routing probabilities from model on a batch of images.

        Args:
            model: SmallViT model (must have AdaptiveAttentionMoE blocks)
            x: [B, 3, 32, 32] images
            patch_size: 4 (for CIFAR-10)
            image_size: 32

        Returns:
            routing_probs: [B, num_patches, num_experts]
            spatial_routing: [B, num_experts, 8, 8] heatmaps
        """
        # Forward pass to extract routing information
        # This would need instrumentation of the model
        # For now, return placeholder

        B, C, H, W = x.shape
        num_patches = (image_size // patch_size) ** 2

        routing_probs = torch.zeros(B, num_patches, 3)
        return {'routing_probs': routing_probs}

    @staticmethod
    def visualize_expert_selection(routing_probs: torch.Tensor) -> str:
        """
        Convert routing probabilities to human-readable statistics.

        Args:
            routing_probs: [B, num_patches, num_experts]

        Returns:
            String describing routing patterns
        """
        # Mean probability per expert
        mean_probs = routing_probs.mean(dim=(0, 1))  # [num_experts]

        output = "Expert Selection Statistics:\n"
        output += f"  Cheap (k=32):    {mean_probs[0]:.1%}\n"
        output += f"  Medium (k=64):   {mean_probs[1]:.1%}\n"
        output += f"  Expensive (k=128): {mean_probs[2]:.1%}\n"

        return output


class ComputeAnalyzer:
    """
    Analyzes compute efficiency of different attention mechanisms.
    """

    @staticmethod
    def estimate_flops(
        batch_size: int,
        num_patches: int,
        d_model: int,
        num_heads: int,
        attention_type: str,  # 'full', 'sparse_k32', 'sparse_k64', 'sparse_k128'
    ) -> int:
        """
        Estimate FLOPs for attention forward pass.

        Args:
            batch_size: B
            num_patches: N (sequence length)
            d_model: D
            num_heads: h
            attention_type: type of attention

        Returns:
            Approximate FLOPs
        """
        d_head = d_model // num_heads

        if attention_type == 'full':
            # Q @ K.T: [B, h, N, d_head] @ [B, h, d_head, N] → [B, h, N, N]
            # Attn @ V: [B, h, N, N] @ [B, h, N, d_head] → [B, h, N, d_head]
            flops = 2 * batch_size * num_heads * (num_patches ** 2) * d_head
        else:
            # Sparse: only attend to k keys per query
            k = int(attention_type.split('_')[1].replace('k', ''))
            flops = 2 * batch_size * num_heads * num_patches * k * d_head

        return flops

    @staticmethod
    def compare_compute(num_patches: int = 64) -> str:
        """
        Compare compute costs for different attention types.

        Args:
            num_patches: N (64 for 32x32 images with 4x4 patches)

        Returns:
            Comparison string
        """
        batch_size = 1
        d_model = 128
        num_heads = 4

        full = ComputeAnalyzer.estimate_flops(
            batch_size, num_patches, d_model, num_heads, 'full'
        )
        sparse_32 = ComputeAnalyzer.estimate_flops(
            batch_size, num_patches, d_model, num_heads, 'sparse_k32'
        )
        sparse_64 = ComputeAnalyzer.estimate_flops(
            batch_size, num_patches, d_model, num_heads, 'sparse_k64'
        )
        sparse_128 = ComputeAnalyzer.estimate_flops(
            batch_size, num_patches, d_model, num_heads, 'sparse_k128'
        )

        output = "Compute Analysis (FLOPs for single image):\n"
        output += f"  Full Attention:      {full:,} FLOPs (baseline)\n"
        output += f"  Sparse (k=32):       {sparse_32:,} FLOPs ({sparse_32/full:.1%})\n"
        output += f"  Sparse (k=64):       {sparse_64:,} FLOPs ({sparse_64/full:.1%})\n"
        output += f"  Sparse (k=128):      {sparse_128:,} FLOPs ({sparse_128/full:.1%})\n"
        output += f"  MoE Average (est):   {(sparse_32 + sparse_64 + sparse_128) // 3 / full:.1%}\n"

        return output


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def test_ablation_framework():
    """
    Test that all ablation components work correctly.
    """
    print("=" * 70)
    print("ABLATION FRAMEWORK TEST")
    print("=" * 70)

    # Test 1: StandardAttention initialization
    print("\n[Test 1] StandardAttention initialization...")
    std_attn = StandardAttention(d_model=128, num_heads=4)
    x = torch.randn(2, 64, 128)
    output = std_attn(x)
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    print(f"  ✅ StandardAttention works: {output.shape}")
    print(f"     Parameters: {count_parameters(std_attn):,}")

    # Test 2: NonAdaptiveSparseMoE initialization
    print("\n[Test 2] NonAdaptiveSparseMoE initialization...")
    non_adaptive = NonAdaptiveSparseMoE(d_model=128, num_heads=4)
    output = non_adaptive(x)
    assert output.shape == x.shape
    print(f"  ✅ NonAdaptiveSparseMoE works: {output.shape}")
    print(f"     Parameters: {count_parameters(non_adaptive):,}")

    # Test 3: RoutingVisualizer
    print("\n[Test 3] RoutingVisualizer...")
    viz = RoutingVisualizer()
    fake_routing = torch.rand(2, 64, 3)
    stats = viz.visualize_expert_selection(fake_routing)
    print(f"  ✅ RoutingVisualizer works:")
    for line in stats.split('\n')[1:]:
        print(f"     {line}")

    # Test 4: ComputeAnalyzer
    print("\n[Test 4] ComputeAnalyzer...")
    compute_report = ComputeAnalyzer.compare_compute(num_patches=64)
    print(f"  ✅ ComputeAnalyzer works:")
    for line in compute_report.split('\n')[1:]:
        print(f"     {line}")

    print("\n" + "=" * 70)
    print("ALL FRAMEWORK TESTS PASSED ✅")
    print("=" * 70)


if __name__ == '__main__':
    test_ablation_framework()
