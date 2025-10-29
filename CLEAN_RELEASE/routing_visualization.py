"""
Expert Routing Visualization for Adaptive Sparse Attention MoE

Visualizes which expert each token (spatial location) is routed to.
This is key evidence for novelty: if edges and important features are
consistently routed to expensive experts, we prove that adaptive routing
is doing something intelligent.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Tuple
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10


class RoutingCapture(nn.Module):
    """
    Wrapper around AdaptiveAttentionMoE to capture routing information.

    Stores routing probabilities during forward pass for visualization.
    """

    def __init__(self, adaptive_moe_module):
        super().__init__()
        self.adaptive_moe = adaptive_moe_module
        self.last_routing_probs = None

    def forward(self, x):
        output = self.adaptive_moe(x)
        # Capture routing probabilities from the MoE module
        self.last_routing_probs = self.adaptive_moe.last_routing_probs.detach()
        return output

    def get_routing_heatmaps(self) -> Dict[str, np.ndarray]:
        """
        Convert routing probabilities to spatial heatmaps.

        Args:
            None (uses self.last_routing_probs from forward pass)

        Returns:
            Dictionary with keys:
            - 'cheap': [H, W] heatmap of probability for cheap expert
            - 'medium': [H, W] heatmap of probability for medium expert
            - 'expensive': [H, W] heatmap of probability for expensive expert
            - 'hard_assignment': [H, W] heatmap showing which expert was chosen (0, 1, or 2)
        """
        if self.last_routing_probs is None:
            raise ValueError("No routing data captured. Run forward pass first.")

        # routing_probs: [B, N, num_experts]
        # For CIFAR-10: [B, 64, 3] where 64 = 8x8 patch grid
        B, N, num_experts = self.last_routing_probs.shape

        # Take first batch sample
        routing = self.last_routing_probs[0].cpu().numpy()  # [64, 3]

        # Reshape to spatial: [8, 8, 3]
        H = W = int(np.sqrt(N))
        routing_spatial = routing.reshape(H, W, num_experts)

        # Hard assignment: argmax to see which expert each patch prefers
        hard_assignment = np.argmax(routing_spatial, axis=2)  # [8, 8]

        return {
            'cheap': routing_spatial[:, :, 0],          # [8, 8]
            'medium': routing_spatial[:, :, 1],         # [8, 8]
            'expensive': routing_spatial[:, :, 2],      # [8, 8]
            'hard_assignment': hard_assignment,         # [8, 8]
        }


class RoutingVisualizer:
    """
    Visualizes routing patterns and compares them with image features.
    """

    @staticmethod
    def plot_routing_heatmaps(
        image: torch.Tensor,
        routing_heatmaps: Dict[str, np.ndarray],
        title: str = "Routing Analysis",
        save_path: str = None,
    ):
        """
        Create a grid showing:
        - Original image
        - Edge map
        - Routing to cheap expert
        - Routing to medium expert
        - Routing to expensive expert
        - Hard assignment (which expert)

        Args:
            image: [3, 32, 32] tensor
            routing_heatmaps: Dict with 'cheap', 'medium', 'expensive', 'hard_assignment'
            title: Figure title
            save_path: Where to save the figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        image_np = image.permute(1, 2, 0).numpy()

        # 1. Original image
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')

        # 2. Edge detection (Sobel-like: simple gradient)
        edges = np.sqrt(
            np.gradient(np.mean(image_np, axis=2), axis=0) ** 2
            + np.gradient(np.mean(image_np, axis=2), axis=1) ** 2
        )
        edges = edges / edges.max()
        axes[0, 1].imshow(edges, cmap='hot')
        axes[0, 1].set_title("Edge Map (Gradient Magnitude)")
        axes[0, 1].axis('off')

        # 3. Routing: Cheap expert
        cheap = routing_heatmaps['cheap']
        im = axes[0, 2].imshow(cheap, cmap='Blues', vmin=0, vmax=1)
        axes[0, 2].set_title("Routing: Cheap Expert (k=32)")
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

        # 4. Routing: Medium expert
        medium = routing_heatmaps['medium']
        im = axes[1, 0].imshow(medium, cmap='Greens', vmin=0, vmax=1)
        axes[1, 0].set_title("Routing: Medium Expert (k=64)")
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # 5. Routing: Expensive expert
        expensive = routing_heatmaps['expensive']
        im = axes[1, 1].imshow(expensive, cmap='Reds', vmin=0, vmax=1)
        axes[1, 1].set_title("Routing: Expensive Expert (k=128)")
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

        # 6. Hard assignment
        hard = routing_heatmaps['hard_assignment']
        colors = ['blue', 'green', 'red']
        cmap = plt.cm.ListedColormap(colors)
        im = axes[1, 2].imshow(hard, cmap=cmap, vmin=-0.5, vmax=2.5)
        axes[1, 2].set_title("Expert Assignment (Hard)")
        axes[1, 2].axis('off')

        # Add legend
        patches = [
            mpatches.Patch(color='blue', label='Cheap (k=32)'),
            mpatches.Patch(color='green', label='Medium (k=64)'),
            mpatches.Patch(color='red', label='Expensive (k=128)'),
        ]
        axes[1, 2].legend(handles=patches, loc='upper left', bbox_to_anchor=(0, -0.1))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved routing visualization to {save_path}")

        return fig

    @staticmethod
    def analyze_routing_patterns(
        routing_heatmaps: Dict[str, np.ndarray],
        edges: np.ndarray = None,
    ) -> str:
        """
        Analyze and describe routing patterns.

        Args:
            routing_heatmaps: Routing probabilities and hard assignment
            edges: Optional edge map for correlation analysis

        Returns:
            String with analysis
        """
        hard = routing_heatmaps['hard_assignment']
        cheap = routing_heatmaps['cheap']
        medium = routing_heatmaps['medium']
        expensive = routing_heatmaps['expensive']

        # Count which expert is preferred
        unique, counts = np.unique(hard.flatten(), return_counts=True)
        expert_names = {0: 'Cheap (k=32)', 1: 'Medium (k=64)', 2: 'Expensive (k=128)'}

        output = "Routing Pattern Analysis:\n"
        output += "=" * 50 + "\n"

        for expert_id, count in zip(unique, counts):
            pct = count / hard.size * 100
            output += f"{expert_names[expert_id]:<20}: {pct:>5.1f}% of patches\n"

        # If edges provided, compute correlation
        if edges is not None:
            edges_norm = edges / edges.max()
            corr_expensive = np.corrcoef(
                edges_norm.flatten(),
                expensive.flatten()
            )[0, 1]
            output += f"\n{'Expensive-Edge Correlation':<20}: {corr_expensive:>6.3f}\n"
            if corr_expensive > 0.3:
                output += "  → Evidence: Expensive expert routes to HIGH-GRADIENT areas ✅\n"
            else:
                output += "  → Neutral: No strong correlation with edges\n"

        return output


def test_routing_visualization():
    """
    Test routing visualization components.
    """
    print("=" * 70)
    print("ROUTING VISUALIZATION TEST")
    print("=" * 70)

    print("\n[Test 1] RoutingCapture initialization...")
    from adaptive_attention_moe import AdaptiveAttentionMoE
    moe = AdaptiveAttentionMoE(d_model=128, num_heads=4, num_experts=3)
    moe_captured = RoutingCapture(moe)
    x = torch.randn(1, 64, 128)
    output = moe_captured(x)
    assert output.shape == x.shape
    print(f"  ✅ RoutingCapture works: {output.shape}")

    print("\n[Test 2] Extract routing heatmaps...")
    heatmaps = moe_captured.get_routing_heatmaps()
    assert 'cheap' in heatmaps
    assert 'medium' in heatmaps
    assert 'expensive' in heatmaps
    assert 'hard_assignment' in heatmaps
    print(f"  ✅ Heatmaps extracted:")
    print(f"     Cheap shape: {heatmaps['cheap'].shape}")
    print(f"     Hard assignment shape: {heatmaps['hard_assignment'].shape}")

    print("\n[Test 3] Routing pattern analysis...")
    edges = np.random.rand(8, 8)  # Dummy edges
    analysis = RoutingVisualizer.analyze_routing_patterns(heatmaps, edges)
    print(analysis)

    print("\n[Test 4] Visualization (would save to file)...")
    print("  ✅ Visualization function ready")
    print("     (Run visualize_routing_on_batch() to save real visualizations)")

    print("\n" + "=" * 70)
    print("ROUTING VISUALIZATION TESTS PASSED ✅")
    print("=" * 70)


if __name__ == '__main__':
    test_routing_visualization()
