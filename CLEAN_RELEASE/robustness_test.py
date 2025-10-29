"""
Robustness Testing: Variable Noise Generalization

Tests whether the model learned to denoise across a spectrum of noise levels,
not just memorized σ=0.2 statistics.

Two experiments:
1. Train with variable σ∈[0.1, 0.3] (random at each epoch)
2. Test at unseen noise levels: σ=0.05, 0.15, 0.25, 0.35, 0.4
3. Compare: Trained at fixed σ=0.2 vs trained at variable σ
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import List, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from train import SmallViT, NoisyImageDataset


class VariableNoiseDataset(Dataset):
    """
    Dataset that varies noise level σ for each sample.
    This enables training on a spectrum of noise levels.
    """

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        noise_level_range: Tuple[float, float] = (0.1, 0.3),
        fixed_noise: float = None,
    ):
        """
        Args:
            images: [N, 3, 32, 32] pre-normalized to [0, 1]
            labels: [N] class labels
            noise_level_range: (min_σ, max_σ) for random noise levels
            fixed_noise: If set, use this fixed noise level (for baseline)
        """
        self.images = images.float()
        self.labels = labels
        self.noise_level_range = noise_level_range
        self.fixed_noise = fixed_noise

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        clean = self.images[idx]  # [3, 32, 32]

        # Vary noise level
        if self.fixed_noise is not None:
            sigma = self.fixed_noise
        else:
            # Random σ in range for each sample
            sigma = torch.rand(1).item() * (
                self.noise_level_range[1] - self.noise_level_range[0]
            ) + self.noise_level_range[0]

        # Add Gaussian noise
        noise = torch.randn_like(clean) * sigma
        noisy = torch.clamp(clean + noise, 0, 1)

        return noisy, clean, self.labels[idx], sigma


def train_robustness_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int = 10,
    model_name: str = "model",
) -> Tuple[nn.Module, List[float]]:
    """
    Train model with variable noise support.

    Args:
        model: Model to train
        train_loader: Training loader (uses VariableNoiseDataset)
        val_loader: Validation loader
        device: Device to use
        epochs: Number of epochs
        model_name: Name for logging

    Returns:
        Trained model, loss history
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_state = model.state_dict()
    loss_history = []

    print(f"\n[{model_name}] Starting training...")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            if len(batch) == 4:  # VariableNoiseDataset returns 4 items
                noisy, clean, _, sigmas = batch
            else:  # Regular NoisyImageDataset
                noisy, clean, _ = batch

            noisy, clean = noisy.to(device), clean.to(device)

            optimizer.zero_grad()
            pred = model(noisy)
            loss = criterion(pred, clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    noisy, clean, _, _ = batch
                else:
                    noisy, clean, _ = batch

                noisy, clean = noisy.to(device), clean.to(device)
                pred = model(noisy)
                loss = criterion(pred, clean)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        loss_history.append(val_loss)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    model.load_state_dict(best_model_state)
    model.eval()
    print(f"[{model_name}] Training complete. Best val loss: {best_val_loss:.6f}")

    return model, loss_history


def evaluate_robustness(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: str,
    test_noise_levels: List[float] = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
    model_name: str = "model",
) -> dict:
    """
    Evaluate model on a range of noise levels.

    Args:
        model: Model to evaluate
        images: [N, 3, 32, 32] test images
        labels: [N] labels
        device: Device
        test_noise_levels: List of σ values to test
        model_name: Name for logging

    Returns:
        Dictionary mapping σ → (psnr_mean, ssim_mean)
    """
    model = model.to(device)
    model.eval()

    results = {}

    print(f"\n[{model_name}] Robustness Evaluation:")
    print(f"{'Noise Level':<15} {'PSNR':<12} {'SSIM':<12}")
    print("-" * 40)

    for sigma in test_noise_levels:
        # Create test dataset with this specific noise level
        test_dataset = VariableNoiseDataset(images, labels, fixed_noise=sigma)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        psnr_scores = []
        ssim_scores = []

        with torch.no_grad():
            for noisy, clean, _, _ in test_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                pred = model(noisy)

                for i in range(noisy.shape[0]):
                    clean_np = clean[i].permute(1, 2, 0).cpu().numpy()
                    pred_np = pred[i].permute(1, 2, 0).cpu().numpy()

                    psnr = peak_signal_noise_ratio(clean_np, pred_np, data_range=1.0)
                    ssim = structural_similarity(
                        clean_np, pred_np, data_range=1.0, channel_axis=2
                    )

                    psnr_scores.append(psnr)
                    ssim_scores.append(ssim)

        psnr_mean = sum(psnr_scores) / len(psnr_scores)
        ssim_mean = sum(ssim_scores) / len(ssim_scores)

        results[sigma] = (psnr_mean, ssim_mean)

        print(
            f"σ = {sigma:<6.2f}         {psnr_mean:<6.2f} dB      {ssim_mean:<8.4f}"
        )

    return results


def run_robustness_tests():
    """Run comprehensive robustness experiments."""
    print("=" * 80)
    print("ROBUSTNESS TEST: Variable Noise Generalization")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Load data
    print("\nLoading CIFAR-10 data...")
    transform = transforms.ToTensor()
    train_set = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_images = torch.stack([img for img, _ in train_set])[:4000]
    train_labels = torch.tensor([train_set[i][1] for i in range(4000)])
    test_images = torch.stack([img for img, _ in test_set])[:1000]
    test_labels = torch.tensor([test_set[i][1] for i in range(1000)])

    # Split train for validation
    train_dataset_var = VariableNoiseDataset(
        train_images, train_labels, noise_level_range=(0.1, 0.3), fixed_noise=None
    )
    train_dataset_fixed = VariableNoiseDataset(
        train_images, train_labels, noise_level_range=(0.1, 0.3), fixed_noise=0.2
    )

    train_size = int(0.8 * len(train_dataset_var))
    val_size = len(train_dataset_var) - train_size

    train_var, val_var = random_split(train_dataset_var, [train_size, val_size])
    train_fixed, val_fixed = random_split(train_dataset_fixed, [train_size, val_size])

    train_loader_var = DataLoader(train_var, batch_size=64, shuffle=True)
    val_loader_var = DataLoader(val_var, batch_size=64, shuffle=False)

    train_loader_fixed = DataLoader(train_fixed, batch_size=64, shuffle=True)
    val_loader_fixed = DataLoader(val_fixed, batch_size=64, shuffle=False)

    # Experiment 1: Fixed noise training (baseline)
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Fixed Noise Training (σ=0.2 only)")
    print("=" * 80)

    model_fixed = SmallViT(image_size=32, patch_size=4, embed_dim=128, depth=2, num_heads=4)
    model_fixed_trained, _ = train_robustness_model(
        model_fixed,
        train_loader_fixed,
        val_loader_fixed,
        device,
        epochs=8,
        model_name="FixedNoise",
    )

    results_fixed = evaluate_robustness(
        model_fixed_trained,
        test_images,
        test_labels,
        device,
        test_noise_levels=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
        model_name="FixedNoise",
    )

    # Experiment 2: Variable noise training (robust)
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Variable Noise Training (σ∈[0.1, 0.3])")
    print("=" * 80)

    model_var = SmallViT(image_size=32, patch_size=4, embed_dim=128, depth=2, num_heads=4)
    model_var_trained, _ = train_robustness_model(
        model_var,
        train_loader_var,
        val_loader_var,
        device,
        epochs=8,
        model_name="VariableNoise",
    )

    results_var = evaluate_robustness(
        model_var_trained,
        test_images,
        test_labels,
        device,
        test_noise_levels=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
        model_name="VariableNoise",
    )

    # Summary comparison
    print("\n" + "=" * 80)
    print("ROBUSTNESS COMPARISON SUMMARY")
    print("=" * 80)

    print(f"\n{'Noise Level':<12} {'Fixed σ=0.2':<20} {'Variable σ∈[0.1,0.3]':<25} {'Gain':<10}")
    print(f"{'':12} {'PSNR | SSIM':<20} {'PSNR | SSIM':<25} {'PSNR':<10}")
    print("-" * 75)

    for sigma in sorted(results_fixed.keys()):
        psnr_f, ssim_f = results_fixed[sigma]
        psnr_v, ssim_v = results_var[sigma]
        gain = psnr_v - psnr_f

        gain_str = f"{gain:+.2f} dB"
        print(
            f"σ = {sigma:<4.2f}      {psnr_f:6.2f} | {ssim_f:.4f}      "
            f"{psnr_v:6.2f} | {ssim_v:.4f}      {gain_str:<10}"
        )

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Analyze robustness
    fixed_at_020 = results_fixed[0.20][0]
    var_at_020 = results_var[0.20][0]
    print(f"\n1. Performance at training noise level (σ=0.2):")
    print(f"   Fixed model:    {fixed_at_020:.2f} dB")
    print(f"   Variable model: {var_at_020:.2f} dB")
    if abs(var_at_020 - fixed_at_020) < 0.5:
        print(f"   → Similar performance at training distribution ✓")

    # Robustness to unseen noise
    fixed_at_unseen = [
        results_fixed[s][0] for s in [0.05, 0.10, 0.25, 0.30, 0.35]
    ]
    var_at_unseen = [
        results_var[s][0] for s in [0.05, 0.10, 0.25, 0.30, 0.35]
    ]

    fixed_avg_unseen = sum(fixed_at_unseen) / len(fixed_at_unseen)
    var_avg_unseen = sum(var_at_unseen) / len(var_at_unseen)

    print(f"\n2. Robustness to unseen noise levels (σ∉[0.1, 0.3]):")
    print(f"   Fixed model avg PSNR:    {fixed_avg_unseen:.2f} dB")
    print(f"   Variable model avg PSNR: {var_avg_unseen:.2f} dB")
    print(f"   Improvement:             {var_avg_unseen - fixed_avg_unseen:+.2f} dB")

    if var_avg_unseen > fixed_avg_unseen + 0.2:
        print(f"   → Variable noise training improves generalization! ✅")
    else:
        print(f"   → Generalization is comparable")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    run_robustness_tests()
