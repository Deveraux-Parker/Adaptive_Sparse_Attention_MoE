"""
PROPER Denoising Experiment - Train at native CIFAR-10 32x32 resolution.

Previous experiment had a data pipeline issue:
- Upsampled 32x32 → 64x64 (blur/simplification)
- Model learned to denoise the BLURRY version (easy)
- Downsampling revealed the failure

This version:
- Keeps images at native 32x32
- Uses a smaller ViT architecture appropriate for 32x32
- Actual denoising task (not upsampling artifact reduction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from adaptive_attention_moe import AdaptiveAttentionMoE


class NoisyImageDataset(Dataset):
    """Creates noisy versions of images."""

    def __init__(self, images, labels, noise_level=0.2):
        """
        Args:
            images: [N, 3, 32, 32] float32 tensor already normalized to [0, 1]
            labels: [N] class labels
            noise_level: std of Gaussian noise as fraction
        """
        # Images from CIFAR10 with transforms.ToTensor() are already [0, 1]
        self.images = images.float()  # Just ensure float type, don't rescale
        self.labels = labels
        self.noise_level = noise_level

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        clean = self.images[idx]  # [3, 32, 32]

        # Add Gaussian noise
        noise = torch.randn_like(clean) * self.noise_level
        noisy = torch.clamp(clean + noise, 0, 1)

        return noisy, clean, self.labels[idx]


class SmallViT(nn.Module):
    """Smaller ViT for 32x32 images with Adaptive Sparse Attention."""

    def __init__(self, image_size=32, patch_size=4, embed_dim=128, depth=2, num_heads=4):
        """
        Args:
            image_size: 32 for CIFAR-10
            patch_size: 4 → 64 patches (8x8 grid)
            embed_dim: 128 (smaller than before)
            depth: 2 transformer blocks
            num_heads: 4 heads
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2

        # Patch embedding: [B, 3, 32, 32] → [B, 128, 8, 8] → [B, 64, 128]
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

        # Transformer blocks with adaptive sparse attention
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attn': AdaptiveAttentionMoE(embed_dim, num_heads, num_experts=3),
                'norm2': nn.LayerNorm(embed_dim),
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, 2 * embed_dim),
                    nn.GELU(),
                    nn.Linear(2 * embed_dim, embed_dim),
                ),
            })
            for _ in range(depth)
        ])

        # Patch unembedding: [B, 64, 128] → [B, 128, 8, 8] → [B, 3, 32, 32]
        self.patch_unembed = nn.ConvTranspose2d(embed_dim, 3, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Args:
            x: [B, 3, 32, 32] image

        Returns:
            [B, 3, 32, 32] denoised image
        """
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)  # [B, 128, 8, 8]
        x = x.flatten(2).transpose(1, 2)  # [B, 64, 128]

        # Add position embeddings
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x_norm = block['norm1'](x)
            x_attn = block['attn'](x_norm)
            x = x + x_attn

            x_norm = block['norm2'](x)
            x_ffn = block['ffn'](x_norm)
            x = x + x_ffn

        # Reshape for unembedding
        x = x.transpose(1, 2)  # [B, 128, 64]
        x = x.reshape(B, self.embed_dim, H // self.patch_size, W // self.patch_size)  # [B, 128, 8, 8]

        # Patch unembedding
        out = self.patch_unembed(x)  # [B, 3, 32, 32]
        out = torch.clamp(out, 0, 1)

        return out


def load_cifar10_data():
    """Load CIFAR-10 and prepare train/val/test split."""
    print("Downloading CIFAR-10...")

    transform = transforms.ToTensor()
    train_set = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_images = torch.stack([img for img, _ in train_set])  # [50000, 3, 32, 32]
    train_labels = torch.tensor(train_set.targets)
    test_images = torch.stack([img for img, _ in test_set])  # [10000, 3, 32, 32]
    test_labels = torch.tensor(test_set.targets)

    print(f"CIFAR-10 loaded: {train_images.shape} train, {test_images.shape} test")

    # Split train into train/val (40000/10000)
    train_images_split, val_images_split = random_split(
        range(len(train_images)),
        [40000, 10000]
    )
    train_idx = train_images_split.indices
    val_idx = val_images_split.indices

    train_imgs = train_images[train_idx]
    train_lbls = train_labels[train_idx]
    val_imgs = train_images[val_idx]
    val_lbls = train_labels[val_idx]

    return (train_imgs, train_lbls), (val_imgs, val_lbls), (test_images, test_labels)


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    count = 0

    for batch_idx, (noisy, clean, _) in enumerate(loader):
        noisy = noisy.to(device)
        clean = clean.to(device)

        denoised = model(noisy)
        loss = F.mse_loss(denoised, clean)

        # Load balancing loss
        lb_loss_total = torch.tensor(0.0, device=device)
        for block in model.blocks:
            lb_loss_total = lb_loss_total + block['attn'].get_load_balance_loss()

        total_loss_with_lb = loss + 0.01 * lb_loss_total

        optimizer.zero_grad()
        total_loss_with_lb.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        count += 1

        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{len(loader)}: loss={loss.item():.4f}")

    return total_loss / count


@torch.no_grad()
def evaluate(model, loader, device, compute_metrics=False):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    count = 0

    psnr_scores = []
    ssim_scores = []

    for noisy, clean, _ in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        denoised = model(noisy)
        loss = F.mse_loss(denoised, clean)
        total_loss += loss.item()
        count += 1

        if compute_metrics:
            for i in range(clean.shape[0]):
                c = clean[i].cpu().numpy().transpose(1, 2, 0)
                d = denoised[i].cpu().numpy().transpose(1, 2, 0)

                psnr = peak_signal_noise_ratio(c, d, data_range=1.0)
                ssim = structural_similarity(c, d, data_range=1.0, channel_axis=2)

                psnr_scores.append(psnr)
                ssim_scores.append(ssim)

    if compute_metrics:
        return total_loss / count, np.mean(psnr_scores), np.mean(ssim_scores), psnr_scores, ssim_scores
    else:
        return total_loss / count, None, None, None, None


def train_model(device='cuda', num_epochs=30, batch_size=64):
    """Train the denoising model."""
    print("=" * 70)
    print("PROPER DENOISING EXPERIMENT: Native 32x32 CIFAR-10")
    print("=" * 70)

    # Load data
    (train_imgs, train_lbls), (val_imgs, val_lbls), (test_imgs, test_lbls) = load_cifar10_data()

    train_dataset = NoisyImageDataset(train_imgs, train_lbls, noise_level=0.2)
    val_dataset = NoisyImageDataset(val_imgs, val_lbls, noise_level=0.2)
    test_dataset = NoisyImageDataset(test_imgs, test_lbls, noise_level=0.2)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model (smaller, appropriate for 32x32)
    print("\nCreating model...")
    model = SmallViT(image_size=32, patch_size=4, embed_dim=128, depth=2, num_heads=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {param_count:,} trainable parameters")
    print(f"Architecture: SmallViT (32x32 input, patch_size=4, 64 patches, depth=2)")

    # Training loop
    print("\nTraining...")
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        t0 = time.perf_counter()

        train_loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        val_loss, _, _, _, _ = evaluate(model, val_loader, device, compute_metrics=False)

        elapsed = time.perf_counter() - t0

        print(f"Epoch {epoch + 1}/{num_epochs}: train={train_loss:.4f}, val={val_loss:.4f}, time={elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'best_denoising_model_proper.pt')
            print(f"  → New best model saved")

    # Load best model
    print(f"\nLoading best model from epoch {best_epoch}")
    model.load_state_dict(torch.load('best_denoising_model_proper.pt'))

    # Test
    print("\nEvaluating on test set...")
    test_loss, test_psnr, test_ssim, psnr_scores, ssim_scores = evaluate(
        model, test_loader, device, compute_metrics=True
    )

    print("\n" + "=" * 70)
    print("TEST SET RESULTS (Native 32x32 CIFAR-10)")
    print("=" * 70)
    print(f"MSE Loss:        {test_loss:.4f}")
    print(f"PSNR (dB):       {test_psnr:.2f} ± {np.std(psnr_scores):.2f}")
    print(f"SSIM:            {test_ssim:.4f} ± {np.std(ssim_scores):.4f}")
    print(f"Images evaluated: {len(psnr_scores)}")
    print("=" * 70)

    return model, test_loader


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    model, test_loader = train_model(device=device, num_epochs=30, batch_size=64)
