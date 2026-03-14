"""
Lightweight 3D CNN classifier for nnDetection FP reduction.

Architecture:
  - 3 ResBlocks (channels 32 → 64 → 128) with BatchNorm + ReLU
  - Each block: Conv3d → BN → ReLU → Conv3d → BN + skip → ReLU → MaxPool3d(2)
  - Global average pooling
  - FC head: Linear(128, fc_hidden) → ReLU → Dropout → Linear(fc_hidden, 1)
  - Output: raw logit (apply sigmoid externally or use BCEWithLogitsLoss)

Input: (B, 1, 64, 64, 64)
Output: (B, 1)
Parameters: ~2.5M (under 5M requirement)
"""

import torch
import torch.nn as nn


class ResBlock3D(nn.Module):
    """3D residual block with optional channel expansion."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Skip connection with 1x1 conv if channels change
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        out = self.pool(out)
        return out


class PatchClassifier3D(nn.Module):
    """3D patch classifier for lesion vs false positive discrimination."""

    def __init__(self, in_channels=1, base_channels=32, num_blocks=3,
                 fc_hidden=128, dropout=0.3):
        super().__init__()

        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )

        # ResBlocks with channel doubling
        channels = [base_channels * (2 ** i) for i in range(num_blocks)]
        # channels = [32, 64, 128]
        blocks = []
        in_ch = base_channels
        for out_ch in channels:
            blocks.append(ResBlock3D(in_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool3d(1)

        # FC classification head
        self.head = nn.Sequential(
            nn.Linear(channels[-1], fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 1, D, H, W) input patch

        Returns:
            (B, 1) raw logit
        """
        x = self.stem(x)
        x = self.blocks(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

    def predict_proba(self, x):
        """Return sigmoid probability."""
        return torch.sigmoid(self.forward(x))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = PatchClassifier3D(in_channels=1, base_channels=32, num_blocks=3,
                              fc_hidden=128, dropout=0.3)
    print(f"Total parameters: {count_parameters(model):,}")
    # Test forward pass
    x = torch.randn(2, 1, 64, 64, 64)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    print(f"Output values: {out.detach().numpy().flatten()}")
    prob = model.predict_proba(x)
    print(f"Probabilities: {prob.detach().numpy().flatten()}")
