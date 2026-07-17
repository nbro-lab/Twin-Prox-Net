"""Complex-domain MDI-Net ratio refinement modules."""

from __future__ import annotations

import torch
import torch.nn as nn

from mdi_function import calculate_R2, calculate_delta_S


class PointwiseRatioHead(nn.Module):
    """Pointwise mapping head used after spatial ratio correction."""

    def __init__(self, channels: int = 2, features: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, features, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, channels, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class SpatialRatioCorrector(nn.Module):
    """Full-resolution complex residual corrector for one analytical ratio map."""

    def __init__(self, channels: int = 2, features: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, channels, kernel_size=3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.net(inputs)


class Learned_S2(nn.Module):
    """Refine the integrated complex echo-decay ratio from manuscript Eq. (11)."""

    def __init__(self) -> None:
        super().__init__()
        self.spatial_corrector = SpatialRatioCorrector()
        self.mapping_head = PointwiseRatioHead()

    def forward(self, retained_images: torch.Tensor, brain_mask: torch.Tensor) -> torch.Tensor:
        delta_s = calculate_delta_S(retained_images)
        inputs = torch.stack([delta_s.real, delta_s.imag], dim=1)
        outputs = self.mapping_head(self.spatial_corrector(inputs))
        real, imag = torch.chunk(outputs, 2, dim=1)
        return torch.complex(real, imag).squeeze(1) * brain_mask


class Learned_R2(nn.Module):
    """Refine the integrated complex flip-angle ratio from manuscript Eq. (11)."""

    def __init__(self) -> None:
        super().__init__()
        self.spatial_corrector = SpatialRatioCorrector()
        self.mapping_head = PointwiseRatioHead()

    def forward(self, retained_images: torch.Tensor, brain_mask: torch.Tensor) -> torch.Tensor:
        ratio_r = calculate_R2(retained_images)
        inputs = torch.stack([ratio_r.real, ratio_r.imag], dim=1)
        outputs = self.mapping_head(self.spatial_corrector(inputs))
        real, imag = torch.chunk(outputs, 2, dim=1)
        return torch.complex(real, imag).squeeze(1) * brain_mask


# Backward-compatible aliases used by earlier internal scripts.
Learned_S = Learned_S2
Learned_R = Learned_R2
