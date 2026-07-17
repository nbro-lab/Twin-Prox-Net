"""Differentiable structural similarity (SSIM) loss for 2D images."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= window_size // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _create_window(window_size: int, channel: int, sigma: float = 1.5) -> torch.Tensor:
    kernel_1d = _gaussian(window_size, sigma)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(channel, 1, window_size, window_size).contiguous()


class SSIM(nn.Module):
    """Compute mean SSIM for tensors shaped ``[batch, channel, height, width]``."""

    def __init__(
        self,
        data_range: float = 1.0,
        size_average: bool = True,
        channel: int = 1,
        window_size: int = 11,
        sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
    ) -> None:
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        self.data_range = float(data_range)
        self.size_average = bool(size_average)
        self.channel = int(channel)
        self.window_size = int(window_size)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.register_buffer(
            "window", _create_window(self.window_size, self.channel, sigma), persistent=False
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape != y.shape:
            raise ValueError(f"SSIM inputs must have identical shapes, got {x.shape} and {y.shape}")
        if x.ndim != 4:
            raise ValueError(f"SSIM expects 4D tensors, got {x.ndim} dimensions")
        if x.shape[1] != self.channel:
            raise ValueError(
                f"SSIM was initialized for {self.channel} channels, got {x.shape[1]}"
            )

        window = self.window.to(device=x.device, dtype=x.dtype)
        padding = self.window_size // 2

        mu_x = F.conv2d(x, window, padding=padding, groups=self.channel)
        mu_y = F.conv2d(y, window, padding=padding, groups=self.channel)

        mu_x_sq = mu_x.square()
        mu_y_sq = mu_y.square()
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(x * x, window, padding=padding, groups=self.channel) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, window, padding=padding, groups=self.channel) - mu_y_sq
        sigma_xy = F.conv2d(x * y, window, padding=padding, groups=self.channel) - mu_xy

        c1 = (self.k1 * self.data_range) ** 2
        c2 = (self.k2 * self.data_range) ** 2
        score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2) + 1e-12
        )

        if self.size_average:
            return score.mean()
        return score.mean(dim=(1, 2, 3))
