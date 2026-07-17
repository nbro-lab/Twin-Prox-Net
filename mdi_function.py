"""Analytical complex ratios used by the MDI-Net stage."""

from __future__ import annotations

import torch


def _validate_retained_images(images: torch.Tensor) -> None:
    if images.ndim != 6:
        raise ValueError(
            "Expected retained images shaped [batch, coil, flip_angle, echo, ny, nz], "
            f"got {tuple(images.shape)}"
        )
    if images.shape[2] != 2:
        raise ValueError(f"Expected two flip-angle branches, got {images.shape[2]}")
    if images.shape[3] < 2:
        raise ValueError("At least two retained echoes are required")


def calculate_delta_S(images: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute the complex optimal matched-filter echo-decay ratio.

    The transient first image from each six-image FA sequence must be discarded
    before calling this function. The remaining five images are then interpreted
    as ``t=1,...,N_e-1`` in manuscript Eq. (11).
    """
    _validate_retained_images(images)
    numerator = torch.sum(
        torch.conj(images[:, :, :, :-1]) * images[:, :, :, 1:],
        dim=(1, 2, 3),
    )
    denominator = torch.sum(
        torch.abs(images[:, :, :, :-1]) ** 2,
        dim=(1, 2, 3),
    )
    return numerator / (denominator + eps)


def calculate_R2(images: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute the complex flip-angle boundary ratio from manuscript Eq. (11)."""
    _validate_retained_images(images)
    numerator = torch.sum(
        torch.conj(images[:, :, 1]) * images[:, :, 0],
        dim=(1, 2),
    )
    denominator = torch.sum(
        torch.abs(images[:, :, 1]) ** 2,
        dim=(1, 2),
    )
    return numerator / (denominator + eps)
