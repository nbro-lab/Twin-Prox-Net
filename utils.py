"""Complex FFT, BART CFL I/O, and adaptive coil-combination utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def ifftshift(x: torch.Tensor, axes=None) -> torch.Tensor:
    return torch.fft.ifftshift(x, dim=axes)


def fftshift(x: torch.Tensor, axes=None) -> torch.Tensor:
    return torch.fft.fftshift(x, dim=axes)


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered orthonormal 2D inverse Fourier transform on the last two axes."""
    return torch.fft.fftshift(
        torch.fft.ifftn(
            torch.fft.ifftshift(x, dim=(-2, -1)),
            dim=(-2, -1),
            norm="ortho",
        ),
        dim=(-2, -1),
    )


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered orthonormal 2D Fourier transform on the last two axes."""
    return torch.fft.fftshift(
        torch.fft.fftn(
            torch.fft.ifftshift(x, dim=(-2, -1)),
            dim=(-2, -1),
            norm="ortho",
        ),
        dim=(-2, -1),
    )


def fft1c(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.fft(torch.fft.ifftshift(x, dim=dim), dim=dim, norm="ortho"),
        dim=dim,
    )


def ifft1c(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.ifft(torch.fft.ifftshift(x, dim=dim), dim=dim, norm="ortho"),
        dim=dim,
    )


def r2c(x: torch.Tensor) -> torch.Tensor:
    """Convert two real/imaginary channels on dimension 1 into a complex tensor."""
    real, imag = torch.chunk(x, 2, dim=1)
    return torch.complex(real, imag).squeeze(1)


def c2r(x: torch.Tensor) -> torch.Tensor:
    """Convert a complex tensor into two real/imaginary channels on dimension 1."""
    return torch.stack([x.real, x.imag], dim=1)


def ssos(x: torch.Tensor) -> torch.Tensor:
    real, imag = torch.chunk(x, 2, dim=1)
    return torch.sqrt(torch.sum(real.square() + imag.square(), dim=1))


def sos(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return torch.sqrt(torch.sum(torch.abs(x).square(), dim=dim))


def dot_batch(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return (x1 * x2).reshape(x1.shape[0], -1).sum(dim=1)


def readcfl(name: str | Path) -> np.ndarray:
    """Read a BART-style CFL/HDR pair using the basename without an extension."""
    basename = Path(name)
    with basename.with_suffix(".hdr").open("r") as header:
        header.readline()
        dims = [int(value) for value in header.readline().split()]

    total = int(np.prod(dims))
    cumulative = np.cumprod(dims)
    dims = dims[: np.searchsorted(cumulative, total) + 1]
    with basename.with_suffix(".cfl").open("rb") as data_file:
        data = np.fromfile(data_file, dtype=np.complex64, count=total)
    return data.reshape(dims, order="F")


def writecfl(name: str | Path, array: np.ndarray) -> None:
    """Write a NumPy array as a BART-style CFL/HDR pair."""
    basename = Path(name)
    basename.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(array)
    with basename.with_suffix(".hdr").open("w") as header:
        header.write("# Dimensions\n")
        header.write(" ".join(str(value) for value in array.shape) + "\n")
    with basename.with_suffix(".cfl").open("wb") as data_file:
        array.T.astype(np.complex64).tofile(data_file)


def estimate_adaptive_coil_weights(
    coil_images: torch.Tensor,
    ref_echo: int = 0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Estimate reference-echo-derived unit-norm complex coil weights.

    Parameters
    ----------
    coil_images
        Complex tensor shaped [batch, coil, echo, ny, nz].
    ref_echo
        Echo used as the phase reference, normally the first/highest-SNR image.

    Notes
    -----
    The weights are an adaptive phase-alignment interface for the learned
    proximal branch. They are not explicit sensitivity maps and are not used by
    the SNMs residual operator.
    """
    if coil_images.ndim != 5:
        raise ValueError(
            "coil_images must have shape [batch, coil, echo, ny, nz], "
            f"got {tuple(coil_images.shape)}"
        )
    ref_echo = min(max(int(ref_echo), 0), coil_images.shape[2] - 1)
    reference = coil_images[:, :, ref_echo]
    coil_energy = torch.sqrt(
        torch.sum(torch.abs(reference).square(), dim=1, keepdim=True)
    ).clamp_min(eps)
    return (reference / coil_energy).unsqueeze(2)


def adaptive_coil_combine(
    coil_images: torch.Tensor,
    coil_weights: torch.Tensor | None = None,
    ref_echo: int = 0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine complex multi-coil images and return the weights used."""
    if coil_weights is None:
        coil_weights = estimate_adaptive_coil_weights(coil_images, ref_echo, eps)
    if coil_weights.ndim != 5:
        raise ValueError(
            "coil_weights must have shape [batch, coil, 1, ny, nz], "
            f"got {tuple(coil_weights.shape)}"
        )
    denominator = torch.sum(
        torch.abs(coil_weights).square(), dim=1, keepdim=True
    ).clamp_min(eps)
    combined = torch.sum(
        coil_images * torch.conj(coil_weights), dim=1, keepdim=True
    ) / denominator
    return combined.squeeze(1), coil_weights


def adaptive_coil_expand(
    combined_images: torch.Tensor, coil_weights: torch.Tensor
) -> torch.Tensor:
    """Expand [batch, echo, ny, nz] data back to the multi-coil domain."""
    if combined_images.ndim != 4:
        raise ValueError(
            "combined_images must have shape [batch, echo, ny, nz], "
            f"got {tuple(combined_images.shape)}"
        )
    return combined_images.unsqueeze(1) * coil_weights


def adaptive_coil_combine_np(
    coil_images: np.ndarray,
    coil_axis: int = -2,
    echo_axis: int = -1,
    ref_echo: int = 0,
    eps: float = 1e-8,
) -> np.ndarray:
    """NumPy adaptive coil combination for arrays such as [x, y, z, coil, echo]."""
    array = np.asarray(coil_images)
    array = np.moveaxis(array, (coil_axis, echo_axis), (-2, -1))
    ref_echo = min(max(int(ref_echo), 0), array.shape[-1] - 1)
    reference = array[..., :, ref_echo]
    energy = np.sqrt(np.sum(np.abs(reference) ** 2, axis=-1, keepdims=True))
    weights = reference / np.maximum(energy, eps)
    numerator = np.sum(array * np.conj(weights[..., :, None]), axis=-2)
    denominator = np.sum(np.abs(weights) ** 2, axis=-1, keepdims=True)
    return numerator / np.maximum(denominator, eps)
