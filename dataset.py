"""Dataset loaders for multi-section, multi-echo MULTIPLEX reconstruction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import ifft1c, readcfl


class MTP_dataset_echos_mslice(Dataset):
    """Load multi-echo k-space and spatial nulling maps for adjacent sections.

    Expected on-disk layout::

        <root>/<split>/<volume>/Echo_1/{ksp,Null_2d}.{cfl,hdr}
        <root>/<split>/<volume>/Echo_2/ksp.{cfl,hdr}
        ...

    The same calibrated ``Null_2d`` object from ``Echo_1`` is used for all 12
    contrast-encoded images, matching the released reconstruction pipeline.
    Explicit coil sensitivity maps are not loaded.
    """

    def __init__(
        self,
        root: str,
        flag: str,
        vol: str,
        echo: int,
        acc: str,
        nslice: int,
        stride: int,
    ) -> None:
        super().__init__()
        del acc  # Kept in the public API for backward-compatible command lines.
        self.flag = flag
        self.nslice = int(nslice)
        self.stride = int(stride)
        if self.nslice <= 0 or self.stride <= 0:
            raise ValueError("nslice and stride must be positive")

        volume_root = Path(root) / flag / vol
        if not volume_root.exists():
            raise FileNotFoundError(volume_root)

        echoes = []
        for echo_index in range(echo):
            echo_root = volume_root / f"Echo_{echo_index + 1}"
            if not echo_root.exists():
                raise FileNotFoundError(echo_root)
            kspace = readcfl(echo_root / "ksp")
            if kspace.ndim != 4:
                raise ValueError(
                    f"Expected ksp [section, ky, kz, coil], got {kspace.shape}"
                )
            echoes.append(np.transpose(kspace, (0, 3, 1, 2)))

        self.ksp = np.stack(echoes, axis=2)  # [section, coil, echo, ky, kz]

        null_maps = readcfl(volume_root / "Echo_1" / "Null_2d")
        if null_maps.ndim != 5:
            raise ValueError(
                "Expected Null_2d [section, kernel_y, kernel_z, coil_in, coil_out], "
                f"got {null_maps.shape}"
            )
        self.Null_2d = np.transpose(
            null_maps, (0, 4, 3, 1, 2)
        )  # [section, coil_out, coil_in, kernel_y, kernel_z]

        self.nx, self.nc, self.ne, self.ny, self.nz = self.ksp.shape
        self.kernel_size_y = self.Null_2d.shape[-2]
        self.kernel_size_z = self.Null_2d.shape[-1]

        if self.nx < self.nslice:
            raise ValueError(
                f"Volume has {self.nx} sections but nslice={self.nslice} was requested"
            )
        if flag in {"valid", "test"} and self.stride != self.nslice:
            # DataLoader-based validation/testing uses non-overlapping stacks.
            # test_tpn.py accesses the complete volume directly for 7-in/3-out inference.
            raise ValueError("For valid/test datasets, stride must equal nslice")

    def __len__(self) -> int:
        if self.flag == "train":
            return (self.nx - self.nslice) // self.stride + 1
        return (self.nx - self.nslice) // self.stride + 1

    def __getitem__(self, index: int):
        start = index * self.stride
        end = start + self.nslice
        if end > self.nx:
            raise IndexError(index)

        kspace = torch.as_tensor(self.ksp[start:end], dtype=torch.complex64)
        null_images = [
            self.Null_to_image(
                torch.as_tensor(self.Null_2d[section], dtype=torch.complex64)
            )
            for section in range(start, end)
        ]
        return kspace, torch.stack(null_images)

    def Null_to_image(self, null_2d: torch.Tensor) -> torch.Tensor:
        """Zero-pad calibrated kernels and transform them to the image domain."""
        pad_y_total = self.ny - self.kernel_size_y
        pad_z_total = self.nz - self.kernel_size_z
        if pad_y_total < 0 or pad_z_total < 0:
            raise ValueError("Null_2d kernel is larger than the reconstructed matrix")

        pad_y_left = (pad_y_total + 1) // 2
        pad_y_right = pad_y_total - pad_y_left
        pad_z_left = (pad_z_total + 1) // 2
        pad_z_right = pad_z_total - pad_z_left
        padded = torch.nn.functional.pad(
            null_2d,
            (pad_z_left, pad_z_right, pad_y_left, pad_y_right),
        )
        padded = padded * torch.sqrt(
            torch.tensor(self.ny * self.nz, dtype=torch.float32, device=padded.device)
        )
        return ifft1c(ifft1c(padded, dim=-2), dim=-1)


def collate_fn_keep_12echo(batch):
    """Stack adjacent-section samples into a batch dictionary."""
    kspace = torch.stack([sample[0] for sample in batch], dim=0)
    null_maps = torch.stack([sample[1] for sample in batch], dim=0)
    return {"ksp": kspace, "Null_2d": null_maps}
