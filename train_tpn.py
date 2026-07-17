"""Train Stage I of Twin-Prox Net.

This release follows the manuscript configuration:
- 10-iteration CG-SNM initialization;
- 15 learnable unrolled iterations by default;
- seven readout sections per input stack, supervised on the central three;
- adaptive complex coil combination/expansion without explicit CSMs;
- composite complex, phase, logarithmic k-space, SSIM, and joint-TV loss.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from einops import rearrange
from scipy.io import loadmat
from torch.utils.data import ConcatDataset, DataLoader

from dataset import MTP_dataset_echos_mslice, collate_fn_keep_12echo
from model_cginit import Deep_snms_echos_mslice
from ssim import SSIM
from utils import adaptive_coil_combine, fft2c, ifft2c, sos


def parse_volume_list(value: str) -> list[str]:
    """Parse comma-separated volume names or a compact integer range, e.g. 1-24."""
    value = value.strip()
    if not value:
        return []
    if "-" in value and "," not in value and value.replace("-", "").isdigit():
        start, end = (int(item) for item in value.split("-", maxsplit=1))
        return [f"vol{i}" for i in range(start, end + 1)]
    return [item.strip() for item in value.split(",") if item.strip()]


def get_mask(mask_name: str, mask_dir: str) -> torch.Tensor:
    path = Path(mask_dir) / f"{mask_name}.mat"
    if not path.exists():
        raise FileNotFoundError(f"Sampling mask not found: {path}")
    data = loadmat(path)
    if "mask" not in data:
        raise KeyError(f"Expected variable 'mask' in {path}")
    return torch.as_tensor(data["mask"], dtype=torch.complex64)


def joint_tv_loss(images: torch.Tensor) -> torch.Tensor:
    """Joint spatial and echo-wise TV on complex coil-combined images.

    Parameters
    ----------
    images : torch.Tensor
        Complex tensor shaped [batch, section, echo, ny, nz].
    """
    magnitude = torch.abs(images)
    tv_section = torch.abs(magnitude[:, 1:] - magnitude[:, :-1]).mean()
    tv_y = torch.abs(magnitude[..., 1:, :] - magnitude[..., :-1, :]).mean()
    tv_z = torch.abs(magnitude[..., :, 1:] - magnitude[..., :, :-1]).mean()
    if magnitude.shape[2] > 1:
        tv_echo = torch.abs(magnitude[:, :, 1:] - magnitude[:, :, :-1]).mean()
    else:
        tv_echo = magnitude.new_zeros(())
    return tv_section + tv_y + tv_z + tv_echo


def _central_sections(x: torch.Tensor, count: int = 3) -> torch.Tensor:
    if x.shape[1] < count:
        raise ValueError(
            f"At least {count} readout sections are required, got {x.shape[1]}"
        )
    start = (x.shape[1] - count) // 2
    return x[:, start : start + count]


def compute_composite_loss(
    model: torch.nn.Module,
    sample: Dict[str, torch.Tensor],
    mask: torch.Tensor,
    device: torch.device,
    ssim_metric: SSIM,
    valid_sections: int = 3,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Run reconstruction and compute the manuscript composite loss."""
    ksp = sample["ksp"].to(device, non_blocking=True)
    null_2d = sample["Null_2d"].to(device, non_blocking=True)
    batch_size, n_sections, n_coils, n_echoes, ny, nz = ksp.shape

    # One scale factor per volume stack, derived from the fully sampled reference.
    label_coils_unscaled = ifft2c(ksp)
    scale_image = sos(label_coils_unscaled, dim=2)
    scale_factor = scale_image.reshape(batch_size, -1).amax(dim=1).clamp_min(1e-8)
    scale_view = scale_factor.view(batch_size, 1, 1, 1, 1, 1)

    kspace_reference = ksp / scale_view
    label_coils = ifft2c(kspace_reference)
    acquired_kspace = kspace_reference * mask
    x0 = torch.zeros_like(acquired_kspace)

    recon_kspace, recon_coils, _ = model(acquired_kspace, null_2d, mask, x0)

    # Use one adaptive phase reference for both reconstruction and target so the
    # complex-domain supervision is not affected by arbitrary coil-combination phase.
    label_flat = rearrange(
        label_coils, "b s c e y z -> (b s) c e y z"
    )
    recon_flat = rearrange(
        recon_coils, "b s c e y z -> (b s) c e y z"
    )
    label_combined_flat, adaptive_weights = adaptive_coil_combine(
        label_flat, ref_echo=0
    )
    recon_combined_flat, _ = adaptive_coil_combine(
        recon_flat, coil_weights=adaptive_weights
    )

    label_combined = rearrange(
        label_combined_flat,
        "(b s) e y z -> b s e y z",
        b=batch_size,
        s=n_sections,
    )
    recon_combined = rearrange(
        recon_combined_flat,
        "(b s) e y z -> b s e y z",
        b=batch_size,
        s=n_sections,
    )

    recon_mid = _central_sections(recon_combined, valid_sections)
    label_mid = _central_sections(label_combined, valid_sections)

    # L_cplx: complex L1 loss.
    loss_cplx = torch.mean(torch.abs(recon_mid - label_mid))

    # L_phase: magnitude-guided cosine phase loss. Magnitude weights are
    # normalized independently for every section and echo.
    mag_weight = torch.abs(label_mid).detach()
    mag_weight = mag_weight / mag_weight.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    phase_error = 1.0 - torch.cos(torch.angle(recon_mid) - torch.angle(label_mid))
    loss_phase = torch.mean(mag_weight * phase_error)

    # L_ksp: logarithmic k-space penalty computed from the complex coil-combined
    # reconstruction and reference, as defined in the manuscript.
    recon_kspace_combined = fft2c(recon_mid)
    label_kspace_combined = fft2c(label_mid)
    loss_ksp = torch.mean(
        torch.abs(
            torch.log1p(torch.abs(recon_kspace_combined))
            - torch.log1p(torch.abs(label_kspace_combined))
        )
    )

    # L_SSIM: magnitude SSIM, normalized to [0, 1] per section/echo.
    recon_mag = torch.abs(recon_mid).reshape(-1, 1, ny, nz)
    label_mag = torch.abs(label_mid).reshape(-1, 1, ny, nz)
    norm = label_mag.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    recon_mag = (recon_mag / norm).clamp(0.0, 1.0)
    label_mag = (label_mag / norm).clamp(0.0, 1.0)
    loss_ssim = 1.0 - ssim_metric(recon_mag, label_mag)

    # L_JointTV: full-resolution spatial and echo-wise regularization.
    loss_joint_tv = joint_tv_loss(recon_mid)

    total = (
        1.0 * loss_cplx
        + 0.5 * loss_phase
        + 0.1 * loss_ksp
        + 0.05 * loss_ssim
        + 0.1 * loss_joint_tv
    )

    components = {
        "total": total,
        "cplx": loss_cplx,
        "phase": loss_phase,
        "ksp": loss_ksp,
        "ssim": loss_ssim,
        "joint_tv": loss_joint_tv,
    }
    return total, components


def build_dataset(
    data_root: str,
    split: str,
    volumes: Iterable[str],
    n_echoes: int,
    acceleration: str,
    n_slices: int,
    stride: int,
) -> ConcatDataset:
    datasets = [
        MTP_dataset_echos_mslice(
            data_root,
            split,
            volume,
            n_echoes,
            acceleration,
            nslice=n_slices,
            stride=stride,
        )
        for volume in volumes
    ]
    if not datasets:
        raise ValueError(f"No volumes were specified for split '{split}'")
    return ConcatDataset(datasets)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(args),
    }
    torch.save(state, path)
    torch.save(model.state_dict(), path.with_name(f"net_params_{epoch + 1}.pkl"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=str, default="./data/MTP_datasets_total")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--valid_split", type=str, default="valid")
    parser.add_argument("--train_volumes", type=str, default="1-24")
    parser.add_argument("--valid_volumes", type=str, default="25-28")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--experiment_name", type=str, default="twin_prox_net")
    parser.add_argument("--mask_dir", type=str, default="./mask")
    parser.add_argument("--mask", type=str, default="caipi_9x")
    parser.add_argument("--acc", type=str, default="9x")
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--niter", type=int, default=15)
    parser.add_argument("--cg_iterations", type=int, default=10)
    parser.add_argument("--nEcho", type=int, default=12)
    parser.add_argument("--nSlice", type=int, default=7)
    parser.add_argument("--valid_sections", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--validate_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=1)
    args = parser.parse_args()

    if args.nSlice != 7:
        raise ValueError("The released manuscript configuration uses nSlice=7")
    if args.valid_sections != 3:
        raise ValueError("The released manuscript configuration supervises 3 central sections")

    device = torch.device(
        "cpu" if args.device.startswith("cuda") and not torch.cuda.is_available() else args.device
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    log_path = output_dir / f"train_{dt.datetime.now():%Y%m%d-%H%M%S}.log"

    train_volumes = parse_volume_list(args.train_volumes)
    valid_volumes = parse_volume_list(args.valid_volumes)
    train_set = build_dataset(
        args.data_root,
        args.train_split,
        train_volumes,
        args.nEcho,
        args.acc,
        args.nSlice,
        stride=1,
    )
    valid_set = None
    if valid_volumes:
        valid_set = build_dataset(
            args.data_root,
            args.valid_split,
            valid_volumes,
            args.nEcho,
            args.acc,
            args.nSlice,
            stride=args.nSlice,
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn_keep_12echo,
    )
    valid_loader = None
    if valid_set is not None:
        valid_loader = DataLoader(
            valid_set,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_fn_keep_12echo,
        )

    model = Deep_snms_echos_mslice(
        in_channel=2,
        niter=args.niter,
        num_echoes=args.nEcho,
        cg_iterations=args.cg_iterations,
    ).to(device)
    print(f"Trainable reconstruction parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    ssim_metric = SSIM(data_range=1.0, size_average=True, channel=1).to(device)
    mask = get_mask(args.mask, args.mask_dir).to(device)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = int(checkpoint["epoch"]) + 1
        else:
            model.load_state_dict(checkpoint)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.num_epoch):
        model.train()
        epoch_components: Dict[str, list[float]] = {
            key: [] for key in ("total", "cplx", "phase", "ksp", "ssim", "joint_tv")
        }
        epoch_start = time.time()

        for step, sample in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            loss, components = compute_composite_loss(
                model,
                sample,
                mask,
                device,
                ssim_metric,
                valid_sections=args.valid_sections,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            for name, value in components.items():
                epoch_components[name].append(float(value.detach().cpu()))

            if step % 20 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch + 1:03d}/{args.num_epoch} "
                    f"Step {step:04d}/{len(train_loader)} "
                    f"Loss {components['total'].item():.5f} "
                    f"Cplx {components['cplx'].item():.5f} "
                    f"Phase {components['phase'].item():.5f} "
                    f"Ksp {components['ksp'].item():.5f} "
                    f"SSIM {components['ssim'].item():.5f} "
                    f"JTV {components['joint_tv'].item():.5f}"
                )

        scheduler.step()
        train_means = {name: float(np.mean(values)) for name, values in epoch_components.items()}

        valid_loss = float("nan")
        if valid_loader is not None and (epoch + 1) % args.validate_every == 0:
            model.eval()
            validation_values = []
            with torch.no_grad():
                for sample in valid_loader:
                    value, _ = compute_composite_loss(
                        model,
                        sample,
                        mask,
                        device,
                        ssim_metric,
                        valid_sections=args.valid_sections,
                    )
                    validation_values.append(float(value.cpu()))
            valid_loss = float(np.mean(validation_values))

        elapsed = time.time() - epoch_start
        summary = (
            f"epoch={epoch + 1} train_loss={train_means['total']:.6f} "
            f"valid_loss={valid_loss:.6f} lr={optimizer.param_groups[0]['lr']:.8f} "
            f"seconds={elapsed:.1f}"
        )
        print(summary)
        with log_path.open("a") as stream:
            stream.write(summary + "\n")

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch + 1}.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                args,
            )


if __name__ == "__main__":
    main()
