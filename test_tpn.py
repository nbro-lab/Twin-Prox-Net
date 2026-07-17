"""Run Stage I Twin-Prox Net reconstruction on retrospective or prospective data."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from dataset import MTP_dataset_echos_mslice
from model_cginit import Deep_snms_echos_mslice
from utils import adaptive_coil_combine, ifft2c, sos, writecfl


def get_mask(mask_name: str, mask_dir: str) -> torch.Tensor:
    path = Path(mask_dir) / f"{mask_name}.mat"
    if not path.exists():
        raise FileNotFoundError(path)
    data = loadmat(path)
    if "mask" not in data:
        raise KeyError(f"Expected variable 'mask' in {path}")
    return torch.as_tensor(data["mask"], dtype=torch.complex64)


def parse_volumes(value: str) -> list[str]:
    value = value.strip()
    if "-" in value and "," not in value and value.replace("-", "").isdigit():
        start, end = (int(item) for item in value.split("-", maxsplit=1))
        return [f"vol{i}" for i in range(start, end + 1)]
    return [item.strip() for item in value.split(",") if item.strip()]


def complex_nrmse(recon: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.sqrt(np.mean(np.abs(reference) ** 2, dtype=np.float64))
    return float(
        np.sqrt(np.mean(np.abs(recon - reference) ** 2, dtype=np.float64))
        / (denominator + 1e-8)
    )


def run_overlapping_inference(
    model: torch.nn.Module,
    kspace_volume: torch.Tensor,
    null_volume: torch.Tensor,
    mask: torch.Tensor,
    scale_factor: torch.Tensor,
    device: torch.device,
    n_slices: int = 7,
    output_sections: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Seven-section input with the central three sections retained per window."""
    nx, n_coils, n_echoes, ny, nz = kspace_volume.shape
    left_context = (n_slices - output_sections) // 2
    right_context = n_slices - output_sections - left_context

    kspace_padded = torch.cat(
        [
            kspace_volume[:1].expand(left_context, -1, -1, -1, -1),
            kspace_volume,
            kspace_volume[-1:].expand(right_context + output_sections - 1, -1, -1, -1, -1),
        ],
        dim=0,
    )
    null_padded = torch.cat(
        [
            null_volume[:1].expand(left_context, -1, -1, -1, -1),
            null_volume,
            null_volume[-1:].expand(right_context + output_sections - 1, -1, -1, -1, -1),
        ],
        dim=0,
    )

    reconstruction = torch.zeros_like(kspace_volume)
    cg_initialization = torch.zeros_like(kspace_volume)

    model.eval()
    with torch.no_grad():
        for start in range(0, nx, output_sections):
            kspace_chunk = kspace_padded[start : start + n_slices][None].to(device)
            null_chunk = null_padded[start : start + n_slices][None].to(device)
            normalized = kspace_chunk / scale_factor
            acquired = normalized * mask
            x0 = torch.zeros_like(acquired)

            _, recon_chunk, cg_chunk = model(acquired, null_chunk, mask, x0)
            recon_chunk = recon_chunk * scale_factor
            cg_chunk = cg_chunk * scale_factor

            valid = min(output_sections, nx - start)
            reconstruction[start : start + valid] = recon_chunk[
                0, left_context : left_context + valid
            ].cpu()
            cg_initialization[start : start + valid] = cg_chunk[
                0, left_context : left_context + valid
            ].cpu()

    return reconstruction, cg_initialization


def report_metrics(recon: np.ndarray, reference: np.ndarray) -> None:
    """Report echo-wise metrics for arrays shaped [section, echo, ny, nz]."""
    for echo in range(recon.shape[1]):
        echo_recon = np.abs(recon[:, echo])
        echo_reference = np.abs(reference[:, echo])
        data_range = float(echo_reference.max() - echo_reference.min()) or 1.0
        psnr = peak_signal_noise_ratio(
            echo_reference, echo_recon, data_range=data_range
        )
        # skimage SSIM is evaluated section-by-section for 3D stacks.
        ssim_values = [
            structural_similarity(
                echo_reference[index], echo_recon[index], data_range=data_range
            )
            for index in range(echo_reference.shape[0])
        ]
        print(
            f"Echo {echo + 1:02d}: PSNR={psnr:.2f} dB, "
            f"SSIM={np.mean(ssim_values):.4f}, "
            f"NRMSE={complex_nrmse(recon[:, echo], reference[:, echo]):.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=str, default="./data/MTP_datasets_total")
    parser.add_argument("--data_split", type=str, default="test")
    parser.add_argument("--volumes", type=str, default="29-33")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/reconstruction")
    parser.add_argument("--mask_dir", type=str, default="./mask")
    parser.add_argument("--mask", type=str, default="caipi_9x")
    parser.add_argument("--acc", type=str, default="9x")
    parser.add_argument("--niter", type=int, default=15)
    parser.add_argument("--cg_iterations", type=int, default=10)
    parser.add_argument("--nEcho", type=int, default=12)
    parser.add_argument("--nSlice", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--reference_available",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat input ksp as fully sampled reference and report metrics.",
    )
    args = parser.parse_args()

    if args.nSlice != 7:
        raise ValueError("The manuscript configuration uses seven input sections")

    device = torch.device(
        "cpu" if args.device.startswith("cuda") and not torch.cuda.is_available() else args.device
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask = get_mask(args.mask, args.mask_dir).to(device)

    model = Deep_snms_echos_mslice(
        in_channel=2,
        niter=args.niter,
        num_echoes=args.nEcho,
        cg_iterations=args.cg_iterations,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )

    for volume in parse_volumes(args.volumes):
        start_time = time.time()
        dataset = MTP_dataset_echos_mslice(
            args.data_root,
            args.data_split,
            volume,
            args.nEcho,
            args.acc,
            nslice=args.nSlice,
            stride=args.nSlice,
        )
        kspace_volume = torch.as_tensor(dataset.ksp, dtype=torch.complex64)
        null_sections = []
        for section in range(dataset.nx):
            null = torch.as_tensor(dataset.Null_2d[section], dtype=torch.complex64)
            null_sections.append(dataset.Null_to_image(null))
        null_volume = torch.stack(null_sections)

        observed_image = ifft2c(kspace_volume * mask.cpu())
        if args.reference_available:
            reference_coils = ifft2c(kspace_volume)
            scale_factor = sos(reference_coils, dim=1).amax().clamp_min(1e-8).to(device)
        else:
            reference_coils = None
            scale_factor = sos(observed_image, dim=1).amax().clamp_min(1e-8).to(device)

        recon_coils, cg_coils = run_overlapping_inference(
            model,
            kspace_volume,
            null_volume,
            mask,
            scale_factor,
            device,
            n_slices=args.nSlice,
            output_sections=3,
        )

        if reference_coils is not None:
            reference_combined, weights = adaptive_coil_combine(
                reference_coils, ref_echo=0
            )
            recon_combined, _ = adaptive_coil_combine(
                recon_coils, coil_weights=weights
            )
            report_metrics(
                recon_combined.numpy(), reference_combined.numpy()
            )
        else:
            recon_combined, _ = adaptive_coil_combine(recon_coils, ref_echo=0)

        # CFL output uses [section, ny, nz, coil, echo] for multi-coil data and
        # [section, ny, nz, echo] for the adaptive coil-combined sequence.
        recon_multicoil = np.transpose(recon_coils.numpy(), (0, 3, 4, 1, 2))
        cg_multicoil = np.transpose(cg_coils.numpy(), (0, 3, 4, 1, 2))
        recon_combined_np = np.transpose(recon_combined.numpy(), (0, 2, 3, 1))
        writecfl(str(output_dir / f"{volume}_recon_multicoil"), recon_multicoil)
        writecfl(str(output_dir / f"{volume}_recon_adaptivecc"), recon_combined_np)
        writecfl(str(output_dir / f"{volume}_cg_snm_init"), cg_multicoil)
        print(f"{volume} completed in {time.time() - start_time:.1f} s")


if __name__ == "__main__":
    main()
