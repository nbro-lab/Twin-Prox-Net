"""Run Stage II MDI-Net inference and deterministic T1/T2* mapping."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mdi_physics import MDIPhysicsLayer
from train_mapping import make_brain_mask
from utils import writecfl


class MappingInferenceDataset(Dataset):
    def __init__(self, data_dir: str, volume: str, reference_available: bool) -> None:
        root = Path(data_dir)
        recon_path = root / f"{volume}_recon.npy"
        if not recon_path.exists():
            raise FileNotFoundError(recon_path)
        self.recon = np.load(recon_path, mmap_mode="r")
        self.reference = None
        if reference_available:
            reference_path = root / f"{volume}_label.npy"
            if not reference_path.exists():
                raise FileNotFoundError(reference_path)
            self.reference = np.load(reference_path, mmap_mode="r")

    def __len__(self) -> int:
        return int(self.recon.shape[0])

    def __getitem__(self, index: int):
        recon = torch.as_tensor(np.array(self.recon[index]), dtype=torch.complex64)
        if self.reference is None:
            return recon
        reference = torch.as_tensor(
            np.array(self.reference[index]), dtype=torch.complex64
        )
        return recon, reference


def parse_volumes(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def mask_from_reconstruction(images: torch.Tensor) -> torch.Tensor:
    sos_image = torch.sqrt(torch.sum(torch.abs(images[:, :, 0]) ** 2, dim=1))
    robust_max = sos_image.flatten(1).quantile(0.99, dim=1)[:, None, None]
    return (sos_image > 0.12 * robust_max.clamp_min(1e-8)).float()


def nrmse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    difference = (prediction - target) * mask
    denominator = torch.sqrt(torch.mean((target * mask) ** 2)).clamp_min(1e-8)
    return float((torch.sqrt(torch.mean(difference**2)) / denominator).cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=str, default="./data/Mapping_Test_Data_12Echo")
    parser.add_argument("--volumes", type=str, default="vol1")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./results/mapping")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--reference_available",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use paired <volume>_label.npy files and report NRMSE.",
    )
    args = parser.parse_args()

    device = torch.device(
        "cpu" if args.device.startswith("cuda") and not torch.cuda.is_available() else args.device
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    from mapping_checkpoint import load_mapping_model

    model, _ = load_mapping_model(args.checkpoint, device)
    physics = MDIPhysicsLayer().to(device)

    for volume in parse_volumes(args.volumes):
        dataset = MappingInferenceDataset(
            args.data_dir, volume, args.reference_available
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        t1_sections = []
        t2_sections = []
        t1_reference_sections = []
        t2_reference_sections = []
        masks = []

        with torch.no_grad():
            for batch in loader:
                if args.reference_available:
                    recon, reference = batch
                    reference = reference.to(device)
                    mask = make_brain_mask(reference)
                else:
                    recon = batch
                    reference = None
                    mask = mask_from_reconstruction(recon.to(device))

                recon = recon.to(device)
                delta_s, ratio_r = model(recon, mask)
                t1, t2star = physics(delta_s, ratio_r, mask)
                t1_sections.append(t1.cpu())
                t2_sections.append(t2star.cpu())
                masks.append(mask.cpu())

                if reference is not None:
                    from train_mapping import analytical_ratios

                    delta_ref, ratio_ref = analytical_ratios(reference, mask)
                    t1_ref, t2_ref = physics(delta_ref, ratio_ref, mask)
                    t1_reference_sections.append(t1_ref.cpu())
                    t2_reference_sections.append(t2_ref.cpu())

        t1 = torch.cat(t1_sections).numpy() * 1000.0
        t2star = torch.cat(t2_sections).numpy() * 1000.0
        writecfl(str(save_dir / f"{volume}_T1_ms"), t1)
        writecfl(str(save_dir / f"{volume}_T2star_ms"), t2star)

        if args.reference_available:
            mask = torch.cat(masks)
            t1_tensor = torch.from_numpy(t1 / 1000.0)
            t2_tensor = torch.from_numpy(t2star / 1000.0)
            t1_ref = torch.cat(t1_reference_sections)
            t2_ref = torch.cat(t2_reference_sections)
            print(
                f"{volume}: T1 NRMSE={nrmse(t1_tensor, t1_ref, mask):.4f}, "
                f"T2* NRMSE={nrmse(t2_tensor, t2_ref, mask):.4f}"
            )


if __name__ == "__main__":
    main()
