"""Train the complex-domain MDI-Net quantitative mapping stage."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mdi_function import calculate_R2, calculate_delta_S
from mdi_physics import MDIPhysicsLayer
from model_deep_snms import Learned_R2, Learned_S2


class MappingDataset12Echo(Dataset):
    """Memory-mapped paired 12-echo reconstruction/reference dataset."""

    def __init__(self, data_dir: str, volume_names: list[str]) -> None:
        root = Path(data_dir)
        self.recon_data = []
        self.label_data = []
        for volume in volume_names:
            recon_path = root / f"{volume}_recon.npy"
            label_path = root / f"{volume}_label.npy"
            if not recon_path.exists() or not label_path.exists():
                raise FileNotFoundError(
                    f"Missing paired mapping files for {volume}: "
                    f"{recon_path} and {label_path}"
                )
            self.recon_data.append(np.load(recon_path, mmap_mode="r"))
            self.label_data.append(np.load(label_path, mmap_mode="r"))

        self.lengths = [array.shape[0] for array in self.recon_data]
        self.cumulative = np.cumsum(self.lengths)
        self.total_length = int(self.cumulative[-1])

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        volume_index = int(np.searchsorted(self.cumulative, index, side="right"))
        previous = 0 if volume_index == 0 else int(self.cumulative[volume_index - 1])
        section_index = index - previous
        recon = np.array(self.recon_data[volume_index][section_index])
        label = np.array(self.label_data[volume_index][section_index])
        return (
            torch.as_tensor(recon, dtype=torch.complex64),
            torch.as_tensor(label, dtype=torch.complex64),
        )


class DeepMTPMapping12Echo(nn.Module):
    """MDI-Net operating on two independently acquired six-image FA sequences."""

    def __init__(self) -> None:
        super().__init__()
        self.learned_s = Learned_S2()
        self.learned_r = Learned_R2()

    def forward(
        self, recon_images: torch.Tensor, brain_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if recon_images.shape[2] != 12:
            raise ValueError(
                f"Expected 12 contrast-encoded images, got {recon_images.shape[2]}"
            )
        fa1 = recon_images[:, :, :6]
        fa2 = recon_images[:, :, 6:12]
        stacked = torch.stack([fa1, fa2], dim=2)

        # Discard the transient first image from each FA sequence. The remaining
        # images are re-indexed as t=1,...,N_e-1 before Eq. (11) is evaluated.
        retained = stacked[:, :, :, 1:]
        retained = retained * brain_mask[:, None, None, None]
        return self.learned_s(retained, brain_mask), self.learned_r(
            retained, brain_mask
        )


def parse_volumes(value: str) -> list[str]:
    value = value.strip()
    if "-" in value and "," not in value and value.replace("-", "").isdigit():
        start, end = (int(item) for item in value.split("-", maxsplit=1))
        return [f"vol{i}" for i in range(start, end + 1)]
    return [item.strip() for item in value.split(",") if item.strip()]


def make_brain_mask(label_images: torch.Tensor) -> torch.Tensor:
    label_sos = torch.sqrt(torch.sum(torch.abs(label_images) ** 2, dim=1))
    mean_image = label_sos.mean(dim=1)
    robust_max = mean_image.flatten(1).quantile(0.99, dim=1)[:, None, None]
    return (mean_image > 0.05 * robust_max.clamp_min(1e-8)).float()


def analytical_ratios(images: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fa1 = images[:, :, :6]
    fa2 = images[:, :, 6:12]
    retained = torch.stack([fa1, fa2], dim=2)[:, :, :, 1:]
    retained = retained * mask[:, None, None, None]
    return calculate_delta_S(retained), calculate_R2(retained)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=str, default="./data/Mapping_Training_Data_12Echo")
    parser.add_argument("--volumes", type=str, default="1-24")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/mdi_net")
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    device = torch.device(
        "cpu" if args.device.startswith("cuda") and not torch.cuda.is_available() else args.device
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    log_path = save_dir / f"train_{dt.datetime.now():%Y%m%d-%H%M%S}.log"

    dataset = MappingDataset12Echo(args.data_dir, parse_volumes(args.volumes))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = DeepMTPMapping12Echo().to(device)
    physics = MDIPhysicsLayer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
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

    for epoch in range(start_epoch, args.num_epoch):
        model.train()
        losses = []
        epoch_start = time.time()

        for step, (recon_images, label_images) in enumerate(loader, start=1):
            recon_images = recon_images.to(device, non_blocking=True)
            label_images = label_images.to(device, non_blocking=True)
            mask = make_brain_mask(label_images)

            with torch.no_grad():
                delta_ref, ratio_ref = analytical_ratios(label_images, mask)
                t1_ref, t2_ref = physics(delta_ref, ratio_ref, mask)

            delta_pred, ratio_pred = model(recon_images, mask)
            t1_pred, t2_pred = physics(delta_pred, ratio_pred, mask)

            # Quantitative-map supervision keeps the deterministic Bloch layer in
            # the training graph, as described in the manuscript.
            loss_t1 = F.l1_loss(t1_pred / 5.0, t1_ref / 5.0)
            loss_t2 = F.l1_loss(t2_pred / 0.2, t2_ref / 0.2)
            loss = loss_t1 + loss_t2

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

            if step % 20 == 0 or step == len(loader):
                print(
                    f"Epoch {epoch + 1:03d}/{args.num_epoch} "
                    f"Step {step:04d}/{len(loader)} "
                    f"Loss {loss.item():.6f} T1 {loss_t1.item():.6f} T2* {loss_t2.item():.6f}"
                )

        scheduler.step()
        mean_loss = float(np.mean(losses))
        summary = (
            f"epoch={epoch + 1} loss={mean_loss:.6f} "
            f"lr={optimizer.param_groups[0]['lr']:.8f} "
            f"seconds={time.time() - epoch_start:.1f}"
        )
        print(summary)
        with log_path.open("a") as stream:
            stream.write(summary + "\n")

        if (epoch + 1) % 5 == 0:
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": vars(args),
            }
            torch.save(state, save_dir / f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_dir / f"mapping_params_{epoch + 1}.pkl")


if __name__ == "__main__":
    main()
