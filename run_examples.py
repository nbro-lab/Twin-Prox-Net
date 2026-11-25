"""Run one seven-section forward pass and map only its central three sections."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch

from dataset import MTP_dataset_echos_mslice
from model_cginit import Deep_snms_echos_mslice
from mapping_checkpoint import load_mapping_model
from mdi_physics import MDIPhysicsLayer
from train_mapping import analytical_ratios
from test_mapping import mask_from_reconstruction
from utils import ifft2c, fft2c, adaptive_coil_combine

def run_example(args, kind, reconstruction_model, mapping_model, architecture, device):
    started = time.time()
    path = args.root / "examples" / f"{kind}_7sections.npz"
    with np.load(path, allow_pickle=False) as data:
        acquired = torch.from_numpy(data["acquired_kspace"].copy()).to(device)
        if kind == "retrospective":
            reference_images = ifft2c(torch.from_numpy(data["reference_kspace"].copy()).to(device))
        else:
            reference_images = torch.from_numpy(np.load(
                args.root/"examples/prospective_reference_grappa4x.npy",
                allow_pickle=False)).to(device)
        kernels = torch.from_numpy(data["null_kernels"].copy())
        mask = torch.from_numpy(data["mask"].copy()).to(device)
    if acquired.ndim != 5 or reference_images.ndim != 5 or acquired.shape[0] != 7:
        raise ValueError("Expected five-dimensional seven-section input and reference")
    if reference_images.shape[0] != 7 or reference_images.shape[2:] != acquired.shape[2:]:
        raise ValueError("Input and reference section, echo and image dimensions must agree")
    expected_reference_coils = acquired.shape[1] if kind == "retrospective" else 1
    if reference_images.shape[1] != expected_reference_coils:
        raise ValueError("Unexpected number of reference coil channels")
    if acquired.shape[2] != 12 or kernels.shape[:3] != (7, acquired.shape[1], acquired.shape[1]):
        raise ValueError("Echo count or SNM kernel shape is incorrect")
    if mask.shape != acquired.shape[-2:] or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Sampling mask must be binary and match ky/kz")
    if torch.count_nonzero(acquired * (1-mask)):
        raise ValueError("Acquired input has nonzero entries outside its sampling mask")
    # Reuse the public loader's kernel-padding convention, without a CFL volume.
    converter = object.__new__(MTP_dataset_echos_mslice)
    converter.ny, converter.nz = acquired.shape[-2:]
    converter.kernel_size_y, converter.kernel_size_z = kernels.shape[-2:]
    null_images = torch.stack([converter.Null_to_image(k) for k in kernels]).to(device)
    observed = ifft2c(acquired)
    scale = observed.abs().square().sum(dim=1).sqrt().amax().clamp_min(1e-8)
    normalized = acquired[None] / scale
    with torch.no_grad():
        full_kspace, images, cg = reconstruction_model(
            normalized, null_images[None], mask, torch.zeros_like(normalized)
        )
        if not torch.isfinite(images).all():
            raise RuntimeError("Non-finite reconstruction; do not publish this run")
        dc_residual = (torch.linalg.vector_norm((full_kspace-normalized)*mask) /
                       torch.linalg.vector_norm(normalized).clamp_min(1e-8)).item()
        # Exactly one 7-in/3-out pass, with no repeated edge padding.
        recon = images[0, 2:5] * scale
        reference = reference_images[2:5]
        brain_mask = mask_from_reconstruction(recon)
        delta, ratio = mapping_model(recon, brain_mask)
        t1, t2star = MDIPhysicsLayer().to(device)(delta, ratio, brain_mask)
        ref_delta, ref_ratio = analytical_ratios(reference, brain_mask)
        ref_t1, ref_t2 = MDIPhysicsLayer().to(device)(ref_delta, ref_ratio, brain_mask)
        if kind == "retrospective":
            combined_ref, weights = adaptive_coil_combine(reference)
            combined_recon, _ = adaptive_coil_combine(recon, coil_weights=weights)
            relative_error = (torch.linalg.vector_norm(combined_recon-combined_ref) /
                              torch.linalg.vector_norm(combined_ref).clamp_min(1e-8)).item()
            comparison = {"complex_image_relative_error": relative_error}
        else:
            # The separately acquired, coil-combined GRAPPA reference may have a
            # different scanner gain and phase. Compare magnitudes and report the
            # fitted global gain explicitly; saved reference data stays unchanged.
            combined_recon, _ = adaptive_coil_combine(recon)
            pred_mag, ref_mag = combined_recon.abs(), reference[:, 0].abs()
            gain = (pred_mag * ref_mag).sum() / pred_mag.square().sum().clamp_min(1e-8)
            error = (torch.linalg.vector_norm(gain*pred_mag-ref_mag) /
                     torch.linalg.vector_norm(ref_mag).clamp_min(1e-8)).item()
            comparison = {"magnitude_relative_error_after_global_gain_fit": error,
                          "reconstruction_gain_for_image_comparison": float(gain),
                          "spatial_registration": "none; corresponding source indices"}
    for tensor in (recon, t1, t2star, ref_t1, ref_t2):
        if not torch.isfinite(tensor).all():
            raise RuntimeError("Non-finite output")
    output = args.output / kind
    output.mkdir(parents=True, exist_ok=True)
    np.save(output/"example_recon.npy", recon.cpu().numpy())
    np.save(output/"example_label.npy", reference.cpu().numpy())
    np.savez_compressed(output/"maps.npz", T1_ms=t1.cpu().numpy()*1000,
                        T2star_ms=t2star.cpu().numpy()*1000,
                        reference_T1_ms=ref_t1.cpu().numpy()*1000,
                        reference_T2star_ms=ref_t2.cpu().numpy()*1000,
                        mask=brain_mask.cpu().numpy())
    report = dict(example=kind, input_sections=7, forward_passes=1,
                  output_sections=3, retained_indices=[2,3,4],
                  mapping_architecture=architecture,
                  reconstruction_architecture="released_adaptive_coil_combination",
                  normalization="maximum acquired-image RSS within the seven-section stack",
                  normalization_scale=float(scale), acquired_data_relative_residual=dc_residual,
                  image_comparison=comparison,
                  reference_status=("fully_sampled" if kind=="retrospective" else
                    "GRAPPA 4x reconstruction"),
                  reference_shape=list(reference.shape),
                  finite_outputs=True, seconds=time.time()-started,
                  torch_version=torch.__version__, device=str(device))
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--example", choices=["both", "retrospective", "prospective"], default="both")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("results/examples"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    config = json.loads((args.root/"configs/examples.json").read_text())
    if config["input_sections"] != 7 or config["retained_indices"] != [2,3,4]:
        raise ValueError("This demo requires one seven-section stack and central-three output")
    model = Deep_snms_echos_mslice(in_channel=2, niter=config["niter"],
                                  num_echoes=config["num_echoes"],
                                  cg_iterations=config["cg_iterations"])
    model.load_state_dict(torch.load(args.root/config["stage1_checkpoint"],
                                    map_location="cpu",weights_only=True), strict=True)
    model = model.to(device).eval()
    mapping, architecture = load_mapping_model(args.root/config["stage2_checkpoint"],device)
    for kind in (["retrospective","prospective"] if args.example=="both" else [args.example]):
        run_example(args,kind,model,mapping,architecture,device)
if __name__ == "__main__":
    main()
