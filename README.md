# Twin-Prox Net

PyTorch implementation of **Twin-Prox Net**, a physics-driven spatiotemporal unrolled framework for accelerated 3D MULTIPLEX MRI reconstruction and quantitative mapping.

The implementation provides two stages:

1. **Stage I — SNMs-based unrolled reconstruction with Twin-STPN**
2. **Stage II — complex-domain MDI-Net with deterministic Bloch-equation-based \(T_1\) and \(T_2^*\) estimation**

## Seven-section examples (one forward pass per example)

Extract `release_assets.zip` into this repository directory. It supplies two
checkpoints, one retrospective stack, one prospective stack, and a checksum manifest.
Both stacks contain exactly seven adjacent readout sections and all 12 contrasts.
They are already in the hybrid x–ky–kz domain; do not apply another readout FFT.

```bash
sha256sum -c SHA256SUMS
python run_examples.py --device cuda:0
```

Use an available GPU supported by your installed PyTorch build, or `--device cpu`.
Validation passed on an A800 with PyTorch 2.4.1+cu121, and on an RTX PRO 6000 with Python 3.11 / PyTorch 2.11.0+cu128.
The script reads `configs/examples.json`, runs one 7-in/3-out reconstruction per
example, and maps the central three sections. Outputs are written to
`results/examples/{retrospective,prospective}/`.
Select one case using `--example retrospective` or `--example prospective`.

The example files are:

| File | Contents |
| --- | --- |
| `retrospective_7sections.npz` | Retrospective acquired input, fully sampled reference, SNM kernels and mask |
| `prospective_7sections.npz` | Prospective acquired input, SNM kernels and mask |
| `prospective_reference_grappa4x.npy` | Corresponding seven-section GRAPPA 4x reference reconstruction |

Both NPZ files contain `acquired_kspace` with shape `[7,12,12,288,84]`
(section/coil/contrast/ky/kz), `null_kernels` with shape `[7,12,12,9,9]`
(section/coil_out/coil_in/kernel_y/kernel_z), and a binary `mask` of shape
`[288,84]`. The retrospective NPZ additionally contains `reference_kspace`
with the same shape as the acquired input.

The prospective GRAPPA 4x reference is an image-domain complex64 NumPy array of
shape `[7,1,12,288,84]` (section/combined_coil_channel/contrast/y/z).
It is already coil-combined. Do not apply an inverse FFT to this image reference
or duplicate it across coil channels. All seven source section indices match
the prospective input. It is a separately acquired accelerated reference
reconstruction, not a fully sampled ground truth.

The runner uses the central three reference sections for analytical reference
mapping. Prospective image comparison uses magnitudes with an explicitly
reported global gain fit, because acquisition gain and phase may differ.
Reference arrays and reference maps are saved without this gain adjustment;
no spatial registration is applied by the example script.

The supplied Stage II checkpoint uses the **pairwise-ratio 8/10-channel**
architecture in `model_mapping_legacy.py`. Both `run_examples.py` and
`test_mapping.py` identify this checkpoint and load it strictly. The training entry point uses the integrated
two-channel architecture. For inference with the supplied pretrained checkpoint,
use either of the inference entry points above.

## Repository structure

```text
Twin-Prox-Net/
├── dataset.py          # Multi-section, multi-echo MULTIPLEX dataset loaders
├── model_cginit.py     # SNMs unrolling, adaptive coil combination/expansion, Twin-STPN
├── model_deep_snms.py  # Complex-domain MDI-Net ratio correctors
├── mdi_function.py     # Analytical complex ratios from manuscript Eq. (11)
├── mdi_physics.py      # Differentiable analytical Bloch solver for T1 and T2*
├── train_tpn.py        # Stage I training
├── test_tpn.py         # Stage I retrospective/prospective inference
├── train_mapping.py    # Stage II MDI-Net training
├── test_mapping.py     # Stage II quantitative mapping inference
├── ssim.py             # Differentiable SSIM implementation
├── utils.py            # Centered FFT, CFL I/O, and adaptive coil utilities
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Recommended environment:

- Python 3.10+
- PyTorch 2.4+
- CUDA-enabled GPU for full 3D experiments

## Data layout

The Stage I loader expects one directory per contrast-encoded image:

```text
<data_root>/
├── train/
│   ├── vol1/
│   │   ├── Echo_1/
│   │   │   ├── ksp.cfl
│   │   │   ├── ksp.hdr
│   │   │   ├── Null_2d.cfl
│   │   │   └── Null_2d.hdr
│   │   ├── Echo_2/
│   │   └── ... Echo_12/
│   └── ...
├── valid/
└── test/
```

After loading, the multi-echo k-space tensor is arranged as:

```text
[readout_section, coil, contrast, ky, kz]
```

The default command-line volume settings are:

```text
training:   vol1–vol24
validation: vol25–vol28
testing:    vol29–vol33
```

Change `--train_volumes`, `--valid_volumes`, or `--volumes` when local volume names differ.

## Adaptive coil combination

Explicit coil sensitivity maps are **not** required by the released reconstruction code.

Within every unrolled proximal block:

```text
multi-coil image
→ reference-echo-derived adaptive complex coil combination
→ Twin-STPN in a compact single-channel complex domain
→ adaptive coil expansion using the same weights
→ Fourier transform and hard data insertion
```

The adaptive weights are used only as an interface between the multi-coil SNMs representation and the learned proximal module. They do not participate in the SNMs residual operator.

For training supervision, one adaptive phase reference is estimated from the fully sampled target and applied to both the reconstruction and target. This avoids arbitrary coil-combination phase offsets in the complex-domain loss.

## Stage I configuration

The default Stage I configuration is:

- 10 CG-SNM initialization iterations
- 15 learnable unrolled iterations
- 12 contrast-encoded images
- two independently processed six-image FA branches
- shared EchoConvGRU parameters and separately initialized hidden states
- seven adjacent readout sections per network input
- supervision on the central three sections
- batch size 4

The Stage I loss is:

```text
L = 1.0 L_cplx
  + 0.5 L_phase
  + 0.1 L_ksp
  + 0.05 L_SSIM
  + 0.1 L_JointTV
```

where:

- `L_cplx` is complex L1 loss;
- `L_phase` is the magnitude-guided cosine phase loss;
- `L_ksp` is the logarithmic complex k-space magnitude penalty;
- `L_SSIM` is magnitude SSIM loss;
- `L_JointTV` penalizes full-resolution spatial and echo-wise variations.

### Train Stage I

```bash
python train_tpn.py \
  --data_root ./data/MTP_datasets_total \
  --train_volumes 1-24 \
  --valid_volumes 25-28 \
  --output_dir ./checkpoints \
  --experiment_name twin_prox_net \
  --mask_dir ./mask \
  --mask caipi_9x \
  --acc 9x \
  --batch_size 4 \
  --niter 15 \
  --cg_iterations 10 \
  --nSlice 7
```

Resume training with either a full `.pth` checkpoint or a model-only `.pkl` state dict:

```bash
python train_tpn.py --resume ./checkpoints/twin_prox_net/checkpoint_epoch_100.pth
```

### Test Stage I

Retrospective testing with fully sampled references:

```bash
python test_tpn.py \
  --data_root ./data/MTP_datasets_total \
  --data_split test \
  --volumes 29-33 \
  --checkpoint ./checkpoints/twin_prox_net/checkpoint_epoch_200.pth \
  --output_dir ./results/reconstruction \
  --mask_dir ./mask \
  --mask caipi_9x
```

Prospective inference without a fully sampled reference:

```bash
python test_tpn.py \
  --data_root ./data/perspective_data \
  --data_split test \
  --volumes vol1 \
  --checkpoint ./checkpoints/twin_prox_net/checkpoint_epoch_200.pth \
  --no-reference_available
```

The inference script uses a seven-section sliding window and retains the central three reconstructed sections from each pass.

## Stage II: MDI-Net

For each FA-specific six-image sequence, the transient first image is discarded. The remaining five images are re-indexed before computing the integrated complex ratios in manuscript Eq. (11):

- one optimal matched-filter echo-decay ratio \(\Delta S\), integrated over coils, FAs, and adjacent retained echoes;
- one flip-angle boundary ratio \(R\), integrated over coils and retained echoes.

MDI-Net refines these two complex ratio maps. A differentiable analytical Bloch layer then produces \(T_1\) and \(T_2^*\) maps, and quantitative-map losses are backpropagated through the solver.

The Stage II dataset contains paired NumPy arrays:

```text
<volume>_recon.npy
<volume>_label.npy
```

Each section is expected to have shape:

```text
[coil, 12, ky, kz]
```

### Train Stage II

```bash
python train_mapping.py \
  --data_dir ./data/Mapping_Training_Data_12Echo \
  --volumes 1-24 \
  --save_dir ./checkpoints/mdi_net \
  --batch_size 4
```

### Test Stage II

```bash
python test_mapping.py \
  --data_dir ./data/Mapping_Test_Data_12Echo \
  --volumes vol1 \
  --checkpoint ./checkpoints/mdi_net/checkpoint_epoch_100.pth \
  --save_dir ./results/mapping
```

Use `--no-reference_available` for inference-only prospective data.

## Paths and local configuration

No workstation-specific paths or fixed GPU IDs are stored in the released scripts. All data, mask, checkpoint, result, and device locations are command-line options. Default paths are relative to the repository root:

```text
./data/
./mask/
./checkpoints/
./results/
```

## QSM processing

The repository releases Twin-Prox reconstruction and the MDI-based \(T_1\)/\(T_2^*\) mapping code. QSM is derived from reconstructed multi-echo phase using the standard external QSM pipeline described in the manuscript and is not implemented in this repository.

## Checkpoints and representative data

The prepared asset bundle contains:

1. one de-identified retrospective example with a fully sampled reference;
2. one prospective accelerated seven-section example and a separate GRAPPA 4x reference reconstruction;
3. Stage I and Stage II pretrained checkpoints;
4. the sampling mask used for inference.

The full cohort is not included because of human-subject data restrictions and the storage burden of multi-coil, multi-echo 3D raw k-space data.

## Checkpoint loading

- The code has no dependency on explicit CSM files.
- The supplied Stage II checkpoint requires the included legacy pairwise-ratio architecture. The inference loader detects it and enforces strict weight loading; it cannot be substituted into the integrated two-channel architecture.
- Parameter counts can be obtained from the loaded release checkpoints.

## Citation

Please cite the Twin-Prox Net manuscript when using this code.
