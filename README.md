# Twin-Prox Net

This repository contains the PyTorch implementation used for **Twin-Prox Net**, a physics-driven unrolled reconstruction and quantitative mapping framework for accelerated 3D MULTIPLEX MRI.

The code is organized into two main stages:

1. **Stage I: SNMs-based unrolled reconstruction with Twin-STPN**
2. **Stage II: MDI-based quantitative mapping for \(T_1\) and \(T_2^*\)**

## Repository structure

```text
Twin-Prox-Net/
├── dataset.py             # Dataset classes for multi-slice and multi-echo MULTIPLEX data
├── model_cginit.py        # Main Twin-Prox Net reconstruction model with CG/SENSE initialization
├── model_deep_snms.py     # Deep SNMs and MDI mapping modules used by the mapping-stage scripts
├── mdi_function.py        # MDI ratio calculation and analytical signal synthesis utilities
├── train_tpn.py           # Training script for the reconstruction network
├── test_tpn.py            # Inference/evaluation script for the reconstruction network
├── train_mapping.py       # Training script for the MDI-Net mapping stage
├── test_mapping.py        # Inference/evaluation script for quantitative mapping
├── utils.py               # FFT/CFL I/O and complex-valued utility functions
├── requirements.txt       # Python package requirements
└── pyproject.toml         # Formatting configuration
```

## Environment

The code was developed with PyTorch and CUDA-enabled GPUs. Install the required packages with:

```bash
pip install -r requirements.txt
```

Recommended core dependencies:

- Python >= 3.8
- PyTorch with CUDA support
- NumPy
- SciPy
- scikit-image
- Matplotlib
- einops
- torchvision

## Default data and checkpoint paths

The scripts keep the project-specific default paths used in the original experiments. By default, data and checkpoints are assumed to be located under:

```text
/data0/zhaojiaying/MTP_datasets_total/             # fully sampled retrospective training/testing data
/data0/zhaojiaying/MTP_data/perspective_data/      # prospectively accelerated data
/data0/zhaojiaying/Mapping_Training_Data_12Echo/   # pre-generated mapping-stage training data
/data0/zhaojiaying/Mappping_test_data/             # mapping-stage test data
/data0/zhaojiaying/DeepSNMs_MTP_checkpoints/       # checkpoints and reconstruction outputs
```

If your data are stored elsewhere, update the corresponding command-line arguments or the path variables in the training and testing scripts.

## Expected data layout

For the reconstruction stage, the dataset loader expects each volume to be organized by echo:

```text
<data_root>/
└── <split>/
    └── <volume_name>/
        ├── Echo_1/
        │   ├── ksp.cfl / ksp.hdr
        │   ├── ksp_grappa_9x.cfl / ksp_grappa_9x.hdr
        │   └── Null_2d.cfl / Null_2d.hdr
        ├── Echo_2/
        │   ├── ksp.cfl / ksp.hdr
        │   ├── ksp_grappa_9x.cfl / ksp_grappa_9x.hdr
        │   └── Null_2d.cfl / Null_2d.hdr
        └── ...
```

The expected complex k-space dimensions are internally transposed to:

```text
[nx, nc, ne, ny, nz]
```

where:

- `nx`: readout sections
- `nc`: compressed coil channels
- `ne`: number of echoes, typically 12
- `ny`, `nz`: phase-encoding dimensions

## Required auxiliary files

Some scripts require additional local files that are not included in this minimal code package:

```text
mask/<mask_name>.mat          # sampling masks, e.g. caipi_9x.mat or caipi_16x_cen30.mat
ssim.py                       # SSIM loss implementation used by train_tpn.py/test_tpn.py
pretrained checkpoints         # model weights used for inference
representative test data       # CFL/HDR or NumPy files following the layout above
```

## Training Stage I: Twin-Prox reconstruction

Example command:

```bash
python train_tpn.py \
    --num_epoch 200 \
    --batch_size 2 \
    --learning_rate 0.0005 \
    --niter 15 \
    --nEcho 12 \
    --acc 9x \
    --mask caipi_16x_cen30
```

The default training script uses:

```text
/data0/zhaojiaying/MTP_datasets_total/
```

as the training data root and saves checkpoints under:

```text
/data0/zhaojiaying/DeepSNMs_MTP_checkpoints/
```

## Testing Stage I: Reconstruction inference

Example command:

```bash
python test_tpn.py \
    --batch_size 1 \
    --niter 15 \
    --nEcho 12 \
    --nSlice 7 \
    --acc 9x \
    --mask caipi_9x
```

The script loads a trained checkpoint, applies the reconstruction network, and saves reconstructed k-space/image outputs in CFL format.

## Training Stage II: MDI-Net quantitative mapping

Example command:

```bash
python train_mapping.py \
    --num_epoch 100 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --data_dir /data0/zhaojiaying/Mapping_Training_Data_12Echo \
    --save_dir /data0/zhaojiaying/DeepSNMs_MTP_checkpoints/Mapping_Stage2
```

The mapping-stage dataset is expected to contain paired NumPy files:

```text
<volume_name>_recon.npy
<volume_name>_label.npy
```

where each file stores 12-echo complex-valued reconstructed/reference images.

## Testing Stage II: Quantitative mapping

Example command:

```bash
python test_mapping.py \
    --batch_size 4 \
    --data_dir /data0/zhaojiaying/Mappping_test_data/ \
    --save_dir ./persp_mapping_results
```

The script computes \(T_1\) and \(T_2^*\) maps using the analytical MDI physics solver and/or the learned MDI-Net mapping stage, depending on the selected section of the script.

## Data format

This code uses BART-style complex CFL/HDR files for most k-space and reconstruction data. Utility functions for reading and writing these files are implemented in `utils.py`.

A pair of files is expected for each CFL object:

```text
name.cfl
name.hdr
```

## Notes on released data

The full fully sampled training data are not publicly released because of institutional privacy regulations for human-subject MRI data and the large storage burden of multi-coil, multi-echo 3D raw k-space acquisitions.

For reproducibility, the public repository is intended to include de-identified representative testing data, including:

1. one retrospectively undersampled test example with its fully sampled reference, and
2. one prospectively accelerated test example.

The retrospective example should contain a small stack of adjacent readout sections, such as three to five neighboring sections, sufficient to test preprocessing, reconstruction, and quantitative mapping inference. These example datasets are intended for inference and pipeline verification rather than full retraining or cohort-level statistical reproduction.

## Important implementation notes

- The scripts contain project-specific default paths. Update them before running on a different workstation.
- The release is formatted for readability but preserves the original algorithmic logic.
- Runtime depends on GPU memory, the number of unrolled iterations, the number of coils, and the number of readout sections.
- The code has been syntax-checked with `python -m py_compile`; full execution requires the data, masks, checkpoints, and local auxiliary files described above.

## Citation

If you use this code, please cite the corresponding Twin-Prox Net manuscript.
