# Release notes

This release provides a runnable seven-section reconstruction and quantitative
mapping example for Twin-Prox Net.

## Included components

- Stage I reconstruction and Stage II mapping pretrained checkpoints.
- One retrospective example with a fully sampled reference.
- One prospective accelerated example with a separate GRAPPA 4x reference reconstruction.
- Seven adjacent readout sections and all 12 contrasts per example.
- Calibrated SNM kernels, sampling masks, inference configuration and SHA256 checksums.
- A single-pass inference script that retains the central three reconstructed sections
  and produces T1 and T2* maps.
- Automatic architecture selection and strict loading of the supplied mapping checkpoint.

## Validation

Both examples completed reconstruction and mapping with finite outputs.
The acquired-data consistency residual was zero in both cases.
Validation reports are included in the validation directory.

See README.md for installation, data layout, checkpoint loading and execution commands.

## Reference data layout

The prospective reference is supplied separately as
`examples/prospective_reference_grappa4x.npy`: seven sections, one combined-coil
channel and 12 contrasts. The example runner reads this image directly and uses
the matching central three sections for reference mapping.
