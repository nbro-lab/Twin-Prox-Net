# GitHub release alignment notes

This package was revised to match the final Twin-Prox Net manuscript, except for the total parameter count, which depends on the final mapping-network checkpoint/version selected for release.

Implemented alignment changes:

- replaced explicit CSM/SENSE combination with reference-echo-derived adaptive complex coil combination;
- used the same adaptive weights for coil expansion before each subsequent hard data-consistency step;
- removed all workstation-specific `/data0/...` paths and fixed GPU IDs;
- set the default Stage I configuration to 10 CG-SNM initialization iterations, 15 unrolled iterations, seven input sections, three central supervised sections, and batch size four;
- implemented the manuscript composite loss with coefficients 1.0/0.5/0.1/0.05/0.1;
- retained the two independent six-image FA branches with shared EchoConvGRU parameters and separately initialized hidden states;
- changed MDI-Net input construction from multiple pairwise ratio maps to the two integrated complex ratio maps in manuscript Eq. (11);
- discarded and re-indexed the transient first image before MDI ratio estimation;
- trained MDI-Net through the differentiable Bloch solver using quantitative T1/T2* targets;
- changed retrospective evaluation to use the fully sampled reference rather than CG initialization;
- added a local differentiable SSIM implementation;
- removed duplicate executable blocks and legacy path-dependent mapping code;
- documented that QSM processing uses an external standard pipeline.

The code was syntax-checked and smoke-tested with small synthetic tensors. Full numerical reproduction still requires the original masks, pretrained checkpoints, calibrated SNMs maps, and de-identified example data.
