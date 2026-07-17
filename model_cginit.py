"""Stage I Twin-Prox Net reconstruction model.

The implementation combines:
- CG-SNM initialization;
- SNMs residual-correction unrolling with hard data insertion;
- reference-echo-derived adaptive coil combination and expansion;
- FA-specific shared-weight EchoConvGRU temporal regularization;
- full-resolution dilated spatial proximal blocks with learnable soft thresholding;
- a final inter-section 3D CNN and hard data-consistency operation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import init
from torch.utils.checkpoint import checkpoint

from utils import (
    adaptive_coil_combine,
    adaptive_coil_expand,
    c2r,
    fft2c,
    ifft2c,
    r2c,
)


def checkpoint_module(module: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Use non-reentrant checkpointing when available, with a legacy fallback."""
    try:
        return checkpoint(module, inputs, use_reentrant=False)
    except TypeError:  # PyTorch 1.12 compatibility
        return checkpoint(module, inputs)


def init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02) -> None:
    """Initialize convolutional and linear layers."""

    def init_func(module: nn.Module) -> None:
        classname = module.__class__.__name__
        if hasattr(module, "weight") and (
            "Conv" in classname or "Linear" in classname
        ):
            if init_type == "normal":
                init.normal_(module.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(module.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(module.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    f"Initialization method '{init_type}' is not implemented"
                )
            if getattr(module, "bias", None) is not None:
                init.constant_(module.bias.data, 0.0)
        elif "BatchNorm2d" in classname:
            init.normal_(module.weight.data, 1.0, init_gain)
            init.constant_(module.bias.data, 0.0)

    net.apply(init_func)


def init_net(
    net: nn.Module,
    init_type: str = "normal",
    init_gain: float = 0.02,
    gpu_ids=None,
) -> nn.Module:
    """Backward-compatible network initialization helper."""
    gpu_ids = gpu_ids or []
    if gpu_ids:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required when gpu_ids are provided")
        net.to(gpu_ids[0])
        net = nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain)
    return net


class ConvGRUCell(nn.Module):
    """Lightweight 2D convolutional GRU cell."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv_gates = nn.Conv2d(
            input_dim + hidden_dim,
            2 * hidden_dim,
            kernel_size,
            padding=padding,
        )
        self.conv_can = nn.Conv2d(
            input_dim + hidden_dim,
            hidden_dim,
            kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gates = self.conv_gates(torch.cat([x, h], dim=1))
        reset, update = torch.chunk(gates, 2, dim=1)
        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update)
        candidate = torch.tanh(self.conv_can(torch.cat([x, reset * h], dim=1)))
        return (1.0 - update) * h + update * candidate


class EchoConvGRU(nn.Module):
    """Process one ordered multi-echo sequence with a freshly initialized state."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = ConvGRUCell(input_dim, hidden_dim)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        batch, _, echoes, height, width = x_seq.shape
        hidden = torch.zeros(
            batch,
            self.hidden_dim,
            height,
            width,
            device=x_seq.device,
            dtype=x_seq.dtype,
        )
        outputs = []
        for echo in range(echoes):
            hidden = self.cell(x_seq[:, :, echo], hidden)
            outputs.append(hidden)
        return torch.stack(outputs, dim=2)


class SoftThreshold2D(nn.Module):
    """Channel-wise learnable soft-thresholding operator."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.lambd = nn.Parameter(torch.full((1, channels, 1, 1), 0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        threshold = F.relu(self.lambd)
        return torch.sign(x) * F.relu(torch.abs(x) - threshold)


class MultiDict_Im_Layer_CDL_2D(nn.Module):
    """Twin-STPN temporal and full-resolution spatial proximal module."""

    def __init__(self, num_echoes: int = 12, filter_base: int = 32) -> None:
        super().__init__()
        self.num_echoes = num_echoes
        in_channels = 2 * num_echoes
        self.c1 = filter_base

        self.echo_temporal_gru = EchoConvGRU(input_dim=2, hidden_dim=2)
        self.enc_in = nn.Conv2d(
            in_channels, self.c1, kernel_size=3, padding=1, bias=False
        )
        self.block1 = self._res_block_2d(self.c1, dilation=1)
        self.block2 = self._res_block_2d(self.c1, dilation=2)
        self.block3 = self._res_block_2d(self.c1, dilation=1)
        self.st1 = SoftThreshold2D(self.c1)
        self.st2 = SoftThreshold2D(self.c1)
        self.st3 = SoftThreshold2D(self.c1)
        self.dec_out = nn.Conv2d(
            self.c1, in_channels, kernel_size=3, padding=1, bias=False
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _res_block_2d(channels: int, dilation: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: [batch, echo, ky, kz] complex
        complex_channels = torch.stack([inputs.real, inputs.imag], dim=1)

        if self.num_echoes == 12:
            fa1, fa2 = torch.split(complex_channels, 6, dim=2)
            # The same ConvGRU parameters are reused, while each call initializes
            # a separate hidden state. The two FA branches are therefore not
            # treated as one continuous 12-point decay trajectory.
            fa1_features = self.echo_temporal_gru(fa1) + fa1
            fa2_features = self.echo_temporal_gru(fa2) + fa2
            temporal_features = torch.cat([fa1_features, fa2_features], dim=2)
        else:
            temporal_features = self.echo_temporal_gru(complex_channels) + complex_channels

        flattened = rearrange(
            temporal_features, "b real_imag e y z -> b (real_imag e) y z"
        )
        features = self.enc_in(flattened)
        features = self.st1(self.block1(features) + features)
        features = self.st2(self.block2(features) + features)
        features = self.st3(self.block3(features) + features)
        output = flattened + self.alpha * self.dec_out(features)
        real, imag = torch.chunk(output, 2, dim=1)
        return torch.complex(real, imag)


class Deepsnms_block_ns_image(nn.Module):
    """One SNMs residual-correction and learned proximal iteration."""

    def __init__(self, num_echoes: int = 1) -> None:
        super().__init__()
        self.rho = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.ImageLayer = MultiDict_Im_Layer_CDL_2D(
            num_echoes=num_echoes,
            filter_base=16,
        )

    def forward(self, data, x: torch.Tensor, operator):
        acquired_kspace, mask = data
        missing_update = x - self.rho * operator.Ax_b(x)
        full_kspace = missing_update + acquired_kspace
        coil_images = ifft2c(full_kspace)

        combined, weights = adaptive_coil_combine(coil_images, ref_echo=0)
        regularized = checkpoint_module(self.ImageLayer, combined)
        expanded = adaptive_coil_expand(regularized, weights)

        missing_kspace = fft2c(expanded)
        missing_kspace = missing_kspace * (1 - mask).view(1, 1, 1, *mask.shape)
        reconstructed_images = ifft2c(missing_kspace + acquired_kspace)
        return data, missing_kspace, reconstructed_images


class Deep_snms_echos(nn.Module):
    """Single-section multi-echo SNMs unrolling network."""

    def __init__(
        self,
        niter: int,
        num_echoes: int = 12,
        cg_iterations: int = 10,
    ) -> None:
        super().__init__()
        self.niter = niter
        self.cg_iterations = int(cg_iterations)
        self.blocklist = nn.ModuleList(
            [
                Deepsnms_block_ns_image(num_echoes=num_echoes)
                for _ in range(self.niter)
            ]
        )

    def forward(
        self,
        kData: torch.Tensor,
        Null_2d: torch.Tensor,
        mask: torch.Tensor,
        x0: torch.Tensor,
    ):
        operator = Aclass_Vectorized(kData, Null_2d, mask)
        target = -operator.b
        missing_init = CG_Vectorized(
            x0,
            operator,
            target,
            max_iter=self.cg_iterations,
        )
        cg_image = ifft2c(missing_init + kData)
        missing_kspace = missing_init * operator.mask_nacq

        data = [kData, mask]
        for block in self.blocklist:
            data, missing_kspace, _ = block(data, missing_kspace, operator)

        full_kspace = missing_kspace + kData
        images = ifft2c(full_kspace)
        return full_kspace, images, cg_image


class Im_cnn3d_layer_mslice_enhanced(nn.Module):
    """Residual 3D CNN for inter-section consistency along the readout axis."""

    def __init__(
        self,
        in_channel: int,
        filter: int = 32,
        slice_kernel: int = 7,
    ) -> None:
        super().__init__()
        section_padding = slice_kernel // 2
        self.conv1 = nn.Conv3d(
            in_channel,
            filter,
            kernel_size=(slice_kernel, 3, 3),
            padding=(section_padding, 1, 1),
            bias=False,
        )
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=filter)
        self.conv2 = nn.Conv3d(
            filter,
            filter,
            kernel_size=(slice_kernel, 3, 3),
            padding=(section_padding, 1, 1),
            bias=False,
        )
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=filter)
        self.conv3 = nn.Conv3d(
            filter,
            in_channel,
            kernel_size=(slice_kernel, 3, 3),
            padding=(section_padding, 1, 1),
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.dropout(self.relu(self.norm1(self.conv1(inputs))))
        features = self.dropout(self.relu(self.norm2(self.conv2(features))))
        return self.conv3(features) + inputs


class Deep_snms_echos_mslice(nn.Module):
    """Seven-section Twin-Prox reconstruction with final inter-section refinement."""

    def __init__(
        self,
        in_channel: int,
        niter: int,
        num_echoes: int = 12,
        cg_iterations: int = 10,
    ) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.niter = niter
        self.unrolling_net = Deep_snms_echos(
            niter,
            num_echoes=num_echoes,
            cg_iterations=cg_iterations,
        )
        self.Im_conv3d = Im_cnn3d_layer_mslice_enhanced(
            in_channel=self.in_channel
        )

    def forward(
        self,
        kData: torch.Tensor,
        Null_2d: torch.Tensor,
        mask: torch.Tensor,
        x0: torch.Tensor,
    ):
        batch, sections, coils, echoes, height, width = kData.shape
        x0_flat = rearrange(x0, "b s c e y z -> (b s) c e y z")
        kdata_flat = rearrange(kData, "b s c e y z -> (b s) c e y z")
        null_flat = rearrange(
            Null_2d,
            "b s c_out c_in y z -> (b s) c_out c_in y z",
        )

        _, images_flat, cg_flat = self.unrolling_net(
            kdata_flat,
            null_flat,
            mask,
            x0_flat,
        )

        combined, weights = adaptive_coil_combine(images_flat, ref_echo=0)
        inter_section_input = rearrange(
            combined,
            "(b s) e y z -> (b e) s y z",
            b=batch,
            s=sections,
        )
        inter_section_output = r2c(self.Im_conv3d(c2r(inter_section_input)))
        combined_refined = rearrange(
            inter_section_output,
            "(b e) s y z -> (b s) e y z",
            b=batch,
            s=sections,
        )
        expanded = adaptive_coil_expand(combined_refined, weights)
        expanded = rearrange(
            expanded,
            "(b s) c e y z -> b s c e y z",
            b=batch,
            s=sections,
            c=coils,
            e=echoes,
        )

        missing_kspace = fft2c(expanded)
        missing_kspace = missing_kspace * (1 - mask).view(
            1,
            1,
            1,
            1,
            height,
            width,
        )
        full_kspace = missing_kspace + kData
        images = ifft2c(full_kspace)
        cg_images = rearrange(
            cg_flat,
            "(b s) c e y z -> b s c e y z",
            b=batch,
            s=sections,
        )
        return full_kspace, images, cg_images


class Aclass_Vectorized:
    """Vectorized calibrated SNMs residual operator over all echoes."""

    def __init__(
        self,
        kData: torch.Tensor,
        Null_2d: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        self.kData = kData
        self.Null_2d = Null_2d
        self.mask_nacq = (1 - mask).view(1, 1, 1, *mask.shape)

        acquired_images = ifft2c(kData).unsqueeze(1)
        null_expanded = Null_2d.unsqueeze(3)
        residual_images = torch.sum(null_expanded * acquired_images, dim=2)
        self.b = fft2c(residual_images) * self.mask_nacq

    def A(self, x: torch.Tensor) -> torch.Tensor:
        missing = x * self.mask_nacq
        missing_images = ifft2c(missing).unsqueeze(1)
        null_expanded = self.Null_2d.unsqueeze(3)
        residual_images = torch.sum(null_expanded * missing_images, dim=2)
        return fft2c(residual_images) * self.mask_nacq

    def AH(self, y: torch.Tensor) -> torch.Tensor:
        # Retained for compatibility with the original SNMs implementation.
        return self.A(y)

    def AHAx_AHb(self, x: torch.Tensor) -> torch.Tensor:
        return self.AH(self.Ax_b(x))

    def Ax_b(self, x: torch.Tensor) -> torch.Tensor:
        return self.A(x) + self.b


def CG_Vectorized(
    x: torch.Tensor,
    operator: Aclass_Vectorized,
    b: torch.Tensor,
    max_iter: int = 20,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Vectorized conjugate-gradient initialization over coils and echoes."""
    residual = b - operator.A(x)
    direction = residual
    sum_dims = tuple(range(1, x.ndim))
    residual_norm = torch.sum(
        torch.conj(residual) * residual,
        dim=sum_dims,
        keepdim=True,
    ).real

    for _ in range(max_iter):
        if residual_norm.abs().max() < eps:
            break
        applied = operator.A(direction)
        denominator = torch.sum(
            torch.conj(direction) * applied,
            dim=sum_dims,
            keepdim=True,
        ).real
        alpha = residual_norm / (denominator + eps)
        x = x + alpha * direction
        residual = residual - alpha * applied
        new_norm = torch.sum(
            torch.conj(residual) * residual,
            dim=sum_dims,
            keepdim=True,
        ).real
        beta = new_norm / (residual_norm + eps)
        direction = residual + beta * direction
        residual_norm = new_norm
    return x
