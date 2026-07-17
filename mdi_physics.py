"""Differentiable analytical MDI mapping for MULTIPLEX MRI."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MDIPhysicsLayer(nn.Module):
    """Convert refined complex MDI ratios into quantitative T1 and T2* maps.

    The input ratios are the complex echo-decay ratio ``delta_s`` and the
    flip-angle boundary ratio ``ratio_r`` defined in Eq. (11) of the manuscript.
    Output units are seconds.
    """

    def __init__(
        self,
        tr1: float = 0.00797,
        tr2: float = 0.03823,
        fa1_deg: float = 4.0,
        fa2_deg: float = 16.0,
        delta_te: float = 0.0063,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.tr1 = float(tr1)
        self.tr2 = float(tr2)
        self.delta_te = float(delta_te)
        self.n = self.tr2 / self.tr1
        self.eps = float(eps)

        fa1 = math.radians(fa1_deg)
        fa2 = math.radians(fa2_deg)
        self.ca1 = math.cos(fa1)
        self.ca2 = math.cos(fa2)
        self.sa1 = math.sin(fa1)
        self.sa2 = math.sin(fa2)

    def forward(
        self,
        delta_s: torch.Tensor,
        ratio_r: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decay = torch.clamp(torch.abs(delta_s), min=0.01, max=0.999)
        t2star = -self.delta_te / torch.log(decay)

        r = torch.abs(ratio_r) * (self.sa2 / self.sa1)
        a = self.n * (self.n + 1) * self.ca1 * self.ca2 * (r * self.ca1 - self.ca2)
        b = (self.n + 1) * (
            (self.n + self.ca1) * self.ca2**2
            - (self.n + self.ca2) * self.ca1**2 * r
        ) + self.n * (r * self.sa1**2 * self.ca2 - self.ca1 * self.sa2**2)
        c = self.sa2**2 * (self.n + self.ca1) - r * self.sa1**2 * (
            self.n + self.ca2
        )

        discriminant = torch.clamp(b.square() - 4 * a * c, min=0.0)
        root = (-b - torch.sqrt(discriminant)) / (2 * a + self.eps)
        t1 = self.tr1 / (torch.abs(root) + self.eps)

        t1 = torch.clamp(t1, min=0.0, max=5.0)
        t2star = torch.clamp(t2star, min=0.0, max=0.2)
        if mask is not None:
            t1 = t1 * mask
            t2star = t2star * mask
        return t1, t2star
