# -*- coding: utf-8 -*-
"""Circular Bi-Mamba boundary encoder.

This module is used for circular sequence modeling of text contour points.
It keeps the original interface:

    CircularBiMambaBT(node_feats, adj=None) -> offsets

Input:
    node_feats: (B, P, C) or (B, C, P)

Output:
    offsets: (B, 2, P)
"""

import math
from typing import Optional

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError as exc:
    raise ImportError(
        "CircularBiMambaBT requires mamba-ssm. "
        "Please install it with: pip install mamba-ssm"
    ) from exc


class CircularPositionalEmbedding(nn.Module):
    """Circular positional embedding for closed contour sequences.

    For each contour point index i, this module uses:
        theta = 2 * pi * i / P

    Then it maps [sin(theta), cos(theta)] to the feature dimension.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add circular positional embedding.

        Args:
            x: Tensor with shape (B, P, C).

        Returns:
            Tensor with shape (B, P, C).
        """
        if x.dim() != 3:
            raise ValueError("Input x must be a 3D tensor with shape (B, P, C).")

        _, num_points, _ = x.shape
        device = x.device

        index = torch.arange(num_points, device=device, dtype=torch.float32)
        theta = 2.0 * math.pi * index / float(num_points)

        pos = torch.stack(
            [torch.sin(theta), torch.cos(theta)],
            dim=-1,
        )
        pos = self.proj(pos).unsqueeze(0)

        return x + pos


class RingDepthwiseConv1d(nn.Module):
    """Depthwise 1D convolution with circular padding.

    The contour sequence is treated as a closed ring. Therefore, the left
    padding comes from the end of the sequence and the right padding comes
    from the beginning of the sequence.

    Input and output shapes are both (B, P, C).
    """

    def __init__(self, d_model: int, kernel_size: int = 3) -> None:
        super().__init__()

        if kernel_size % 2 != 1:
            raise ValueError("kernel_size should be odd for ring convolution.")

        self.padding = kernel_size // 2
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply circular depthwise convolution.

        Args:
            x: Tensor with shape (B, P, C).

        Returns:
            Tensor with shape (B, P, C).
        """
        if x.dim() != 3:
            raise ValueError("Input x must be a 3D tensor with shape (B, P, C).")

        if self.padding > 0:
            left = x[:, -self.padding :, :]
            right = x[:, : self.padding, :]
            x = torch.cat([left, x, right], dim=1)

        x = x.permute(0, 2, 1)
        x = self.depthwise_conv(x)
        x = x.permute(0, 2, 1)

        return x


class MambaEncoderBlock(nn.Module):
    """Mamba encoder block with FFN and residual normalization.

    The input and output shapes are both (B, P, D).
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation.

        Args:
            x: Tensor with shape (B, P, D).

        Returns:
            Tensor with shape (B, P, D).
        """
        y = self.mamba(x)
        y = self.norm1(x + y)

        ffn_out = self.ffn(y)
        out = self.norm2(y + ffn_out)

        return out


class CircularBiMambaBT(nn.Module):
    """Circular sequence boundary encoder based on Mamba.

    The module consists of:
        1. Input linear projection.
        2. Circular positional embedding.
        3. Ring depthwise convolution for local circular context.
        4. Stacked Mamba encoder blocks.
        5. Skip connection from projected features.
        6. MLP offset prediction head.

    Args:
        in_dim: Input feature dimension.
        hidden_dim: Hidden feature dimension.
        n_layers: Number of Mamba encoder blocks.
        d_state: State dimension of Mamba.
        d_conv: Convolution kernel size inside Mamba.
        expand: Expansion ratio inside Mamba.
        local_kernel: Kernel size of ring depthwise convolution.
        ffn_mult: Expansion ratio of FFN.

    Input:
        node_feats: Tensor with shape (B, P, C) or (B, C, P).

    Output:
        offsets: Tensor with shape (B, 2, P).
    """

    def __init__(
        self,
        in_dim: int = 36,
        hidden_dim: int = 128,
        n_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        local_kernel: int = 3,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.circular_pe = CircularPositionalEmbedding(hidden_dim)
        self.local_mix = RingDepthwiseConv1d(
            d_model=hidden_dim,
            kernel_size=local_kernel,
        )

        self.layers = nn.ModuleList(
            [
                MambaEncoderBlock(
                    d_model=hidden_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    ffn_mult=ffn_mult,
                )
                for _ in range(n_layers)
            ]
        )

        self.skip_proj = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=1,
            bias=True,
        )

        self.offset_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def _format_input(self, node_feats: torch.Tensor) -> torch.Tensor:
        """Convert input to shape (B, P, C)."""
        if node_feats.dim() != 3:
            raise ValueError(
                "node_feats must be a 3D tensor with shape (B, P, C) or (B, C, P)."
            )

        if node_feats.size(1) == self.in_dim:
            return node_feats.permute(0, 2, 1)

        if node_feats.size(-1) == self.in_dim:
            return node_feats

        raise ValueError(
            "The feature dimension of node_feats does not match in_dim. "
            f"Expected {self.in_dim}, but got shape {tuple(node_feats.shape)}."
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward propagation.

        Args:
            node_feats: Tensor with shape (B, P, C) or (B, C, P).
            adj: Unused argument, kept for compatibility with previous modules.

        Returns:
            Predicted point offsets with shape (B, 2, P).
        """
        del adj

        x = self._format_input(node_feats)

        x = self.input_proj(x)

        skip = self.skip_proj(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = self.circular_pe(x)
        x = x + self.local_mix(x)

        for layer in self.layers:
            x = layer(x)

        x = torch.cat([x, skip], dim=-1)

        offsets = self.offset_head(x)
        offsets = offsets.permute(0, 2, 1)

        return offsets