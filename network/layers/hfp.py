# -*- coding: utf-8 -*-
"""High-Frequency Perception Module.

This file implements the High-Frequency Perception (HFP) module used for
enhancing high-frequency text details. It contains:

    1. ConvModule
    2. DctSpatialInteraction
    3. DctChannelInteraction
    4. HFP
    5. SDP

The implementation removes the dependency on mmcv and uses pure PyTorch.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as DCT
from einops import rearrange


def build_high_frequency_mask(
    height: int,
    width: int,
    ratio: Tuple[float, float],
    device: torch.device,
) -> torch.Tensor:
    """Build a DCT high-frequency mask.

    The low-frequency region in the top-left corner is set to 0, and the
    remaining high-frequency region is set to 1.

    Args:
        height: Feature map height.
        width: Feature map width.
        ratio: Ratio of the low-frequency region.
        device: Device of the input tensor.

    Returns:
        A tensor with shape (1, height, width).
    """
    h_cut = int(height * ratio[0])
    w_cut = int(width * ratio[1])

    mask = torch.ones((height, width), dtype=torch.float32, device=device)
    mask[:h_cut, :w_cut] = 0.0

    return mask.view(1, height, width)


class ConvModule(nn.Sequential):
    """A lightweight Conv-BN-ReLU block.

    This module is used as a replacement for mmcv.cnn.ConvModule.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        super().__init__(*layers)


class DctSpatialInteraction(nn.Module):
    """Spatial interaction branch based on DCT high-frequency filtering."""

    def __init__(
        self,
        in_channels: int,
        ratio: Tuple[float, float],
        is_dct: bool = True,
    ) -> None:
        super().__init__()

        self.ratio = ratio
        self.is_dct = is_dct

        if not self.is_dct:
            self.spatial_proj = ConvModule(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=1,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation.

        Args:
            x: Input tensor with shape (B, C, H, W).

        Returns:
            Enhanced tensor with shape (B, C, H, W).
        """
        if not self.is_dct:
            spatial_weight = torch.sigmoid(self.spatial_proj(x))
            return x * spatial_weight

        _, _, height, width = x.size()

        dct_feat = DCT.dct_2d(x, norm="ortho")
        mask = build_high_frequency_mask(
            height=height,
            width=width,
            ratio=self.ratio,
            device=x.device,
        )
        mask = mask.expand_as(dct_feat)

        high_freq = dct_feat * mask
        spatial_weight = DCT.idct_2d(high_freq, norm="ortho")

        return x * spatial_weight


class DctChannelInteraction(nn.Module):
    """Channel interaction branch based on DCT high-frequency filtering."""

    def __init__(
        self,
        in_channels: int,
        patch: Tuple[int, int],
        ratio: Tuple[float, float],
        is_dct: bool = True,
        groups: int = 32,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.patch = patch
        self.ratio = ratio
        self.is_dct = is_dct

        self.channel_proj_1 = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            groups=groups,
            bias=False,
        )

        self.channel_proj_2 = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            groups=groups,
            bias=False,
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation.

        Args:
            x: Input tensor with shape (B, C, H, W).

        Returns:
            Enhanced tensor with shape (B, C, H, W).
        """
        batch_size, channels, height, width = x.size()

        if not self.is_dct:
            max_pool = F.adaptive_max_pool2d(x, output_size=(1, 1))
            avg_pool = F.adaptive_avg_pool2d(x, output_size=(1, 1))

            channel_feat = self.channel_proj_1(self.relu(max_pool))
            channel_feat = channel_feat + self.channel_proj_1(self.relu(avg_pool))
            channel_weight = torch.sigmoid(self.channel_proj_2(channel_feat))

            return x * channel_weight

        dct_feat = DCT.dct_2d(x, norm="ortho")
        mask = build_high_frequency_mask(
            height=height,
            width=width,
            ratio=self.ratio,
            device=x.device,
        )
        mask = mask.expand_as(dct_feat)

        high_freq = dct_feat * mask
        high_freq = DCT.idct_2d(high_freq, norm="ortho")

        max_pool = F.adaptive_max_pool2d(high_freq, output_size=self.patch)
        avg_pool = F.adaptive_avg_pool2d(high_freq, output_size=self.patch)

        max_pool = torch.sum(self.relu(max_pool), dim=[2, 3]).view(
            batch_size,
            channels,
            1,
            1,
        )
        avg_pool = torch.sum(self.relu(avg_pool), dim=[2, 3]).view(
            batch_size,
            channels,
            1,
            1,
        )

        channel_feat = self.channel_proj_1(max_pool) + self.channel_proj_1(avg_pool)
        channel_weight = torch.sigmoid(self.channel_proj_2(channel_feat))

        return x * channel_weight


class HFP(nn.Module):
    """High-Frequency Perception module.

    HFP enhances high-frequency text details by combining spatial and channel
    interactions in the DCT domain.
    """

    def __init__(
        self,
        in_channels: int,
        ratio: Tuple[float, float],
        patch: Tuple[int, int] = (8, 8),
        is_dct: bool = True,
        groups: int = 32,
    ) -> None:
        super().__init__()

        self.spatial = DctSpatialInteraction(
            in_channels=in_channels,
            ratio=ratio,
            is_dct=is_dct,
        )

        self.channel = DctChannelInteraction(
            in_channels=in_channels,
            patch=patch,
            ratio=ratio,
            is_dct=is_dct,
            groups=groups,
        )

        self.out = nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(groups, in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation.

        Args:
            x: Input tensor with shape (B, C, H, W).

        Returns:
            Enhanced tensor with shape (B, C, H, W).
        """
        spatial_feat = self.spatial(x)
        channel_feat = self.channel(x)

        return self.out(spatial_feat + channel_feat)


class SDP(nn.Module):
    """Semantic decoupling projection module.

    This module is kept for compatibility with previous experiments. It uses
    patch-wise attention to enhance low-level features with high-level features.
    """

    def __init__(
        self,
        dim: int = 256,
        inter_dim: Optional[int] = None,
        groups: int = 32,
    ) -> None:
        super().__init__()

        self.inter_dim = inter_dim if inter_dim is not None else dim

        self.conv_q = nn.Sequential(
            ConvModule(
                in_channels=dim,
                out_channels=self.inter_dim,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            nn.GroupNorm(groups, self.inter_dim),
        )

        self.conv_k = nn.Sequential(
            ConvModule(
                in_channels=dim,
                out_channels=self.inter_dim,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            nn.GroupNorm(groups, self.inter_dim),
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x_low: torch.Tensor,
        x_high: torch.Tensor,
        patch_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Forward propagation.

        Args:
            x_low: Low-level feature map with shape (B, C, H, W).
            x_high: High-level feature map with shape (B, C, H, W).
            patch_size: Patch size used for local attention.

        Returns:
            Enhanced low-level feature map with shape (B, C, H, W).
        """
        batch_size, _, height, width = x_low.size()
        patch_h, patch_w = patch_size

        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(
                "The height and width of x_low must be divisible by patch_size. "
                f"Got feature size ({height}, {width}) and patch size {patch_size}."
            )

        q = rearrange(
            self.conv_q(x_low),
            "b c (h p1) (w p2) -> (b h w) c (p1 p2)",
            p1=patch_h,
            p2=patch_w,
        )
        q = q.transpose(1, 2)

        k = rearrange(
            self.conv_k(x_high),
            "b c (h p1) (w p2) -> (b h w) c (p1 p2)",
            p1=patch_h,
            p2=patch_w,
        )

        attention = torch.matmul(q, k)
        attention = attention / np.sqrt(float(self.inter_dim))
        attention = self.softmax(attention)

        v = k.transpose(1, 2)
        out = torch.matmul(attention, v)

        out = rearrange(
            out.transpose(1, 2).contiguous(),
            "(b h w) c (p1 p2) -> b c (h p1) (w p2)",
            b=batch_size,
            h=height // patch_h,
            w=width // patch_w,
            p1=patch_h,
            p2=patch_w,
        )

        return out + x_low