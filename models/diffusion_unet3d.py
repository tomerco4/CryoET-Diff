# models/diffusion_unet3d.py

from typing import Tuple

import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard sinusoidal position/time embedding.

    Input: t (B,) integer or float in [0, T]
    Output: (B, dim)
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Time embedding dimension must be even.")
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (...,) tensor of timesteps
        Returns:
            emb: (..., dim) tensor
        """
        half_dim = self.dim // 2
        # log(10000) / (half_dim - 1)
        factor = math.log(10000.0) / (half_dim - 1)
        # (..., 1)
        t = t.float().unsqueeze(-1)
        # (half_dim,)
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * -factor)
        # (..., half_dim)
        args = t * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class TimeClassLocationEmbedding(nn.Module):
    """
    Combine time, class-onehot, and (x,y,z) location into a single
    conditioning vector of size cond_dim.
    """

    def __init__(self, num_classes: int, cond_dim: int = 128):
        super().__init__()
        self.cond_dim = cond_dim

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.class_mlp = nn.Linear(num_classes, cond_dim)
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, t: Tensor, class_onehot: Tensor, coords: Tensor) -> Tensor:
        """
        Args:
            t: (B,) timesteps
            class_onehot: (B, C)
            coords: (B, 3) normalized coordinates

        Returns:
            cond: (B, cond_dim)
        """
        t_emb = self.time_mlp(t)           # (B, cond_dim)
        c_emb = self.class_mlp(class_onehot)  # (B, cond_dim)
        p_emb = self.coord_mlp(coords)     # (B, cond_dim)
        return t_emb + c_emb + p_emb       # (B, cond_dim)


class CondResBlock3D(nn.Module):
    """
    3D residual block with FiLM-like additive conditioning.

    - GroupNorm + SiLU + Conv3d
    - Add projected cond vector
    - GroupNorm + SiLU + Conv3d
    - Residual connection

    Input: x (B, Cin, D, H, W), cond (B, cond_dim)
    Output: (B, Cout, D, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        groups: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        # project cond into per-channel bias
        self.cond_proj = nn.Linear(cond_dim, out_channels)

        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: (B, Cin, D, H, W)
            cond: (B, cond_dim)
        """
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        # cond bias
        cond_bias = self.cond_proj(cond)  # (B, Cout)
        # reshape to (B, Cout, 1, 1, 1)
        cond_bias = cond_bias[..., None, None, None]
        h = h + cond_bias

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        return h + self.skip(x)


class Downsample3D(nn.Module):
    """Simple 3D downsample by factor 2 via strided conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(
            channels, channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    """Simple 3D upsample by factor 2 via ConvTranspose3d."""

    def __init__(self, channels: int):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(
            channels, channels, kernel_size=2, stride=2
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.deconv(x)


class DiffusionUNet3D(nn.Module):
    """
    3D U-Net backbone for DDPM.

    Input:
      x: (B, 1, D, H, W)
      t: (B,)
      class_onehot: (B, num_classes)
      coords: (B, 3)
    Output:
      pred: (B, 1, D, H, W)  (predicted clean volume x0)
    """

    def __init__(
        self,
        num_classes: int,
        cond_dim: int = 128,
        base_channels: int = 32,
        in_channels: int = 1,
        out_channels: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cond_dim = cond_dim

        # conditioning
        self.cond_emb = TimeClassLocationEmbedding(
            num_classes=num_classes,
            cond_dim=cond_dim,
        )

        # Encoder
        self.enc1 = CondResBlock3D(in_channels, base_channels, cond_dim, groups=in_channels)
        self.down1 = Downsample3D(base_channels)

        self.enc2 = CondResBlock3D(base_channels, base_channels * 2, cond_dim)
        self.down2 = Downsample3D(base_channels * 2)

        self.enc3 = CondResBlock3D(base_channels * 2, base_channels * 4, cond_dim)
        self.down3 = Downsample3D(base_channels * 4)

        # Bottleneck
        self.bot = CondResBlock3D(base_channels * 4, base_channels * 4, cond_dim)

        # Decoder
        self.up1 = Upsample3D(base_channels * 4)
        self.dec1 = CondResBlock3D(base_channels * 4 + base_channels * 4,
                                   base_channels * 2,
                                   cond_dim)

        self.up2 = Upsample3D(base_channels * 2)
        self.dec2 = CondResBlock3D(base_channels * 2 + base_channels * 2,
                                   base_channels,
                                   cond_dim)

        self.up3 = Upsample3D(base_channels)
        self.dec3 = CondResBlock3D(base_channels + base_channels,
                                   base_channels,
                                   cond_dim)

        # Output conv
        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        class_onehot: Tensor,
        coords: Tensor,
    ) -> Tensor:
        """
        Args:
            x: (B, 1, D, H, W)
            t: (B,)
            class_onehot: (B, num_classes)
            coords: (B, 3)
        Returns:
            (B, 1, D, H, W)
        """
        cond = self.cond_emb(t, class_onehot, coords)  # (B, cond_dim)

        # Encoder
        h1 = self.enc1(x, cond)            # -> (B, C, D, H, W)
        h2_in = self.down1(h1)
        h2 = self.enc2(h2_in, cond)
        h3_in = self.down2(h2)
        h3 = self.enc3(h3_in, cond)
        hb_in = self.down3(h3)
        hb = self.bot(hb_in, cond)

        # Decoder
        d1_up = self.up1(hb)
        d1 = torch.cat([d1_up, h3], dim=1)
        d1 = self.dec1(d1, cond)

        d2_up = self.up2(d1)
        d2 = torch.cat([d2_up, h2], dim=1)
        d2 = self.dec2(d2, cond)

        d3_up = self.up3(d2)
        d3 = torch.cat([d3_up, h1], dim=1)
        d3 = self.dec3(d3, cond)

        out = self.out_conv(d3)
        return out
