# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import modulus
import torch
import torch.nn.functional as F
from modulus.models.afno import AFNO

class WrapConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        **kwargs
    ):
        self.pad_size = kernel_size // 2
        padding = (0, 0)
        super().__init__(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs
        )

    def forward(self, x):
        left_padding = x[..., -self.pad_size:]
        right_padding = x[..., :self.pad_size]
        x = torch.concat([left_padding, x, right_padding], dim=-1)

        upper_padding = torch.flip(x[:, :, -self.pad_size:, :], dims=[2])
        lower_padding = torch.flip(x[:, :, :self.pad_size, :], dims=[2])
        x = torch.concat([lower_padding, x, upper_padding], dim=-2)
        return super().forward(x)


class WrapConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_kernel_size: int,
        spatial_kernel_size: int,
        **kwargs
    ):
        padding = (0, 0, 0)
        self.spatial_pad_size = spatial_kernel_size // 2
        kernel_size = (time_kernel_size, spatial_kernel_size, spatial_kernel_size)
        super().__init__(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs
        )

    def forward(self, x):
        left_padding = x[..., -self.spatial_pad_size:]
        right_padding = x[..., :self.spatial_pad_size]
        x = torch.concat([left_padding, x, right_padding], dim=-1)

        upper_padding = torch.flip(x[:, :, :, -self.spatial_pad_size:, :], dims=[3])
        lower_padding = torch.flip(x[:, :, :, :self.spatial_pad_size, :], dims=[3])
        x = torch.concat([lower_padding, x, upper_padding], dim=-2)
        return super().forward(x)


class PatchEmbed(nn.Module):
    """Patch embedding layer

    Converts 2D/3D patch into a 1D vector for input to AFNO

    Parameters
    ----------
    inp_shape : List[int]
        Input image dimensions [height, width]
    in_channels : int
        Number of input channels
    patch_size : List[int], optional
        Size of image patches, by default [16, 16]
    embed_dim : int, optional
        Embedded channel size, by default 256
    """

    def __init__(
        self,
        inp_shape: List[int],
        in_channels: int,
        patch_size: List[int] = [16, 16],
        embed_dim: int = 256,
    ):
        super().__init__()
        if len(inp_shape) not in (2, 3):
            raise ValueError("inp_shape should be a list of length 2 or 3")
        if len(patch_size) not in (2, 3):
            raise ValueError("patch_size should be a list of length 2 or 3")

        self.dim = len(inp_shape)
        if self.dim == 2:
            num_patches = ((inp_shape[1] // patch_size[1]) + 1) * ((inp_shape[0] // patch_size[0]) + 1)
        else:
            num_patches = ((inp_shape[2] // patch_size[2]) + 1) * ((inp_shape[1] // patch_size[1]) + 1) * (inp_shape[0] // patch_size[0]) 
        self.inp_shape = inp_shape
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        if self.dim == 2:        
            self.proj = WrapConv2d(
                in_channels, embed_dim, kernel_size=patch_size[0], stride=patch_size
            )
        elif self.dim == 3:
            self.proj = nn.Sequential()
            self.proj.add_module("conv_1",
                WrapConv3d(
                    in_channels, embed_dim, time_kernel_size=patch_size[0], spatial_kernel_size=patch_size[1], stride=patch_size
                    )
            )
            self.proj.add_module("flatten_1",
                nn.Flatten(start_dim=1, end_dim=2)
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.dim == 2:
            B, C, H, W = x.shape
        elif self.dim == 3:
            B, C, T, H, W = x.shape
        if not (H == self.inp_shape[-2] and W == self.inp_shape[-1]):
            raise ValueError(
                f"Input image size ({H}*{W}) doesn't match model ({self.inp_shape[-2]}*{self.inp_shape[-1]})."
            )
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class SolarRadiationNet(Module):
    """Adaptive Fourier neural operator (AFNO) model for solar radiation prediction.

    Parameters
    ----------
    inp_shape : List[int]
        Input image dimensions [height, width]
    in_channels : int
        Number of input channels
    out_channels: int
        Number of output channels
    patch_size : List[int], optional
        Size of image patches, by default [16, 16]
    embed_dim : int, optional
        Embedded channel size, by default 256
    depth : int, optional
        Number of AFNO layers, by default 4
    mlp_ratio : float, optional
        Ratio of layer MLP latent variable size to input feature size, by default 4.0
    drop_rate : float, optional
        Drop out rate in layer MLPs, by default 0.0
    num_blocks : int, optional
        Number of blocks in the block-diag frequency weight matrices, by default 16
    sparsity_threshold : float, optional
        Sparsity threshold (softshrink) of spectral features, by default 0.01
    hard_thresholding_fraction : float, optional
        Threshold for limiting number of modes used [0,1], by default 1
    """

    def __init__(
        self,
        inp_shape: List[int],
        in_channels: int,
        out_channels: int,
        patch_size: List[int] = [16, 16],
        embed_dim: int = 256,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        num_blocks: int = 16,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
    ) -> None:
        super().__init__(meta=MetaData())
        self.dim = len(inp_shape)
        if self.dim not in (2, 3):
            raise ValueError("inp_shape should be a list of length 2 or 3")
        if self.dim not in (2, 3):
            raise ValueError("patch_size should be a list of length 2 or 3")

        
        for i in range(self.dim):
            if not (
                inp_shape[i] % patch_size[i] == 0
            ):
                raise ValueError(
                    f"input shape {inp_shape} should be divisible by patch_size {patch_size}"
                )

        self.in_chans = in_channels
        self.out_chans = out_channels
        self.inp_shape = inp_shape
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            inp_shape=inp_shape,
            in_channels=self.in_chans,
            patch_size=self.patch_size,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.h = inp_shape[-2] // self.patch_size[-2] + 1
        self.w = inp_shape[-1] // self.patch_size[-1] + 1

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim=embed_dim,
                    num_blocks=self.num_blocks,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    norm_layer=norm_layer,
                    sparsity_threshold=sparsity_threshold,
                    hard_thresholding_fraction=hard_thresholding_fraction,
                )
                for i in range(depth)
            ]
        )

        self.head = nn.Linear(
            embed_dim,
            self.out_chans * self.patch_size[-2] * self.patch_size[-1],
            bias=False,
        )

        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Init model weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # What is this for
    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {"pos_embed", "cls_token"}

    def forward_features(self, x: Tensor) -> Tensor:
        """Forward pass of core AFNO"""
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        
        # Correct tensor shape back into [B, C, H, W]
        # [b h w (p1 p2 c_out)]
        x = x.view(list(x.shape[:-1]) + [self.patch_size[-2], self.patch_size[-1], -1])
        # [b h w p1 p2 c_out]
        x = torch.permute(x, (0, 5, 1, 3, 2, 4))
        # [b c_out, h, p1, w, p2]
        x = x.reshape(list(x.shape[:2]) + [self.inp_shape[-2]+self.patch_size[-2], self.inp_shape[-1]+self.patch_size[-1]])
        # [b c_out, (h*p1), (w*p2)]
        return x[:, :, self.patch_size[-2]//2:-self.patch_size[-2]//2, self.patch_size[-1]//2:-self.patch_size[-1]//2]