from typing import *

import torch
from torch import nn

from vox2vec.default_params import * 
from .blocks import ResBlock3d, StackMoreLayers, ResBlock3d_IterNorm



class FPN3d(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            base_channels: int, 
            num_scales: int, 
            deep: bool = False
    ) -> None:
        """Feature Pyramid Network (FPN) with 3D UNet architecture.

        Args:
            in_channels (int, optional):
                Number of input channels.
            out_channels (int, optional):
                Number of channels in the base of output feature pyramid.
            num_scales (int, optional):
                Number of pyramid levels.
            deep (bool):
                If True, add more layers at the bottom levels of UNet.
        """
        super().__init__()

        c = base_channels
        self.first_conv = nn.Conv3d(in_channels, c, kernel_size=3, padding=1)

        left_blocks, down_blocks, up_blocks, skip_blocks, right_blocks = [], [], [], [], []
        num_blocks = 2  # default
        for i in range(num_scales - 1):
            if deep:
                if i >= 2:
                    num_blocks = 4
                if i >= 4:
                    num_blocks = 8

            left_blocks.append(StackMoreLayers(ResBlock3d, [c] * (num_blocks + 1), kernel_size=3, padding=1))
            down_blocks.append(nn.Sequential(
                nn.MaxPool3d(kernel_size=2, ceil_mode=True),
                nn.Conv3d(c, c * 2, kernel_size=1)
                
            ))
            up_blocks.insert(0, nn.Sequential(
                # nn.Tanh(),
                nn.Conv3d(c * 2, c, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='nearest')
            ))
            skip_blocks.insert(0, nn.Conv3d(c, c, kernel_size=1))
            right_blocks.insert(0, StackMoreLayers(ResBlock3d, [c] * (num_blocks + 1), kernel_size=3, padding=1))
            c *= 2

        self.left_blocks = nn.ModuleList(left_blocks)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bottom_block = StackMoreLayers(ResBlock3d, [c] * (num_blocks + 1), kernel_size=3, padding=1)
        # self.bottom_block = StackMoreLayers(ResBlock3d_IterNorm, [c] * (num_blocks + 1), kernel_size=3, padding=1)
        self.up_blocks = nn.ModuleList(up_blocks)
        

        self.skip_blocks = nn.ModuleList(skip_blocks)
        self.right_blocks = nn.ModuleList(right_blocks)
        self.base_channels = base_channels
        self.num_scales = num_scales
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_conv(x)

        feature_pyramid = []
        for left, down in zip(self.left_blocks, self.down_blocks):
            x = left(x)
            feature_pyramid.append(x)
            x = down(x)

        x = (self.bottom_block(x))
        feature_pyramid.insert(0, x)

        for up, skip, right in zip(self.up_blocks, self.skip_blocks, self.right_blocks):
            x = up(x)
            fmap = feature_pyramid.pop()
            x = x[(..., *map(slice, fmap.shape[-3:]))]
            x += skip(fmap)  # skip connection
            x = (right(x))
            
            feature_pyramid.insert(0, x)

        return feature_pyramid



class FPN3d_iternorm(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            base_channels: int, 
            num_scales: int, 
            deep: bool = False
    ) -> None:
        """Feature Pyramid Network (FPN) with 3D UNet architecture.

        Args:
            in_channels (int, optional):
                Number of input channels.
            out_channels (int, optional):
                Number of channels in the base of output feature pyramid.
            num_scales (int, optional):
                Number of pyramid levels.
            deep (bool):
                If True, add more layers at the bottom levels of UNet.
        """
        super().__init__()

        c = base_channels
        self.first_conv = nn.Conv3d(in_channels, c, kernel_size=3, padding=1)

        left_blocks, down_blocks, up_blocks, skip_blocks, right_blocks = [], [], [], [], []
        num_blocks = 2  # default
        for i in range(num_scales - 1):
            if deep:
                if i >= 2:
                    num_blocks = 4
                if i >= 4:
                    num_blocks = 8

            # left_blocks.append(StackMoreLayers(ResBlock3d, [c] * (num_blocks + 1), kernel_size=3, padding=1))
            left_blocks.append(StackMoreLayers(ResBlock3d_IterNorm, [c] * (num_blocks + 1), kernel_size=3, padding=1))
            down_blocks.append(nn.Sequential(
                nn.MaxPool3d(kernel_size=2, ceil_mode=True),
                nn.Conv3d(c, c * 2, kernel_size=1)
                
            ))
            up_blocks.insert(0, nn.Sequential(
                # nn.Tanh(),
                nn.Conv3d(c * 2, c, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='nearest')
            ))
            skip_blocks.insert(0, nn.Conv3d(c, c, kernel_size=1))
            right_blocks.insert(0, StackMoreLayers(ResBlock3d_IterNorm, [c] * (num_blocks + 1), kernel_size=3, padding=1))

            c *= 2

        self.left_blocks = nn.ModuleList(left_blocks)
        self.down_blocks = nn.ModuleList(down_blocks)
        # self.bottom_block = StackMoreLayers(ResBlock3d, [c] * (num_blocks + 1), kernel_size=3, padding=1)
        self.bottom_block = StackMoreLayers(ResBlock3d_IterNorm, [c] * (num_blocks + 1), kernel_size=3, padding=1)
        self.up_blocks = nn.ModuleList(up_blocks)
        self.skip_blocks = nn.ModuleList(skip_blocks)
        self.right_blocks = nn.ModuleList(right_blocks)
        self.base_channels = base_channels
        self.num_scales = num_scales
        
        # self.right_blocks[4].layers[1].layers[5].weight = torch.nn.Parameter(torch.zeros_like(self.right_blocks[4].layers[1].layers[5].weight))
        # self.right_blocks[4].layers[1].layers[5].bias = torch.nn.Parameter(torch.zeros_like(self.right_blocks[4].layers[1].layers[5].bias))
        # self.right_blocks[4].layers[1].layers[4] = nn.Tanh()
        # self.right_blocks[2].layers[1].layers[2].weight = torch.nn.Parameter(torch.zeros_like(self.right_blocks[2].layers[1].layers[5].weight))
        # self.right_blocks[2].layers[1].layers[2].bias = torch.nn.Parameter(torch.zeros_like(self.right_blocks[2].layers[1].layers[5].bias))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_conv(x)

        feature_pyramid = []
        for left, down in zip(self.left_blocks, self.down_blocks):
            x = left(x)
            feature_pyramid.append(x)
            x = down(x)

        x = (self.bottom_block(x))
        feature_pyramid.insert(0, x)

        for up, skip, right in zip(self.up_blocks, self.skip_blocks, self.right_blocks):
            x = up(x)
            fmap = feature_pyramid.pop()
            x = x[(..., *map(slice, fmap.shape[-3:]))]
            x += skip(fmap)  # skip connection
            x = (right(x))
            
            feature_pyramid.insert(0, x)

        return feature_pyramid



class UNet3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        num_scales: int,
        deep: bool = False
    ) -> None:
        super(UNet3d, self).__init__()

        c = base_channels
        self.first_conv = nn.Conv3d(in_channels, c, kernel_size=3, padding=1)

        # Encoder and decoder blocks
        left_blocks, down_blocks, up_blocks, right_blocks, reduction_blocks = [], [], [], [], []
        num_blocks = 2  # default number of blocks

        for i in range(num_scales - 1):
            # Adjust depth of the blocks based on `deep` flag
            if deep:
                if i >= 2:
                    num_blocks = 4
                if i >= 4:
                    num_blocks = 8

            # Encoder (down-sampling path)
            left_blocks.append(StackMoreLayers(ResBlock3d, [c] * (num_blocks + 1), kernel_size=3, padding=1))
            down_blocks.append(nn.Sequential(
                nn.MaxPool3d(kernel_size=2, ceil_mode=True),
                nn.Conv3d(c, c * 2, kernel_size=1)
            ))

            # Decoder (up-sampling path)
            up_blocks.insert(0, nn.Sequential(
                nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
            ))

            # Reduction block after concatenation
            reduction_blocks.insert(0, nn.Conv3d(c * 2, c, kernel_size=3, padding=1))

            # Decoder right block
            right_blocks.insert(0, StackMoreLayers(ResBlock3d, [c] * (num_blocks + 1), kernel_size=3, padding=1))

            c *= 2

        # Final bottleneck at the bottom of the U-Net
        self.left_blocks = nn.ModuleList(left_blocks)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bottom_block = StackMoreLayers(ResBlock3d, [c] * (num_blocks + 1), kernel_size=3, padding=1)

        self.up_blocks = nn.ModuleList(up_blocks)
        self.reduction_blocks = nn.ModuleList(reduction_blocks)
        self.right_blocks = nn.ModuleList(right_blocks)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        enc_features = []  # To store features for skip connections

        # Encoder path (down-sampling)
        x = self.first_conv(x)
        for left, down in zip(self.left_blocks, self.down_blocks):
            x = left(x)
            enc_features.append(x)  # Store for skip connection
            x = down(x)

        # Bottleneck
        x = self.bottom_block(x)

        # Decoder path (up-sampling)
        feature_pyramid = [x]  # Collect features for a pyramid like FPN
        for up, reduce, right, enc_feature in zip(self.up_blocks, self.reduction_blocks, self.right_blocks, reversed(enc_features)):
            x = up(x)
            # Concatenate skip connections instead of addition
            x = torch.cat((x, enc_feature), dim=1)
            x = reduce(x)  # Reduce channels back to the correct number
            x = right(x)
            feature_pyramid.insert(0, x)  # Collect features for pyramid

        return feature_pyramid




class FPNLinearHead(nn.Module):
    def __init__(self, base_channels: int, num_scales: int, num_classes: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Conv3d(base_channels * 2 ** i, num_classes, kernel_size=1, bias=(i == 0))
            for i in range(num_scales)
        ])
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.num_scales = num_scales

    def forward(self, feature_pyramid: Sequence[torch.Tensor]) -> torch.Tensor:
        # assert len(feature_pyramid) == self.num_scales

        feature_pyramid = [layer(x) for x, layer in zip(feature_pyramid, self.layers)]

        x = feature_pyramid[-1]
        for fmap in reversed(feature_pyramid[:-1]):
            x = self.up(x)
            x = x[(..., *map(slice, fmap.shape[-3:]))]
            x += fmap
        return x


class FPNNonLinearHead(nn.Module):
    def __init__(self, base_channels: int, num_scales: int, num_classes: int) -> None:
        super().__init__()

        c = base_channels
        up_blocks, skip_blocks, right_blocks = [], [], []
        for _ in range(num_scales - 1):
            up_blocks.insert(0, nn.Sequential(
                nn.Conv3d(c * 2, c, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='nearest')
            ))
            skip_blocks.insert(0, nn.Conv3d(c, c, kernel_size=1))
            right_blocks.insert(0, ResBlock3d_IterNorm(c, c, kernel_size=1))
            c *= 2

        self.bottom_block = ResBlock3d_IterNorm(c, c, kernel_size=1)
        self.up_blocks = nn.ModuleList(up_blocks)
        self.skip_blocks = nn.ModuleList(skip_blocks)
        self.right_blocks = nn.ModuleList(right_blocks)
        self.final_block = nn.Conv3d(base_channels, num_classes, kernel_size=1)
        self.num_scales = num_scales

    def forward(self, feature_pyramid: Sequence[torch.Tensor]) -> torch.Tensor:
        assert len(feature_pyramid) == self.num_scales

        x = feature_pyramid[-1]
        x = self.bottom_block(x)
        for up, skip, right, fmap in zip(self.up_blocks, self.skip_blocks, self.right_blocks,
                                         reversed(feature_pyramid[:-1])):
            x = up(x)
            x = x[(..., *map(slice, fmap.shape[-3:]))]
            x += skip(fmap)  # skip connection
            x = right(x)

        x = self.final_block(x)

        return x
