from typing import *

import logging
logging.getLogger().setLevel(logging.WARNING)

import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl

from vox2vec.nn import Lambda
from vox2vec.nn.functional import select_from_pyramid, sum_pyramid_channels
from vox2vec.nn import FPNLinearHead

# augmentation samples
from vox2vec.pretrain.my_transormations import sample_view
from vox2vec.pretrain.my_transormations import rot_rand
from vox2vec.pretrain.my_transormations import get_non_overlapping_crops


class Upsample_scales(nn.Module):
    def __init__(self, base_channels: int, num_scales: int, num_classes: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Conv3d(base_channels * 2 ** i, num_classes, kernel_size=1, bias=(i == 0))
            # nn.Conv3d(base_channels * 1 ** i, num_classes, kernel_size=1, bias=(i == 0))
            for i in range(num_scales)
        ])
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.num_scales = num_scales

    def forward(self, feature_pyramid: Sequence[torch.Tensor]) -> torch.Tensor:
        assert len(feature_pyramid) == self.num_scales

        feature_pyramid = [layer(x) for x, layer in zip(feature_pyramid, self.layers)]

        x = feature_pyramid[-1]
        for fmap in reversed(feature_pyramid[:-1]):
            x = self.up(x)
            x = x[(..., *map(slice, fmap.shape[-3:]))]
            x += fmap
        return x




class Vox2Vec(pl.LightningModule):
    def __init__(
            self,
            backbone: nn.Module,
            base_channels: int,
            num_scales: int,
            proj_dim: int = 128,
            temp: float = 0.1,
            lr: float = 3e-4,
    ):
        """vox2vec model.

        Args:
            backbone (nn.Module):
                Takes an image of size ``(n, c, h, w, d)`` and returns a feature pyramid of sizes
                ``[(n, c_b, h_b, w_b, d_b), (n, c_b * 2, h_b // 2, w_b // 2, d_b // 2), ...]``,
                where ``c_b = base_channels`` and ``(h_b, w_b, d_b) = (h, w, d)``.
            base_channels (int):
                A number of channels in the base of the output feature pyramid.
            num_scales (int):
                A number of feature pyramid levels.
            proj_dim (int, optional):
                The output dimensionality of the projection head. Defaults to 128.
            temp (float, optional):
                Info-NCE loss temperature. Defaults to 0.1.
            lr (float, optional):
                Learning rate. Defaults to 3e-4.
        """
        super().__init__()

        self.save_hyperparameters(ignore='backbone')

        self.backbone = backbone
        embed_dim = sum_pyramid_channels(base_channels, num_scales)
        # embed_dim = 128
        embed_dim = 1024
        # embed_dim= 524288
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, proj_dim),
            # Lambda(F.normalize)
        )

        self.local_proj_head = nn.Sequential(
            # nn.Linear(8192, 128),
            # nn.Linear(8192, 8192),
            # nn.BatchNorm1d(8192),
            # nn.ReLU(),
            nn.Linear(8192, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Lambda(F.normalize)
        )

        self.temp = temp
        self.lr = lr

    def _vox_to_vec(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
        feature_pyramid = self.backbone(patches)
        return torch.cat([select_from_pyramid([x[j] for x in feature_pyramid], v) for j, v in enumerate(voxels)])
    




    # def crops_return(self, patches: torch.Tensor) -> torch.Tensor:
    #     layer = Upsample_scales(16, 6, 16).to('cuda')
    #     feature_pyramid = self.backbone(patches)
    #     feature_map = layer(feature_pyramid)
        
    #     return (get_non_overlapping_crops(feature_map, (1, 1, 32, 32, 8))), feature_map[0].view(feature_map[0].size(0), -1)


    def local_positive_crops(self, patches: torch.Tensor) -> torch.Tensor:
        layer = Upsample_scales(16, 6, 16).to('cuda')
        feature_pyramid = self.backbone(patches)
        feature_map = layer(feature_pyramid)
        
        return(get_non_overlapping_crops(feature_map, (1, 1, 32, 32, 8)))
        # return(get_non_overlapping_crops(feature_map, (1, 1, 8, 8, 2)))
    


    def global_contrastive_loss(self, patches: torch.Tensor) -> torch.Tensor:
        layer = Upsample_scales(128, 3, 1).to('cuda')
        feature_pyramid = self.backbone(patches)[-3:]
        feature_map = layer(feature_pyramid)
        feature_map_mean = torch.mean(feature_map, axis=1)
        return feature_map_mean.view(feature_map_mean.size(0), -1)



    # def global_contrastive_loss(self, patches: torch.Tensor) -> torch.Tensor:
    #     feature_pyramid = self.backbone(patches)
    #     feature_map = feature_pyramid[3][0]
    #     # feature_map = feature_pyramid[0][0]
    #     return feature_map.view(feature_map.size(0), -1)
    #     # return feature_map.view(1, 128)

    def training_step(self, batch, batch_idx):
        """Computes Info-NCE loss.
        """
        # patches_1, patches_2, voxels_1, voxels_2 = batch['pretrain']
        patches_1, patches_1_positive, patches_2, _, _ = batch['pretrain']

        assert self.backbone.training
        assert self.proj_head.training
        # patches_2 = patches_2.flip(dims=[-1])



        original_crops = self.local_positive_crops(patches_1)
        original_crops_positive = self.local_positive_crops(patches_1_positive)

        # Flatten the last three dimensions for processing
        # tensor1_flat = original_crops.view(original_crops.size(0),-1)  # Shape: [1024, 32*32*8]
        # tensor2_flat = original_crops_positive.view(original_crops_positive.size(0), -1)  # Shape: [1024, 32*32*8]

        tensor1_flat = self.local_proj_head(original_crops.view(original_crops.size(0), -1))
        tensor2_flat = self.local_proj_head(original_crops_positive.view(original_crops.size(0), -1))

        temperature = 0.9

        # Normalize the vectors (important for InfoNCE)
        tensor1_norm = F.normalize(tensor1_flat, p=2, dim=1)
        tensor2_norm = F.normalize(tensor2_flat, p=2, dim=1)
        similarity_matrix = torch.matmul(tensor1_norm, tensor2_norm.T) / temperature

        # Create labels for the positive pairs
        labels = torch.arange(original_crops.size(0)).to(original_crops.device)
        # Calculate the InfoNCE loss
        info_nce_loss_local = F.cross_entropy(similarity_matrix, labels)
        embeds_1 = self.proj_head(self.global_contrastive_loss(patches_1))
        embeds_2 = self.proj_head(self.global_contrastive_loss(patches_2))



        tensor1_flat_global = embeds_1.view(embeds_1.size(0),-1)  # Shape: [1024, 32*32*8]
        tensor2_flat_global = embeds_2.view(embeds_2.size(0),-1)  # Shape: [1024, 32*32*8]
        tensor1_norm_global = F.normalize(tensor1_flat_global, p=2, dim=1)
        tensor2_norm_global = F.normalize(tensor2_flat_global, p=2, dim=1)

        similarity_matrix = torch.matmul(tensor1_norm_global, tensor2_norm_global.T) / temperature
        labels = torch.arange(embeds_1.size(0)).to(embeds_1.device)
        info_nce_loss_global = F.cross_entropy(similarity_matrix, labels)
        loss = (info_nce_loss_local + info_nce_loss_global) / 2
        print('losses', info_nce_loss_local.item(), info_nce_loss_global.item())

        # self.log(f'pretrain/info_nce_loss', loss, on_epoch=True)

        return info_nce_loss_local

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not self.trainer.training:
            # skip device transfer for the val dataloader
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
