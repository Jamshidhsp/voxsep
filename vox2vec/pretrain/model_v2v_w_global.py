from typing import *

import logging
logging.getLogger().setLevel(logging.WARNING)

import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl

from vox2vec.nn import Lambda
from vox2vec.nn.functional import select_from_pyramid, sum_pyramid_channels
import numpy as np

class Global_projector(nn.Module):
        def __init__(self):
            super(Global_projector, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.conv = nn.Conv3d(512, 1, 1, 1)
            self.linear1 = nn.Linear(16, 8)
            self.linear2 = nn.Linear(8, 1)
            self.relu = nn.LeakyReLU()
        def forward(self, x):
            x = self.conv(x)
            # x = self.avg_pool(x).view(x.shape[0], -1)
            x = self.linear1(x.view(x.shape[0], -1))
            x = self.relu(x)
            x = self.linear2(x)
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
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
            Lambda(F.normalize)
        )

        self.temp = temp
        self.lr = lr

        self.global_projector = Global_projector()

    def _vox_to_vec(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
        feature_pyramid = self.backbone(patches)
        return torch.cat([select_from_pyramid([x[j] for x in feature_pyramid], v) for j, v in enumerate(voxels)])

    def training_step(self, batch, batch_idx):
        """Computes Info-NCE loss.
        """
        patches_1, patches_2, voxels_1, voxels_2 = batch['pretrain']

        assert self.backbone.training
        assert self.proj_head.training


        patches_1_negative = patches_1.clone()  
        
        max_shuffle_size=int(0.5*patches_1.size(-1))
        for i in range(patches_1.size(0)):
            # patches_1_negative_list = []
            for _ in range(np.random.randint(1, 10)):
                size = np.random.randint(4, max_shuffle_size)
                # size = 8
                src_h, src_w, src_d = np.random.randint(0, patches_1.size(-1)-size, 3)
                des_h, des_w, des_d =  np.random.randint(0, patches_1.size(-1)-size, 3)
                patches_1_negative[i, 0, src_h:src_h+size, src_w:src_w+size, src_d:src_d+size] = patches_1_negative[i, 0, des_h:des_h+size, des_w:des_w+size, des_d:des_d+size]
                # patches_1_negative_list.append(patches_1_negative)
        


        global_positive = self.global_projector(self.backbone(patches_1)[-1])
        global_negative = self.global_projector(self.backbone(patches_1_negative)[-1])
        global_logits = torch.cat((global_positive, global_negative), dim=1)
        labels = torch.arange(2.0).repeat(global_logits.shape[0], 1).to(global_logits.device)
        global_loss = F.binary_cross_entropy_with_logits(global_logits, labels)



        embeds_1 = self.proj_head(self._vox_to_vec(patches_1, voxels_1))
        embeds_2 = self.proj_head(self._vox_to_vec(patches_2, voxels_2))

        logits_11 = torch.matmul(embeds_1, embeds_1.T) / self.temp
        logits_11.fill_diagonal_(float('-inf'))
        logits_12 = torch.matmul(embeds_1, embeds_2.T) / self.temp
        logits_22 = torch.matmul(embeds_2, embeds_2.T) / self.temp
        logits_22.fill_diagonal_(float('-inf'))
        loss_1 = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_11, logits_12], dim=1), dim=1))
        loss_2 = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_12.T, logits_22], dim=1), dim=1))
        loss = (loss_1 + loss_2) / 2

        self.log(f'pretrain/info_nce_loss', loss, on_epoch=True)

        return loss + global_loss

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not self.trainer.training:
            # skip device transfer for the val dataloader
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
