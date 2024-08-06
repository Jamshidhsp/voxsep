from typing import *

import logging
logging.getLogger().setLevel(logging.WARNING)

import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl

from vox2vec.nn import Lambda
from vox2vec.nn.functional import select_from_pyramid, sum_pyramid_channels


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
        
        
        self.globale_proj_head = nn.Sequential(
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

    def _vox_to_vec(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor], local=True) -> torch.Tensor:
        feature_pyramid = self.backbone(patches)
        if local:
            return torch.cat([select_from_pyramid([x[j] for x in feature_pyramid], v) for j, v in enumerate(voxels)])
        else:
            averaged_feature_list = [torch.mean(feature_pyramid[i], dim=(2, 3, 4)) for i in range(len(feature_pyramid))]
            return torch.cat(averaged_feature_list, dim=1)

    def training_step(self, batch, batch_idx):
        """Computes Info-NCE loss.
        """
        patches_1, patches_2, voxels_1, voxels_2 = batch['pretrain']

        assert self.backbone.training
        assert self.proj_head.training

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
        
        embeds_1_global = self._vox_to_vec(patches_1, voxels_1, local=False)
        embeds_2_global = self._vox_to_vec(patches_2, voxels_2, local=False)
        
        print('losses', loss_1, loss_2)
        global_loss = torch.matmul(embeds_1_global, embeds_2_global.T).diagonal()
        print('global_loss_mean', torch.mean((global_loss)/(torch.mean(embeds_1_global)+torch.mean(embeds_2_global))))
        gloss = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_11, logits_12], dim=1), dim=1))
        print('gloss', gloss)

        self.log(f'pretrain/info_nce_loss', loss, on_epoch=True)
        self.log(f'pretrain/global_loss', gloss, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not self.trainer.training:
            # skip device transfer for the val dataloader
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
