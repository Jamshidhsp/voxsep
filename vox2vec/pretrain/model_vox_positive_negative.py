from typing import *

import logging
logging.getLogger().setLevel(logging.WARNING)

import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl

from vox2vec.nn import Lambda
from vox2vec.nn.functional import select_from_pyramid, sum_pyramid_channels, scales_loss



class Contrastive_scale(nn.Module):
    def __init__(self, base_channels: int, num_scales: int, num_classes: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Conv3d(base_channels * 2 ** i, num_classes, kernel_size=1, bias=(i == 0))
            for i in range(num_scales)
        ])
    def forward(self, feature_pyramid: Sequence[torch.Tensor]) -> torch.Tensor:

        feature_pyramid = [layer(x) for x, layer in zip(feature_pyramid, self.layers)]
        return feature_pyramid





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
        self.contrastive_scales = Contrastive_scale(16, 6, 16)
    
    def preprocess_feature_map(self, patch, size=16):
        feature_pyramid = self.constrastive_loss(self.backbone(patch)[:])
        for i in range(len(feature_pyramid)):
            feature_pyramid[i]=feature_pyramid[i].mean(dim=(-3, -2,-1)) #e.g. (b, 128)
            feature_pyramid[i]=feature_pyramid[i].view(feature_pyramid[i].size(0), size, -1 ).unsqueeze(1)
            feature_pyramid[i]=feature_pyramid[i].transpose(1, 2).reshape(-1, size)

        # return torch.cat(tuple(fp for fp in feature_pyramid), dim=0)
        return torch.cat(tuple(fp.unsqueeze(1) for fp in feature_pyramid), dim=1)
    
    
    
    # def constrastive_scales_loss(self, features, tau=0.07):
    #     batch_size, num_scales, _, _ = features.shape
    #     features = features.view(batch_size * num_scales, -1)  # Reshape for simplicity: [batch_size * num_scales, feature_dim]
    #     features = F.normalize(features, p=2, dim=1) 
        
    #     # Compute similarity matrix
    #     sim_matrix = torch.matmul(features, features.t()) / tau
    #     sim_matrix = sim_matrix - torch.eye(batch_size * num_scales).to(features.device)
        
    #     # Create masks for positive and negative samples
    #     pos_mask = torch.block_diag(*[torch.ones(num_scales, num_scales) - torch.eye(num_scales) for _ in range(batch_size)]).to(features.device)
    #     neg_mask = 1 - pos_mask - torch.eye(batch_size * num_scales).to(features.device)
        
    #     # Compute loss
    #     exp_sim_matrix = torch.exp(sim_matrix) * neg_mask  # Apply negative mask
    #     sum_exp_sim_matrix = torch.sum(exp_sim_matrix, dim=1, keepdim=True)
        
    #     pos_sim = torch.log(torch.exp(sim_matrix)) * pos_mask
    #     pos_sim_sum = torch.sum(pos_sim, dim=1, keepdim=True)
        
    #     loss = -((pos_sim_sum+1e-5) / sum_exp_sim_matrix+1e-1)
    #     loss = torch.sum(loss) / (batch_size * num_scales)
        
    #     return loss
    
    
    
    
    

    def constrastive_scales_loss(self, features, tau=0.07):
        batch_size, num_scales, _, _ = features.shape
        features = features.view(batch_size * num_scales, -1)  # Reshape: [batch_size * num_scales, feature_dim]
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.t()) / tau
        
        # Initialize masks
        pos_mask = torch.zeros_like(sim_matrix)
        neg_mask = torch.ones_like(sim_matrix) - torch.eye(batch_size * num_scales).to(features.device)
        
        # Assign positive and negative masks
        for i in range(batch_size):
            for j in range(num_scales):
                pos_indices = [i * num_scales + k for k in range(num_scales) if k != j]
                pos_mask[i * num_scales + j, pos_indices] = 1
        
        # Adjust negative mask to exclude positive pairs
        neg_mask = neg_mask - pos_mask
        
        # Compute loss
        exp_sim_matrix = torch.exp(sim_matrix) * neg_mask  # Apply negative mask
        sum_exp_sim_matrix = torch.sum(exp_sim_matrix, dim=1, keepdim=True)
        
        pos_sim = torch.exp(sim_matrix) * pos_mask
        pos_sim_sum = torch.sum(pos_sim, dim=1, keepdim=True)
        
        loss = -torch.log((pos_sim_sum + 1e-5) / (sum_exp_sim_matrix + 1e-5))
        loss = torch.sum(loss) / (batch_size * num_scales)
        
        return loss

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def _vox_to_vec(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
        
        feature_pyramid = self.backbone(patches)
        return torch.cat([select_from_pyramid([x[j] for x in feature_pyramid], v) for j, v in enumerate(voxels)])
    
    
    
    
    def scales_loss_voxels(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
    
        feature_pyramid = self.backbone(patches)
        feature_pyramid= self.contrastive_scales(feature_pyramid)
        test = [scales_loss([x[j] for x in feature_pyramid], v) for j, v in enumerate(voxels)]
        test = [torch.stack(test[i]) for i in range(len(test))]
        test = torch.stack(test)
        test = self.constrastive_scales_loss(test)
        return  test

    
    
    
    
        
    def training_step(self, batch, batch_idx):
        """Computes Info-NCE loss.
        """
        patches_1, patches_2, voxels_1, voxels_2 = batch['pretrain']

        assert self.backbone.training
        assert self.proj_head.training

        embeds_1 = self.proj_head(self._vox_to_vec(patches_1, voxels_1))
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
        # print('losses', loss_1, loss_2)

        internal_loss_positive = self.scales_loss_voxels(patches_1, voxels_1)
        print('internal_loss_positive', internal_loss_positive)
        # internal_loss_negative = self.scales_loss_voxels(patches_2, voxels_2)
        # self.log(f'pretrain/info_nce_loss', loss, on_epoch=True)
        # self.log(f'pretrain/scales_loss_positive', internal_loss_positive, on_epoch=True)
        # self.log(f'pretrain/scales_loss_negative', internal_loss_negative, on_epoch=True)
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
