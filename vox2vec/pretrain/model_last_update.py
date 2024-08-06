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

from vox2vec.pretrain.my_transormations import get_non_overlapping_crops
import numpy as np

class Upsample_scales(nn.Module):
    def __init__(self, base_channels: int, num_scales: int, num_classes: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Conv3d(base_channels * 2 ** i, num_classes, kernel_size=1, bias=(i == 0))
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





# contrastive loss

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
            # proj_dim: int = 128,
            proj_dim: int = 1024,
            temp: float = 0.1,
            # lr: float = 3e-4,
            lr: float = 1e-5,
    ):

        super().__init__()

        self.save_hyperparameters(ignore='backbone')

        self.backbone = backbone
        
        # embed_dim = sum_pyramid_channels(base_channels, num_scales)
        # embed_dim = 128
        embed_dim = 128
        # local_emb_dim = 8192
        local_emb_dim = 16
        proj_dim_global = 128
        proj_dim_local = 16
        # proj_dim_local = 8192
        # embed_dim= 524288
        
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim_global),

            Lambda(F.normalize)
        )

        # self.local_proj_head = nn.Sequential(
        #     # nn.Linear(local_emb_dim, 1024),
        #     # nn.Linear(local_emb_dim, 1024),
        #     # nn.BatchNorm1d(1024),
        #     # nn.ReLU(),
        #     # nn.Linear(1024, proj_dim_local),
        #     # nn.BatchNorm1d(proj_dim),
            
        #     nn.Linear(local_emb_dim, local_emb_dim),
        #     nn.BatchNorm1d(local_emb_dim),
        #     nn.ReLU(),
        #     nn.Linear(local_emb_dim, 16),
            

        #     Lambda(F.normalize)
        # )
        
        # self.local_upsample_scales = Upsample_scales(16, 6, 16)
        # self.local_upsample_scales = Upsample_scales(16, 6, 512)
        
        
        # self.global_upsample_scales = Upsample_scales(128, 3, 1)
        

        self.constrastive_loss = Contrastive_scale(16, 6, 16)
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        self.temp = temp
        self.lr = lr

    # def _vox_to_vec(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
    #     feature_pyramid = self.backbone(patches)
    #     return torch.cat([select_from_pyramid([x[j] for x in feature_pyramid], v) for j, v in enumerate(voxels)])
    



    def preprocess_feature_map(self, patch, size=16):
        # feature_pyramid = self.backbone(patch)
        feature_pyramid = self.constrastive_loss(self.backbone(patch)[:])
        for i in range(len(feature_pyramid)):
            feature_pyramid[i]=feature_pyramid[i].mean(dim=(-3, -2,-1)) #e.g. (b, 128)
            feature_pyramid[i]=feature_pyramid[i].view(feature_pyramid[i].size(0), size, -1 ).unsqueeze(1)
            feature_pyramid[i]=feature_pyramid[i].transpose(1, 2).reshape(-1, size)

        # return torch.cat(tuple(fp for fp in feature_pyramid), dim=0)
        return torch.cat(tuple(fp.unsqueeze(1) for fp in feature_pyramid), dim=1)

    def generate_feature_pyramid(self, patches: torch.Tensor) -> torch.Tensor:
        
        feature_pyramid = self.backbone(patches)
        
        # feature_map = self.local_upsample_scales(feature_pyramid)
        feature_map = feature_pyramid[0]
        # feature_map_global = self.global_upsample_scales(feature_pyramid[3:])
        feature_map_global = feature_pyramid[-1]
        return feature_map, feature_map_global
        

    
    def local_positive_crops(self, feature_map):
        crop_size = (2, 2, 2)

        
        return(get_non_overlapping_crops(feature_map, crop_size))
    


    def global_contrastive_loss(self, feature_map):
        feature_map_mean = torch.mean(feature_map, axis=1)
        return feature_map_mean.view(feature_map_mean.size(0), -1)





    def adjusted_info_nce_loss(features, temperature=0.1):
        """
        Compute the InfoNCE loss ensuring intra-image scales are treated as positive examples
        and inter-image scales as negative examples.

        Parameters:
        - features: List of tensors, each of shape [batch_size, feature_dim], for different scales.
        - temperature: Temperature scaling parameter for the softmax.

        Returns:
        - The computed InfoNCE loss.
        """
        batch_size, num_scales = features[0].shape[0], len(features)
        device = features[0].device
        
        # Concatenate features from all scales for all images
        all_features = torch.cat([F.normalize(f.view(batch_size, -1), p=2, dim=1) for f in features], dim=0)
        
        # Compute similarity matrix for all concatenated features
        sim_matrix = torch.matmul(all_features, all_features.T) / temperature

        # Construct masks for identifying positive and negative pairs
        eye = torch.eye(batch_size * num_scales, device=device)
        repeat_eye = eye.repeat(num_scales, num_scales)
        # Positive mask: block-diagonal matrix where blocks correspond to images
        pos_mask = torch.block_diag(*[torch.ones(num_scales, num_scales, device=device) for _ in range(batch_size)]) - eye
        # Negative mask: inverse of the positive mask
        neg_mask = 1 - pos_mask - eye  # Exclude self-comparisons

        # Compute log-softmax over negatives for each feature
        neg_logsumexp = torch.logsumexp(sim_matrix * neg_mask, dim=1, keepdim=True)
        
        # Select positive similarities and compute loss
        pos_similarities = sim_matrix * pos_mask
        loss = -torch.sum(F.log_softmax(sim_matrix, dim=1) * pos_mask) / (batch_size * num_scales)

        return loss


    def inter_features_loss(self, feature_pyramid_1, feature_pyramid_1_positive, feature_pyramid_2, temp):
        
        intra_similarity_1 = torch.matmul(feature_pyramid_1, feature_pyramid_1_positive.T) / temp
        inter_similarity = torch.matmul(feature_pyramid_1, feature_pyramid_1_positive.T) / temp  + torch.matmul(feature_pyramid_1_positive, feature_pyramid_2.T) / temp 
        positive_loss = (torch.logsumexp(intra_similarity_1, dim=1) ).mean()
        negative_loss = -torch.logsumexp(inter_similarity, dim=1).mean()
        loss =  0.5*(negative_loss + positive_loss )
        return negative_loss, positive_loss, loss
     
    def intra_features_loss(self, features):
        batch_size, num_scales= 4, 6
        temp = 0.1
        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.t()) / temp
        sim_matrix = sim_matrix - torch.eye(batch_size * num_scales).to(features.device) * 1e9 
        
        # Create masks for positive and negative samples
        pos_mask = torch.block_diag(*[torch.ones(num_scales, num_scales) - torch.eye(num_scales) for _ in range(batch_size)]).to(features.device)
        neg_mask = 1 - pos_mask - torch.eye(batch_size * num_scales).to(features.device)
        
        # Compute loss
        exp_sim_matrix = torch.exp(sim_matrix) * neg_mask  # Apply negative mask
        sum_exp_sim_matrix = torch.sum(exp_sim_matrix, dim=1, keepdim=True)
        
        pos_sim = torch.exp(sim_matrix) * pos_mask
        pos_sim_sum = torch.sum(pos_sim, dim=1, keepdim=True)
        
        loss = -torch.log(pos_sim_sum / sum_exp_sim_matrix)
        # loss = torch.sum(loss) / (batch_size * num_scales)
        loss = torch.sum(loss)
        
        return loss

           


    def training_step(self, batch, batch_idx):
        """Computes Info-NCE loss.
        """
        patches_1, patches_1_positive, patches_2, _, _ = batch['pretrain']

        assert self.backbone.training
        # assert self.proj_head.training
        feature_pyramid_1 = F.normalize(self.preprocess_feature_map(patches_1) , p=2, dim=1)
        feature_pyramid_2 = F.normalize(self.preprocess_feature_map(patches_2) , p=2, dim=1)
        feature_pyramid_1_positive = F.normalize(self.preprocess_feature_map(patches_1_positive), p=2, dim=1)
        feature_pyramid_1 = feature_pyramid_1.view(-1, feature_pyramid_1.size(-1))  # [batch_size * num_scales, feature_dim]
        feature_pyramid_2 = feature_pyramid_2.view(-1, feature_pyramid_2.size(-1))
        feature_pyramid_1_positive = feature_pyramid_1_positive.view(-1, feature_pyramid_2.size(-1))
        negative_loss, positive_loss, loss = self.inter_features_loss(feature_pyramid_1, feature_pyramid_1_positive, feature_pyramid_2, self.temp)

        intra_positive_loss = self.intra_features_loss(feature_pyramid_1)
        self.log(f'pretrain/info_nce_loss', loss, on_epoch=True)
        self.log(f'pretrain/positive', positive_loss, on_epoch=True)
        self.log(f'pretrain/negative', negative_loss, on_epoch=True)
        self.log(f'intra_loss', intra_positive_loss, on_epoch=True)
        # return loss
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





    def info_nce_loss(features, tau=0.07):
        batch_size, num_scales, _ = features.size()
        features = features.view(batch_size * num_scales, -1)  # Reshape for simplicity: [batch_size * num_scales, feature_dim]
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.t()) / tau
        sim_matrix = sim_matrix - torch.eye(batch_size * num_scales).to(features.device) * 1e9 
        
        # Create masks for positive and negative samples
        pos_mask = torch.block_diag(*[torch.ones(num_scales, num_scales) - torch.eye(num_scales) for _ in range(batch_size)]).to(features.device)
        neg_mask = 1 - pos_mask - torch.eye(batch_size * num_scales).to(features.device)
        
        # Compute loss
        exp_sim_matrix = torch.exp(sim_matrix) * neg_mask  # Apply negative mask
        sum_exp_sim_matrix = torch.sum(exp_sim_matrix, dim=1, keepdim=True)
        
        pos_sim = torch.exp(sim_matrix) * pos_mask
        pos_sim_sum = torch.sum(pos_sim, dim=1, keepdim=True)
        
        loss = -torch.log(pos_sim_sum / sum_exp_sim_matrix)
        loss = torch.sum(loss) / (batch_size * num_scales)
        
        return loss

