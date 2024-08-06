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
# from vox2vec.pretrain.my_transormations import sample_view
# from vox2vec.pretrain.my_transormations import rot_rand
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
        # self.backbone.right_blocks[4].layers[1].layers[5].weight = torch.nn.Parameter(torch.zeros_like(self.backbone.right_blocks[4].layers[1].layers[5].weight))
        # self.backbone.right_blocks[4].layers[1].layers[5].bias = torch.nn.Parameter(torch.zeros_like(self.backbone.right_blocks[4].layers[1].layers[5].bias))
        # embed_dim = sum_pyramid_channels(base_channels, num_scales)
        # embed_dim = 128
        embed_dim = 16
        local_emb_dim = 8192
        # local_emb_dim = 1024
        proj_dim_global = 16
        proj_dim_local = 1024
        # proj_dim_local = 8192
        # embed_dim= 524288
        
        # self.proj_head = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.BatchNorm1d(embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim, proj_dim_global),

        #     Lambda(F.normalize)
        # )

        self.local_proj_head = nn.Sequential(
            # nn.Linear(local_emb_dim, 1024),
            nn.Linear(local_emb_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, proj_dim_local),
            # nn.BatchNorm1d(proj_dim),


            Lambda(F.normalize)
        )
        
        # self.local_upsample_scales = Upsample_scales(16, 6, 16)
        
        # self.global_upsample_scales = Upsample_scales(128, 3, 1)


        self.temp = temp
        self.lr = lr

    # def _vox_to_vec(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
    #     feature_pyramid = self.backbone(patches)
    #     return torch.cat([select_from_pyramid([x[j] for x in feature_pyramid], v) for j, v in enumerate(voxels)])
    




    def generate_feature_pyramid(self, patches: torch.Tensor) -> torch.Tensor:
        
        feature_pyramid = self.backbone(patches)
        
        # feature_map = self.local_upsample_scales(feature_pyramid)
        feature_map = feature_pyramid[0]
        # feature_map_global = self.global_upsample_scales(feature_pyramid[3:])
        feature_map_global = feature_pyramid[-1]
        return feature_map, feature_map_global
        

    
    def local_positive_crops(self, feature_map):
        # crop_size = (1, 1, 16, 16, 16)
        crop_size = (1, 1, 32, 32, 8)
        # crop_size = (1, 1, 8, 8, 8)
        
        return(get_non_overlapping_crops(feature_map, crop_size))
    


    def global_contrastive_loss(self, feature_map):
        feature_map_mean = torch.mean(feature_map, axis=1)
        return feature_map_mean.view(feature_map_mean.size(0), -1)

    


    def training_step(self, batch, batch_idx):
        """Computes Info-NCE loss.
        """
        patches_1, patches_1_positive, patches_2, _, _ = batch['pretrain']

        assert self.backbone.training
        # assert self.proj_head.training


        feature_map_1, feature_map_1_mean = self.generate_feature_pyramid(patches_1)
        _, feature_map_2_mean = self.generate_feature_pyramid(patches_2)
        feature_map_1_positive, _ = self.generate_feature_pyramid(patches_1_positive)
        original_crops = self.local_positive_crops(feature_map_1)
        original_crops_positive = self.local_positive_crops(feature_map_1_positive)


        '''loss without the projector
        # Flatten the last three dimensions for processing
        tensor1_flat_local =  F.normalize(original_crops.view(original_crops.size(0),-1), p=2, dim=1)  # Shape: [1024, 32*32*8]
        tensor2_flat_local = F.normalize(original_crops_positive.view(original_crops_positive.size(0), -1), p=2, dim=1)  # Shape: [1024, 32*32*8]
        '''
        selected_indices = np.random.choice(original_crops.size(0), 100, replace=False)

# Use the selected indices to create a new tensor with shape (100, 3000)
        original_crops = original_crops[selected_indices]
        original_crops_positive = original_crops_positive[selected_indices]
        
        tensor1_flat_local = self.local_proj_head(original_crops.view(original_crops.size(0), -1))
        tensor2_flat_local = self.local_proj_head(original_crops_positive.view(original_crops.size(0), -1))
        logits_11 = torch.matmul(tensor1_flat_local, tensor1_flat_local.T) / self.temp
        logits_11.fill_diagonal_(float('-inf'))
        logits_12 = torch.matmul(tensor1_flat_local, tensor2_flat_local.T) / self.temp
        logits_22 = torch.matmul(tensor2_flat_local, tensor2_flat_local.T) / self.temp
        logits_22.fill_diagonal_(float('-inf'))
        loss_1 = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_11, logits_12], dim=1), dim=1))
        loss_2 = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_12.T, logits_22], dim=1), dim=1))
        info_nce_loss_local = (loss_1 + loss_2) / 2
        # print('info_nce_loss_local', info_nce_loss_local.item())

        
        
        
        
        # Normalize the vectors (important for InfoNCE)
        # tensor1_flat_local = F.normalize(tensor1_flat_local, p=2, dim=1)
        # tensor2_flat_local = F.normalize(tensor2_flat_local, p=2, dim=1)
        # similarity_matrix = torch.matmul(tensor1_flat_local, tensor2_flat_local.T) / temperature

        # # Create labels for the positive pairs
        # labels = torch.arange(original_crops.size(0)).to(original_crops.device)
        # # Calculate the InfoNCE loss
        # info_nce_loss_local = F.cross_entropy(similarity_matrix, labels)
        '''
        global loss
        '''
        # embeds_1 = self.proj_head(self.global_contrastive_loss(feature_map_1_mean))
        # embeds_2 = self.proj_head(self.global_contrastive_loss(feature_map_2_mean))



        # tensor1_flat_global = embeds_1.view(embeds_1.size(0),-1)  # Shape: [1024, 32*32*8]
        # tensor2_flat_global = embeds_2.view(embeds_2.size(0),-1)  # Shape: [1024, 32*32*8]
        # logits_11 = torch.matmul(tensor1_flat_global, tensor1_flat_global.T) / self.temp
        # logits_11.fill_diagonal_(float('-inf'))
        # logits_12 = torch.matmul(tensor1_flat_global, tensor2_flat_global.T) / self.temp
        # logits_22 = torch.matmul(tensor2_flat_global, tensor2_flat_global.T) / self.temp
        # logits_22.fill_diagonal_(float('-inf'))
        # loss_1 = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_11, logits_12], dim=1), dim=1))
        # loss_2 = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_12.T, logits_22], dim=1), dim=1))
        # info_nce_loss_global = (loss_1 + loss_2) / 2
        
        # print('info_nce_loss_global', info_nce_loss_global.item())
        '''
        old implementation of the contrastive loss 
        '''
        # tensor1_flat_global = F.normalize(tensor1_flat_global, p=2, dim=1)
        # tensor2_flat_global = F.normalize(tensor2_flat_global, p=2, dim=1)

        # similarity_matrix = torch.matmul(tensor1_flat_global, tensor2_flat_global.T) / temperature
        # labels = torch.arange(embeds_1.size(0)).to(embeds_1.device)
        # info_nce_loss_global = F.cross_entropy(similarity_matrix, labels)
        
        
        # loss = (info_nce_loss_local + info_nce_loss_global) / 2
        # print('losses', info_nce_loss_local.item(), info_nce_loss_global.item())

        # self.log(f'pretrain/info_nce_loss', loss, on_epoch=True)
        self.log(f'pretrain/info_nce_loss_local', info_nce_loss_local, on_epoch=True)
        # self.log(f'pretrain/info_nce_loss_global', info_nce_loss_global, on_epoch=True)
        # return loss
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
