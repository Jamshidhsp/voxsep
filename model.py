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
            nn.Conv3d(base_channels * 2 ** i, base_channels, kernel_size=1, bias=(i == 0))
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

        self.constrastive_loss = Contrastive_scale(16, 6, 128)
        self.temp = temp
        self.lr = lr

    # def _vox_to_vec(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
    #     feature_pyramid = self.backbone(patches)
    #     return torch.cat([select_from_pyramid([x[j] for x in feature_pyramid], v) for j, v in enumerate(voxels)])
    



    def preprocess_feature_map(self, patch, size=16):
        # feature_pyramid = self.backbone(patch)
        feature_pyramid = self.constrastive_loss(self.backbone(patch))
        for i in range(len(feature_pyramid)):
            feature_pyramid[i]=feature_pyramid[i].mean(dim=(-3, -2,-1)) #e.g. (2, 64)
            feature_pyramid[i]=feature_pyramid[i].view(feature_pyramid[i].size(0), 16, -1 )
            feature_pyramid[i]=feature_pyramid[i].transpose(1, 2).reshape(-1, 16)

        return torch.cat(tuple(fp for fp in feature_pyramid), dim=0)

    def generate_feature_pyramid(self, patches: torch.Tensor) -> torch.Tensor:
        
        feature_pyramid = self.backbone(patches)
        
        # feature_map = self.local_upsample_scales(feature_pyramid)
        feature_map = feature_pyramid[0]
        # feature_map_global = self.global_upsample_scales(feature_pyramid[3:])
        feature_map_global = feature_pyramid[-1]
        return feature_map, feature_map_global
        

    
    def local_positive_crops(self, feature_map):
        # crop_size = (1, 1, 16, 16, 16)
        # crop_size = (32, 32, 8)
        crop_size = (2, 2, 2)
        # crop_size = (128, 128, 32)
        # crop_size = (1, 1, 8, 8, 8)
        
        return(get_non_overlapping_crops(feature_map, crop_size))
    


    def global_contrastive_loss(self, feature_map):
        feature_map_mean = torch.mean(feature_map, axis=1)
        return feature_map_mean.view(feature_map_mean.size(0), -1)

    
    # def infoNCE_loss(x1, x2, temperature=0.1):

    #     sim_matrix = torch.matmul(x1, x2.T) / temperature
        
    #     # Positive samples are on the diagonal of the sim_matrix
    #     positives = torch.diag(sim_matrix)
        
    #     # For each positive, calculate the log-sum-exp of all entries (including negatives)
    #     negatives = torch.logsumexp(sim_matrix, dim=1)
        
    #     # InfoNCE loss
    #     loss = -torch.mean(positives - negatives)
    
    #     return loss


    def training_step(self, batch, batch_idx):
        """Computes Info-NCE loss.
        """
        patches_1, patches_1_positive, patches_2, _, _ = batch['pretrain']

        assert self.backbone.training
        # assert self.proj_head.training
        # feature_pyramid_1 = F.normalize(self.preprocess_feature_map(patches_1), p=2, dim=1)
        # feature_pyramid_2 = F.normalize(self.preprocess_feature_map(patches_2), p=2, dim=1)
        feature_pyramid_1 = (self.preprocess_feature_map(patches_1))
        feature_pyramid_1_positive = (self.preprocess_feature_map(patches_1_positive))
        feature_pyramid_2 = (self.preprocess_feature_map(patches_2))
        feature_pyramid_1 = self.proj_head(feature_pyramid_1)
        feature_pyramid_2 = self.proj_head(feature_pyramid_2)
        feature_pyramid_1_positive = self.proj_head(feature_pyramid_1_positive)

    #     positive_similarities = torch.matmul(feature_pyramid_1, feature_pyramid_1.T) / 0.1
    #     negative_similarities = torch.matmul(feature_pyramid_1, feature_pyramid_2.T) / 0.1

    #     mask = torch.eye(feature_pyramid_1.shape[0], dtype=torch.bool).to('cuda')
    #     positive_similarities.masked_fill_(mask, float('-inf')).to('cuda')
    #     negative_similarities.masked_fill_(mask, float('-inf')).to('cuda')
    
    # # Log-sum-exp across negatives for each anchor
    #     negatives_logsumexp = torch.logsumexp(negative_similarities, dim=1)
    
    # # For positives, since each vector in x1 is positive with each other, we take the log-sum-exp excluding self
    #     positives_logsumexp = torch.logsumexp(positive_similarities, dim=1)
    #     loss = torch.mean(negatives_logsumexp - positives_logsumexp)

        # logits_11 = torch.matmul(feature_pyramid_1, feature_pyramid_1.T) / self.temp
        # logits_11.fill_diagonal_(float('-inf'))
        # logits_12 = torch.matmul(feature_pyramid_1, feature_pyramid_2.T) / self.temp
        # logits_22 = torch.matmul(feature_pyramid_2, feature_pyramid_2.T) / self.temp
        # logits_22.fill_diagonal_(float('-inf'))
        # logits_13 = torch.matmul(feature_pyramid_1, feature_pyramid_1_positive.T) / self.temp
        # logits_12.fill_diagonal_(float('-inf'))
        # logits_32 = torch.matmul(feature_pyramid_1_positive, feature_pyramid_2.T) / self.temp
        # loss_1 = torch.mean(-logits_13.diag() + torch.logsumexp(torch.cat([logits_11, logits_12], dim=1), dim=1))
        # loss_2 = torch.mean(-logits_13.diag() + torch.logsumexp(torch.cat([logits_12.T, logits_22], dim=1), dim=1))

        # loss_1 = torch.mean(-logits_12 + torch.logsumexp(torch.cat([logits_11, logits_13], dim=1), dim=1))
        # loss_2 = torch.mean(-logits_32 -torch.logsumexp(torch.cat([logits_12.T, logits_32], dim=1), dim=1))
        #  loss_2 = torch.mean(torch.logsumexp(torch.cat([logits_12.T, logits_32], dim=1), dim=1))
        # loss_1 = torch.mean(torch.logsumexp(torch.cat([logits_32, logits_12], dim=1), dim=1))
        # loss_2 = torch.mean(torch.logsumexp(torch.cat([logits_11, logits_13, logits_22], dim=1), dim=1))
        # loss = (-loss_1+loss_2)
        # print('info_nce_loss_local', loss.item())
        '''new
        
        '''



    # Flatten feature pyramids if they're not already (assuming they are 3D tensors [batch, scales, features])
        # fp1_norm = feature_pyramid_1.view(feature_pyramid_1.size(0) * feature_pyramid_1.size(1), -1)
        # fp2_norm = feature_pyramid_2.view(feature_pyramid_2.size(0) * feature_pyramid_2.size(1), -1)
        
        
        # # Compute similarity matrices
        # intra_similarity_1 = torch.matmul(fp1_norm, fp1_norm.T) / self.temp
        # intra_similarity_2 = torch.matmul(fp2_norm, fp2_norm.T) / self.temp
        # inter_similarity = torch.matmul(fp1_norm, fp2_norm.T) / self.temp
        
        # # Enhance intra-pyramid similarity (positive within pyramids)
        # intra_loss_1 = -torch.logsumexp(intra_similarity_1, dim=1).mean()
        # intra_loss_2 = -torch.logsumexp(intra_similarity_2, dim=1).mean()
        
        # # Penalize inter-pyramid similarity (negative across pyramids)
        # inter_loss = torch.logsumexp(inter_similarity, dim=1).mean()
        
        # # Combine losses
        # total_loss = intra_loss_1 + intra_loss_2 + inter_loss
        
        # return total_loss








    
    # Assuming feature_pyramid_1 and feature_pyramid_2 are tensors of shape [batch_size, num_scales, feature_dim]
    # Flatten feature pyramids
        fp1_flat = feature_pyramid_1.view(-1, feature_pyramid_1.size(-1))  # [batch_size * num_scales, feature_dim]
        fp2_flat = feature_pyramid_2.view(-1, feature_pyramid_2.size(-1))
        
        # Normalize features
   
        
        # Compute similarity matrices
        intra_similarity_1 = torch.matmul(fp1_flat, fp1_flat.T) / self.temp
        intra_similarity_2 = torch.matmul(fp2_flat, fp2_flat.T) / self.temp
        inter_similarity = torch.matmul(fp1_flat, fp1_flat.T) / self.temp    
        # Masking self-similarity for intra-pyramid
        eye_mask = torch.eye(intra_similarity_1.size(0), device=feature_pyramid_1.device).bool()
        intra_similarity_1.masked_fill_(eye_mask, float('-inf'))
        intra_similarity_2.masked_fill_(eye_mask, float('-inf'))
        
        # Positive pair loss (intra-pyramid): Maximize similarity (minimize negative log)
        positive_loss = - (torch.logsumexp(intra_similarity_1, dim=1) + torch.logsumexp(intra_similarity_2, dim=1)).mean()
        
        # Negative pair loss (inter-pyramid): Penalize high similarity
        # Here, we aim to minimize similarity across pyramids, which doesn't directly translate to using logsumexp
        # Instead, we should ensure that this component effectively increases the loss as similarity increases
        negative_loss = torch.logsumexp(inter_similarity, dim=1).mean()
        
        # Combine losses: Since we're minimizing, adding negative_loss contributes to penalizing high inter-pyramid similarity
        total_loss = positive_loss + 5*negative_loss
        
        return total_loss













        # feature_map_1, feature_map_1_mean = self.generate_feature_pyramid(patches_1)
        # _, feature_map_2_mean = self.generate_feature_pyramid(patches_2)
        # feature_map_1_positive, _ = self.generate_feature_pyramid(patches_1_positive)
        # original_crops = self.local_positive_crops(feature_map_1)
        # original_crops_positive = self.local_positive_crops(feature_map_1_positive)


        '''loss without the projector
        # Flatten the last three dimensions for processing
        tensor1_flat_local =  F.normalize(original_crops.view(original_crops.size(0),-1), p=2, dim=1)  # Shape: [1024, 32*32*8]
        tensor2_flat_local = F.normalize(original_crops_positive.view(original_crops_positive.size(0), -1), p=2, dim=1)  # Shape: [1024, 32*32*8]
        '''
        # number_crops = int(0.1*original_crops.size(0))
        
        # selected_indices = np.random.choice(original_crops.size(0), number_crops, replace=False)

# Use the selected indices to create a new tensor with shape (100, 3000)
        # original_crops = original_crops[selected_indices]
        # original_crops_positive = original_crops_positive[selected_indices]
        
        # tensor1_flat_local = self.local_proj_head(original_crops.view(original_crops.size(0), -1))
        # tensor2_flat_local = self.local_proj_head(original_crops_positive.view(original_crops.size(0), -1))
        # logits_11 = torch.matmul(tensor1_flat_local, tensor1_flat_local.T) / self.temp
        # logits_11.fill_diagonal_(float('-inf'))
        # logits_12 = torch.matmul(tenzal.T) / self.temp
        # logits_22 = torch.matmul(tensor2_flat_local, tensor2_flat_local.T) / self.temp
        # logits_22.fill_diagonal_(float('-inf'))
        # loss_1 = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_11, logits_12], dim=1), dim=1))
        # loss_2 = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_12.T, logits_22], dim=1), dim=1))
        # info_nce_loss_local = (loss_1 + loss_2) / 2
        # print('info_nce_loss_local', info_nce_loss_local.item())

        



#         x1_normalized = F.normalize(feature_pyramid_1, p=2, dim=1)
#         x2_normalized = F.normalize(feature_pyramid_2, p=2, dim=1)

# # Compute the loss
        # loss = self.infoNCE_loss(x1_normalized, x2_normalized)
        # print(f"InfoNCE Loss: {loss.item()}")


        
        
        
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

        self.log(f'pretrain/info_nce_loss', loss, on_epoch=True)
        # self.log(f'pretrain/info_nce_loss_local', info_nce_loss_local, on_epoch=True)
        # self.log(f'pretrain/info_nce_loss_global', info_nce_loss_global, on_epoch=True)
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
