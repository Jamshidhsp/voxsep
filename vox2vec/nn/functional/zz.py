from typing import *
import os
import logging
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from copy import deepcopy
from vox2vec.nn import Lambda

proj_dim = 128
embed_dim = 1008

class VoxelPositionalEncoding(nn.Module):
    def __init__(self, dim):
        super(VoxelPositionalEncoding, self).__init__()
        self.linear1 = nn.Linear(3, proj_dim)
        self.dim = dim
    
    def forward(self, anchor_voxels, target_voxels):
        anchor_voxels = anchor_voxels.expand(-1, target_voxels.size(1), -1)
        voxels = anchor_voxels - target_voxels
        voxels = voxels / 128
        pe = self.linear1(voxels)
        return pe

class Global_projector(nn.Module):
    def __init__(self):
        super(Global_projector, self).__init__()
        self.conv = nn.Conv3d(512, 1, 1, 1)
        self.linear1 = nn.Linear(16, 8)
        self.linear2 = nn.Linear(8, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.linear1(x.view(x.shape[0], -1))
        x = self.relu(x)
        x = self.linear2(x)
        return x
    
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Queue:
    def __init__(self, max_size, embedding_size):
        self.max_size = max_size
        self.embedding_size = embedding_size
        self.queue = torch.randn(max_size, embedding_size)
        self.ptr = 0  

    @torch.no_grad()
    def enqueue(self, item):
        self.queue[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.max_size 

    @torch.no_grad()
    def update(self, new_items):
        if len(new_items) >= self.max_size:
            self.queue = new_items[-self.max_size:]
            self.ptr = 0
        else:
            remaining_space = self.max_size - self.ptr
            if len(new_items) <= remaining_space:
                self.queue[self.ptr:self.ptr+len(new_items)] = new_items
                self.ptr += len(new_items)
            else:
                self.queue[self.ptr:] = new_items[:remaining_space]
                self.queue[:len(new_items)-remaining_space] = new_items[remaining_space:]
                self.ptr = len(new_items) - remaining_space

    @torch.no_grad()
    def get(self):
        self.shuffle()
        q = F.normalize(self.queue.clone(), p=2, dim=1)[:]
        return q

    @torch.no_grad()
    def shuffle(self):
        max_index = self.max_size if self.ptr == 0 or self.ptr == self.max_size else self.ptr
        shuffle_indices = torch.randperm(max_index)
        self.queue[:max_index] = self.queue[shuffle_indices]

# ScoringMLP to learn difficulty of negatives based on feature embeddings
class ScoringMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(ScoringMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output score between 0 and 1
        )
    
    def forward(self, x):
        x = self.model(x)
        return x

class Vox2Vec(pl.LightningModule):
    def __init__(
            self,
            backbone: nn.Module,
            base_channels: int,
            num_scales: int,
            proj_dim: int = proj_dim,
            temp: float = 0.2,
            lr: float = 3e-4,
            output_size=(1, 1, 1),
            hidden_dim: int = 64  # Dimension for scoring MLP
    ):
        super().__init__()
        self.backbone = backbone
        self.proj_dim = proj_dim
        self.temperature = temp
        self.lr = lr
        self.num_scales = num_scales
        self.output_size = output_size
        self.num_negatives = 5

        self.adaptive_pooling_layers = nn.ModuleList(
            [nn.AdaptiveAvgPool3d(output_size) for _ in range(num_scales)]
        )
        
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

        # Scoring MLP to learn negative difficulty
        self.scoring_mlp = ScoringMLP(proj_dim, hidden_dim)
        self.scoring_mlp.apply(initialize_weights)
    
    def extract_features_across_scales(self, feature_pyramid):
        pooled_features = []
        for scale_idx, feature_map in enumerate(feature_pyramid):
            pooled_feature = self.adaptive_pooling_layers[scale_idx](feature_map)
            pooled_features.append(pooled_feature)

        concatenated_features = torch.cat(pooled_features, dim=1)
        return concatenated_features
    
    def _vox_to_vec(self, patches: torch.Tensor) -> torch.Tensor:
        feature_pyramid = self.backbone(patches)
        concatenated_features = self.extract_features_across_scales(feature_pyramid)
        flattened_features = concatenated_features.view(concatenated_features.size(0), -1)
        return flattened_features

    def generate_negatives(self, patches, num_negatives=3):
        negatives = []

        # Loop through each patch in the batch
        for patch_idx in range(patches.size(0)):  # patches.size(0) is the batch size
            negative_patch = (patches[patch_idx])  # Clone the current patch for each patch in the batch
            
            for i in range(num_negatives):
                # Swap regions within the negative patch
                num_swaps = np.random.randint(0, 100)
                if num_swaps > 0:
                    max_swap_size = int(0.8 * patches.size(-1))
                    for _ in range(num_swaps):
                        size = np.random.randint(4, max_swap_size)
                        src_h, src_w, src_d = np.random.randint(0, patches.size(-1) - size, 3)
                        des_h, des_w, des_d = np.random.randint(0, patches.size(-1) - size, 3)
                        negative_patch[:, src_h:src_h+size, src_w:src_w+size, src_d:src_d+size] = \
                            negative_patch[:, des_h:des_h+size, des_w:des_w+size, des_d:des_d+size]

                negatives.append(negative_patch)  # Append the generated negative patch

        negatives = torch.stack(negatives)  # Stack the negatives into a single tensor

        return negatives

    def training_step(self, batch, batch_idx):
        patches_1, _, _, _, _ = batch['pretrain']
        
        # Generate negative patches and compute embeddings
        num_negatives = 4
        assert self.proj_head.training
        negative_patches = self.generate_negatives(patches_1, num_negatives)
        global_anchor = self.proj_head(self._vox_to_vec(patches_1))  # Shape: [B, proj_dim]
        global_negatives = self.proj_head(self._vox_to_vec(negative_patches)).view(-1, num_negatives, self.proj_dim)
        print(self.proj_head[0].weight.grad_fn)
        print('global_anchor.grad_fn', global_anchor.grad_fn)
        print('global_negative', global_negatives.grad_fn)
        # Compute similarity logits between anchor and negatives
        print(global_anchor.shape, global_negatives.shape)
        # logits = torch.einsum('nc,nkc->nk', global_anchor, global_negatives)
        # logits = torch.bmm(global_negatives, global_anchor.unsqueeze(-1)).squeeze(-1)
        logits =  (torch.zeros_like(global_anchor)).to(global_anchor.device)*self.temperature
        print(logits.grad)
        print(logits.requires_grad)
        # logits/=0.1
        # softmax_logits = F.softmax(logits, dim=1)

        scores = self.scoring_mlp(global_negatives.view(-1, self.proj_dim)).view(-1, num_negatives)
        softmax_scores = F.softmax(scores.float(), dim=-1)
        # softmax_scores = F.softmax(scores, dim=-1)
        # print(f'softmax_logist ---- {softmax_logits} ---- softmax_logits.log()----{softmax_logits.log()}')
        # print(f'softmax_scors ---- {softmax_scores}')
        print(self.scoring_mlp.model[0].weight.grad_fn)
        
        # Compute the loss (KL-Divergence + MSE)
        # global_loss = F.binary_cross_entropy(softmax_logits, softmax_scores)#+F.kl_div(softmax_logits.log(), softmax_scores, reduction='batchmean') + F.mse_loss(softmax_logits, softmax_scores)
        global_loss= F.mse_loss(torch.ones(logits.shape).to(logits.device), logits)
        return global_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def validation_step(self, batch, batch_idx):
        pass

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not self.trainer.training:
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
