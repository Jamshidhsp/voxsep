from typing import *
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import logging
logging.getLogger().setLevel(logging.WARNING)

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from vox2vec.nn import Lambda
from vox2vec.nn.functional import select_from_pyramid, sum_pyramid_channels
import numpy as np
from copy import deepcopy

from torch.utils.tensorboard import SummaryWriter

tb = SummaryWriter()

proj_dim = 128

import torch.nn as nn

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

class AttentionContextualization(nn.Module):
    def __init__(self, feature_dim, num_heads=4, lambda_param=0.2):
        super(AttentionContextualization, self).__init__()
        self.lambda_param = lambda_param

        # self.memory_bank = torch.randn(memory_bank_size, feature_dim)  # Memory bank
        # self.memory_bank_size = memory_bank_size
        # self.memory_bank.requires_grad = False  # Memory bank is not trained directly
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)
        self.memory_ptr = 0

    def forward(self, queries, memory_bank):
        # Normalize queries for attention mechanism
        queries = F.normalize(queries, dim=-1)

        # Use memory bank as keys and values for cross-attention
        keys = memory_bank.clone().detach().to(queries.device)
        values = memory_bank.clone().detach().to(queries.device)

        # Queries: current image features
        # Keys/Values: memory bank features
        contextualized_features, _ = self.attention(queries.unsqueeze(1), keys.unsqueeze(1), values.unsqueeze(1))
        contextualized_features = contextualized_features.squeeze(1)

        # Combine the original features with the contextualized features
        contextualized_features = self.lambda_param * contextualized_features + (1 - self.lambda_param) * queries

        return contextualized_features

    @torch.no_grad()
    def update_memory_bank(self, new_features):
        batch_size = new_features.size(0)
        end_ptr = self.memory_ptr + batch_size

        if end_ptr >= self.memory_bank_size:
            overflow = end_ptr - self.memory_bank_size
            self.memory_bank[self.memory_ptr:self.memory_bank_size] = new_features[:batch_size - overflow]
            self.memory_bank[0:overflow] = new_features[batch_size - overflow:]
            self.memory_ptr = overflow
        else:
            self.memory_bank[self.memory_ptr:end_ptr] = new_features
            self.memory_ptr = end_ptr

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
        x = self.linear1(x.view(x.shape[0], -1))
        x = self.relu(x)
        x = self.linear2(x)
        return x

def orthogonality(embedding):
    dot_product = torch.mm(embedding, embedding.T)
    dot_product.fill_diagonal_(0)
    loss = torch.sum((dot_product)**2)
    return loss/(embedding.size(0)*(embedding.size(0)-1))

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

    def size(self):
        return min(self.max_size, self.ptr)
    
    @torch.no_grad()
    def shuffle(self):
        max_index = self.max_size if self.ptr == 0 or self.ptr == self.max_size else self.ptr
        shuffle_indices = torch.randperm(max_index)
        self.queue[:max_index] = self.queue[shuffle_indices]

class Vox2Vec(pl.LightningModule):
    def __init__(
            self,
            backbone: nn.Module,
            base_channels: int,
            num_scales: int,
            proj_dim: int = proj_dim,
            temp: float = 0.1,
            lr: float = 3e-4,
    ):

        super().__init__()

        self.save_hyperparameters(ignore='backbone')
        self.epoch = 0
        self.backbone = backbone
        embed_dim = sum_pyramid_channels(base_channels, num_scales)
        self.pe_size = embed_dim

        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )

        self.proj_head_key = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )

        self.backbone_key = deepcopy(self.backbone)
        for param_k in self.backbone_key.parameters():
            param_k.requires_grad = False  

        self.pe = VoxelPositionalEncoding(dim=self.pe_size)
        self.temperature = 0.1
        self.lr = lr
        self.queue = Queue(max_size=9500, embedding_size=proj_dim)
        self.global_projector = Global_projector()

        # Add AttentionContextualization
        self.contextualization = AttentionContextualization(feature_dim=proj_dim)

    def _vox_to_vec(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
        feature_pyramid = self.backbone(patches)[:]
        return torch.cat([select_from_pyramid([x[j] for x in feature_pyramid], v) for j, v in enumerate(voxels)])
    
    def _vox_to_vec_key(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
        feature_pyramid = self.backbone_key(patches)[:]
        return torch.cat([select_from_pyramid([x[j] for x in feature_pyramid], v) for j, v in enumerate(voxels)])
    
    @torch.no_grad()
    def momentum_update(self):
        momentum = 0.999
        with torch.no_grad():
            for param_q, param_k in zip(self.backbone.parameters(), self.backbone_key.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
            for param_q, param_k in zip(self.proj_head.parameters(), self.proj_head_key.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        
    def training_step(self, batch, batch_idx):
        patches_1, patches_1_positive, anchor_voxel_1, _, _ = batch['pretrain']
        positive_voxels = anchor_voxel_1
        patches_1_negative = patches_1.clone()  
        
        max_shuffle_size = int(0.5 * patches_1.size(-1))
        for i in range(patches_1.size(0)):
            for _ in range(np.random.randint(1, 10)):
                size = np.random.randint(4, max_shuffle_size)
                src_h, src_w, src_d = np.random.randint(0, patches_1.size(-1) - size, 3)
                des_h, des_w, des_d = np.random.randint(0, patches_1.size(-1) - size, 3)
                patches_1_negative[i, 0, src_h:src_h+size, src_w:src_w+size, src_d:src_d+size] = patches_1_negative[i, 0, des_h:des_h+size, des_w:des_w+size, des_d:des_d+size]
        
        assert self.backbone.training
        self.backbone_key.training = False
        assert not self.backbone_key.training
        bs = positive_voxels.size(0)

        self.momentum_update()

        # Generate embeddings
        embeds_anchor = self.proj_head(self._vox_to_vec(patches_1, anchor_voxel_1))
        # with torch.no_grad():
        embeds_positive = self.proj_head(self._vox_to_vec_key(patches_1_positive, positive_voxels))

        # Contextualize embeddings using attention
        embeds_anchor_contextualized = self.contextualization(embeds_anchor, self.queue.get())
        embeds_positive_contextualized = self.contextualization(embeds_positive,  self.queue.get())

        # Update the memory bank with new positive embeddings
        # self.contextualization.update_memory_bank(embeds_positive_contextualized.view(-1, proj_dim))

        embeds_key = self.queue.get().to(embeds_anchor.device)

        # Normalize embeddings before computing logits
        embeds_anchor_contextualized = F.normalize(embeds_anchor_contextualized, p=2, dim=1)
        embeds_positive_contextualized = F.normalize(embeds_positive_contextualized, p=2, dim=1)

        logits_12 = torch.matmul(embeds_anchor_contextualized, embeds_positive_contextualized.T) / self.temperature
        logits_22 = torch.matmul(embeds_anchor_contextualized, embeds_key.T) / self.temperature

        loss_1 = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_12, logits_22], dim=1), dim=1))

        running_loss = loss_1   

        # Log metrics
        queue_mean = self.queue.get().mean()
        queue_std = self.queue.get().std()
        self.log('pretrain/info_nce_loss_local', loss_1, on_epoch=True)
        self.log('queue_mean', queue_mean, on_epoch=True)
        self.log('queue_std', queue_std, on_epoch=True)

        global_step = str(self.epoch)
        metadata = ['anchor'] * embeds_positive_contextualized.view(-1, 128).size(0) + ['positive'] * embeds_positive_contextualized.view(-1, 128).size(0) + ['negative'] * embeds_key.view(-1, 128).size(0)
        all_embeddings = torch.cat((embeds_anchor_contextualized.view(-1, embeds_positive_contextualized.size(-1)), embeds_positive_contextualized.view(-1, embeds_positive_contextualized.size(-1)), embeds_key.view(-1, embeds_key.size(-1))), dim=0)
        
        if batch_idx + 1 == 100:
            tb.add_embedding(all_embeddings, metadata=metadata, label_img=None, global_step=global_step, tag='all_embedding')
            self.epoch += 1

        # Update queue
        if batch_idx % 1 == 0:
            self.queue.update(embeds_positive_contextualized.view(-1, proj_dim)[:int(1.0 * (embeds_positive_contextualized.view(-1, proj_dim)).size(0))])

        return running_loss

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not self.trainer.training:
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
