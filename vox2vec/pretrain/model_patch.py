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
import random
from torch.utils.tensorboard import SummaryWriter
import time

tb = SummaryWriter()
proj_dim = 128

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
        self.lambda_param = torch.nn.Parameter(torch.tensor(lambda_param))
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)

    def forward(self, queries, keys, values):
        queries = F.normalize(queries, dim=-1).unsqueeze(0)  # Add sequence length dimension
        keys = F.normalize(keys, dim=-1).unsqueeze(0)        # Add sequence length dimension
        values = F.normalize(values, dim=-1).unsqueeze(0)    # Add sequence length dimension
        contextualized_features, attention_weights = self.attention(queries, keys, values)
        contextualized_features = contextualized_features.squeeze(0)
        contextualized_features = self.lambda_param * contextualized_features + (1 - self.lambda_param) * queries.squeeze(0)
        return contextualized_features, attention_weights

class Global_projector(nn.Module):
    def __init__(self):
        super(Global_projector, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.linear1 = nn.Linear(1008, 512)
        self.linear2 = nn.Linear(512, 128)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.linear1(x.view(x.shape[0], -1))
        x = self.relu(x)
        x = self.linear2(x)
        return x

class Queue:
    def __init__(self, max_size, embedding_size):
        self.max_size = max_size
        self.embedding_size = embedding_size
        self.queue = torch.randn(max_size, embedding_size)
        self.ptr = 0  
        self.reset_fraction = 0.4

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
        q = self.queue.clone()
        return q

    def size(self):
        return min(self.max_size, self.ptr)
    
    @torch.no_grad()
    def shuffle(self):
        max_index = self.max_size if self.ptr == 0 or self.ptr == self.max_size else self.ptr
        shuffle_indices = torch.randperm(max_index)
        self.queue[:max_index] = self.queue[shuffle_indices]

    @torch.no_grad()
    def reset_queue(self, reset_fraction):
        num_reset = int(self.max_size * reset_fraction)
        self.queue[:num_reset] = torch.randn(num_reset, self.embedding_size).to(self.queue.device)

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
        self.backbone = backbone
        embed_dim = 128

        self.backbone_key = deepcopy(self.backbone)
        self.global_projector = Global_projector()
        self.global_projector_key = deepcopy(self.global_projector)
        
        for param_k in self.backbone_key.parameters():
            param_k.requires_grad = False  

        self.temperature = 0.1
        self.lr = lr
        self.queue = Queue(max_size=6000, embedding_size=embed_dim)
        self.warmup_epochs = -1
        self.contextualization = AttentionContextualization(feature_dim=embed_dim)

    @torch.no_grad()
    def momentum_update(self):
        momentum = 0.99
        with torch.no_grad():
            for param_q, param_k in zip(self.backbone.parameters(), self.backbone_key.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
            for param_q, param_k in zip(self.global_projector.parameters(), self.global_projector_key.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)

    def align_and_concatenate_features(self, feature_maps: List[torch.Tensor], target_size=(4, 4, 4)) -> torch.Tensor:

        pooled_features = [
            F.adaptive_avg_pool3d(fm, output_size=target_size) for fm in feature_maps
        ]
        
        concatenated_features = torch.cat(pooled_features, dim=1)  # Concatenate along channel dimension
        
        return concatenated_features


    def training_step(self, batch, batch_idx):
        patches_1, patches_1_positive, _, _, _ = batch['pretrain']

        assert self.backbone.training
        self.backbone_key.training = False
        self.global_projector_key.training = False
        assert not self.backbone_key.training

        self.momentum_update()

        # Extract features from all FPN scales for both anchor and positive samples
        features_anchor_scales = self.backbone(patches_1)
        features_positive_scales = self.backbone(patches_1_positive)

        # Use adaptive pooling to align and concatenate feature maps
        aligned_features_anchor = self.align_and_concatenate_features(features_anchor_scales)
        aligned_features_positive = self.align_and_concatenate_features(features_positive_scales)

        embeds_anchor = self.global_projector(aligned_features_anchor)
        embeds_positive = self.global_projector(aligned_features_positive)

        with torch.no_grad():
            embeds_key = (self.queue.get().detach().clone().to(embeds_anchor.device))

        embeds_anchor = F.normalize(embeds_anchor, p=2, dim=1)
        embeds_positive = F.normalize(embeds_positive, p=2, dim=1)
        embeds_key = F.normalize(embeds_key, p=2, dim=1)

        if self.trainer.current_epoch > self.warmup_epochs:
            logits_12 = torch.matmul(embeds_anchor, embeds_positive.T) / self.temperature
            logits_22 = torch.matmul(embeds_anchor, embeds_key.T) / self.temperature

            loss_1 = (-logits_12.diag()).mean()
            loss_2 = torch.logsumexp(torch.cat([logits_12, logits_22], dim=1), dim=1).mean()
            self.queue.update(embeds_positive.detach().clone().view(-1, embeds_positive.size(-1)))
            running_loss = 0.5 * (loss_1 + loss_2)

            queue_mean = self.queue.get().mean()
            queue_std = self.queue.get().std()
            lambda_ = self.contextualization.lambda_param

            self.log('pretrain/info_nce_loss_local', running_loss, on_epoch=True)
            self.log('queue_mean', queue_mean, on_epoch=True)
            self.log('queue_std', queue_std, on_epoch=True)
            self.log('lambda', lambda_, on_epoch=True)

        else:
            with torch.no_grad():
                self.queue.update(embeds_positive.detach().clone().view(-1, embeds_positive.size(-1)))
            running_loss = None

        return running_loss

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not self.trainer.training:
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
