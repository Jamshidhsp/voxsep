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
import matplotlib.pyplot as plt
import random
from torch.utils.tensorboard import SummaryWriter
import time

# save_dir = "/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/vox2vec/img_save_dir/"
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
        queries = F.normalize(queries, dim=-1)
        keys = F.normalize(keys, dim=-1)
        values = F.normalize(values, dim=-1)
        
        contextualized_features, attention_weights = self.attention(queries.unsqueeze(1), keys.unsqueeze(1), values.unsqueeze(1))
        contextualized_features = contextualized_features.squeeze(1)

        contextualized_features = self.lambda_param * contextualized_features + (1 - self.lambda_param) * queries
        return contextualized_features, attention_weights

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
        q = F.normalize(self.queue.clone(), dim=-1)
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
            # nn.Sigmoid(),
            Lambda(F.normalize)
        )

        self.proj_head_key = deepcopy(self.proj_head)

        self.backbone_key = deepcopy(self.backbone)
        for param_k in self.backbone_key.parameters():
            param_k.requires_grad = False  

        self.pe = VoxelPositionalEncoding(dim=self.pe_size)
        self.temperature = 0.1
        self.lr = lr
        self.queue = Queue(max_size=1024, embedding_size=embed_dim)
        self.global_projector = Global_projector()
        self.warmup_epochs = 1
        self.contextualization = AttentionContextualization(feature_dim=embed_dim)

    def _vox_to_vec(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
        feature_pyramid = self.backbone(patches)[:]
        return torch.cat([select_from_pyramid([x[j] for x in feature_pyramid], v) for j, v in enumerate(voxels)])
    
    def _vox_to_vec_key(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
        feature_pyramid = self.backbone_key(patches)[:]
        return torch.cat([select_from_pyramid([x[j] for x in feature_pyramid], v) for j, v in enumerate(voxels)])
    
    def orthogonality(self, embeddings):
        gram_matrix = torch.matmul(embeddings, embeddings.T)
        identity = torch.eye(gram_matrix.size(0), device=embeddings.device)
        ortho_loss = F.mse_loss(gram_matrix, identity)
        return ortho_loss
    
    def pairwise_loss(self, embeds_anchor, embeds_positive, margin=1.0):
        pairwise_distances = torch.cdist(embeds_anchor, embeds_positive, p=2)
        positive_loss = torch.mean(pairwise_distances.diag())
        negative_loss = torch.mean(F.relu(margin - pairwise_distances + torch.eye(pairwise_distances.size(0)).to(embeds_anchor.device) * margin))
        loss = positive_loss + negative_loss
        return loss
        
    @torch.no_grad()
    def momentum_update(self):
        momentum = 1.0
        with torch.no_grad():
            for param_q, param_k in zip(self.backbone.parameters(), self.backbone_key.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
            for param_q, param_k in zip(self.proj_head.parameters(), self.proj_head_key.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        
    def training_step(self, batch, batch_idx):
        patches_1, patches_1_positive, anchor_voxel_1, _, _ = batch['pretrain']
        positive_voxels = anchor_voxel_1

        assert self.backbone.training
        self.backbone_key.training = False
        self.proj_head_key.training = False
        assert not self.backbone_key.training
        bs = positive_voxels.size(0)

        self.momentum_update()

        features_anchor = self._vox_to_vec(patches_1, anchor_voxel_1)
        features_positive = self._vox_to_vec_key(patches_1_positive, positive_voxels)
        
        keys_values = self.queue.get().detach().clone().to(features_anchor.device)

        features_anchor_contextualized, attention_weights_anchor = self.contextualization(features_anchor, keys_values, keys_values)
        features_positive_contextualized, attention_weights_positive = self.contextualization(features_positive, keys_values, keys_values)
        
        # features_anchor_contextualized = features_anchor
        # features_positive_contextualized = features_positive


        tb.add_histogram('AttentionWeights/Anchor', attention_weights_anchor, self.global_step)
        tb.add_histogram('AttentionWeights/Positive', attention_weights_positive, self.global_step)

        embeds_anchor = self.proj_head(features_anchor_contextualized)
        embeds_positive = self.proj_head_key(features_positive_contextualized)
        
        with torch.no_grad():  
            embeds_key = self.proj_head_key(self.queue.get().detach().clone().to(features_anchor.device))

        # embeds_anchor = F.normalize(embeds_anchor, p=2, dim=1)
        # embeds_positive = F.normalize(embeds_positive, p=2, dim=1)
        # embeds_key = F.normalize(embeds_key, p=2, dim=1)

        if self.trainer.current_epoch > self.warmup_epochs:
            # if random.uniform(0, 1) < 0.2:
            #     # self.queue.reset_queue(random.uniform(0, 1))
            #     with torh.no_grad():
            #         self.queue.shuffle()
            # if random.uniform(0, 1) < 10.5:
            #     with torch.no_grad():
            self.queue.update(features_positive_contextualized.detach().clone().view(-1, features_positive_contextualized.size(-1)))
            
            logits_12 = torch.matmul(embeds_anchor, embeds_positive.T) / self.temperature
            logits_22 = torch.matmul(embeds_anchor, embeds_key.T) / self.temperature
            logits_12_1 = torch.matmul(embeds_positive, embeds_key.T) / self.temperature
            logits_11 = torch.matmul(embeds_anchor, embeds_anchor.T) / self.temperature
            logits_11_1 = torch.matmul(embeds_positive, embeds_positive.T) / self.temperature

            loss_1 = (-logits_12.diag()).mean() 
            # loss_2 = torch.logsumexp(torch.cat([logits_12, logits_22, logits_12_1], dim=1), dim=1).mean()
            loss_2 = torch.logsumexp(torch.cat([logits_12.fill_diagonal_(float('-inf')), logits_11.fill_diagonal_(float('-inf')), logits_11_1.fill_diagonal_(float('-inf'))], dim=1), dim=1).mean()
            running_loss=0.5*(loss_1+loss_2)

            # pairwise = self.pairwise_loss(embeds_anchor, embeds_positive)

            # loss_1 = torch.cdist(embeds_anchor, embeds_positive).diag().mean()
            # loss_2 = torch.cdist(embeds_anchor, embeds_key).mean()
            # running_loss = 10 - loss_2 + loss_1
            # running_loss = loss_1 + loss_2 + 0.01*(self.orthogonality(embeds_anchor)+ self.orthogonality(embeds_positive)) + pairwise
            # running_loss = self.spatial_aware_loss(embeds_anchor, embeds_positive, embeds_key, margin=1.0)

            queue_mean = self.queue.get().mean()
            queue_std = self.queue.get().std()
            lambda_ = self.contextualization.lambda_param
            
            self.log('pretrain/info_nce_loss_local', running_loss, on_epoch=True)
            # self.log('pretrain/positive_loss', loss_1, on_epoch=True)
            # self.log('pretrain/negative_loss', loss_2, on_epoch=True)
            # self.log('pretrain/pairwise_loss', pairwise, on_epoch=True)
            self.log('queue_mean', queue_mean, on_epoch=True)
            self.log('queue_std', queue_std, on_epoch=True)
            self.log('lambda', lambda_, on_epoch=True)
        
        else:
            with torch.no_grad():
                self.queue.update(features_positive_contextualized.detach().clone().view(-1, features_positive_contextualized.size(-1)))
            running_loss = None

        global_step = str(self.trainer.current_epoch)
        metadata = ['anchor'] * embeds_positive.view(-1, 128).size(0) + ['positive'] * embeds_positive.view(-1, 128).size(0) + ['negative'] * embeds_key.view(-1, 128).size(0)
        all_embeddings = torch.cat((embeds_anchor.view(-1, embeds_positive.size(-1)), embeds_positive.view(-1, embeds_positive.size(-1)), embeds_key.view(-1, embeds_key.size(-1))), dim=0)
        
        if batch_idx + 1 == 1:
            tb.add_embedding(all_embeddings, metadata=metadata, label_img=None, global_step=global_step, tag='all_embedding')
            
            # tb.add_embedding(embeds_positive, metadata=['positive']*embeds_positive.view(-1, 128).size(0), label_img=None, global_step=global_step, tag="positiive_embedding")
            # tb.add_embedding(embeds_key, metadata=['negative']*embeds_key.view(-1, 128).size(0), label_img=None, global_step=global_step, tag="queue_embedding")
            # tb.add_embedding(embeds_anchor, metadata=['anchor']*embeds_anchor.view(-1, 128).size(0), label_img=None, global_step=global_step, tag="queue_embedding")
            # tb.add_embedding(features_positive, metadata=['feature_positive']*features_positive.size(0), label_img=None, global_step=global_step, tag="feature_positive")
            # tb.add_embedding(features_anchor, metadata=['feature_anchor']*features_anchor.size(0), label_img=None, global_step=global_step, tag="feature_anchor")
            # tb.add_embedding(features_anchor_contextualized, metadata=['feature_anchor_context']*features_anchor_contextualized.size(0), label_img=None, global_step=global_step, tag="feature_positive_context")
            # tb.add_embedding(features_positive_contextualized, metadata=['feature_positive_context']*features_positive_contextualized.size(0), label_img=None, global_step=global_step, tag="feature_anchor_context")

        return running_loss

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not self.trainer.training:
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
