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
import random 
import matplotlib.pyplot as plt
import time
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

tb = SummaryWriter()


num_positives = 5
num_negative = 5
batch = 5
proj_dim = 128
# proj_dim = 32
max_sampling = 1

img_save_dir = '/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/vox2vec/img_save/'



import torch
import torch.nn as nn
import math

class VoxelPositionalEncoding(nn.Module):
    def __init__(self, dim):
        super(VoxelPositionalEncoding, self).__init__()
        self.linear1 = nn.Linear(3, proj_dim)
        self.dim = dim
    def forward(self, anchor_voxels, target_voxels):
        anchor_voxels = anchor_voxels.expand(-1, target_voxels.size(1), -1)
        voxels = anchor_voxels - target_voxels
        voxels = voxels/128
        pe = (self.linear1(voxels))
        return pe

       
class Projector(nn.Module):
    def __init__(self, proj_dim, embed_dim):
        super(Projector, self).__init__()
        self.pe = VoxelPositionalEncoding(proj_dim)
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )
    
    def forward(self, x, anchor, target):
        x = self.proj_head(x) 
        position = self.pe(anchor, target).view(-1, x.size(-1))
        # x_encoded = x.unsqueeze(1) + position  
        x_encoded = x + position
        # return x_encoded.squeeze(1)
        return x_encoded




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
        # q = F.normalize(self.queue.clone(), p=2, dim=1)
        q = self.queue.clone()[:]
        return q

    def size(self):
        return min(self.max_size, self.ptr)
    
    
    @torch.no_grad()
    def shuffle(self):
        # Shuffle the embeddings up to the current pointer if the queue is not full
        # or shuffle the entire queue if it is full or if the pointer has wrapped around
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
            # temp: torch.nn.Parameter(torch.tensor(0.1)),
            # lr: float = 3e-4,
            lr: float = 1e-2,
            # lr: float = 5e-5,
    ):

        super().__init__()


        


        self.save_hyperparameters(ignore='backbone')
        self.epoch = 0
        self.backbone = backbone
        embed_dim = sum_pyramid_channels(base_channels, num_scales)
        # embed_dim = 48
        self.pe_size = embed_dim
        # self.proj_head = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.BatchNorm1d(embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.BatchNorm1d(embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim, proj_dim),
        #     # nn.Sigmoid(),
        #     # Lambda(F.normalize)
        # )

        self.backbone_key = deepcopy(self.backbone)
        for param_k in self.backbone_key.parameters():
            param_k.requires_grad = False  
        
        
        # self.proj_head_key = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.BatchNorm1d(embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.BatchNorm1d(embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim, proj_dim),
        #     # nn.Sigmoid(),
        #     # Lambda(F.normalize)
        # )
        

        self.projector = Projector(proj_dim, embed_dim)
        # for param_q, param_k in zip(self.proj_head.parameters(), self.proj_head_key.parameters()):
        #     param_k.requires_grad = False  
        



        self.pe = VoxelPositionalEncoding(dim=self.pe_size)

        self.temperature = torch.nn.Parameter(torch.tensor(0.07))
        # self.temperature = 0.9
        # self.temp = 1.0
        self.lr = lr
        self.queue = Queue(max_size=20, embedding_size=proj_dim)

    def _vox_to_vec(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
        feature_pyramid = self.backbone(patches)[:]
        return torch.cat([select_from_pyramid([x[j] for x in feature_pyramid], v) for j, v in enumerate(voxels)])
    
    def _vox_to_vec_key(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
        feature_pyramid = self.backbone_key(patches)[:]
        return torch.cat([select_from_pyramid([x[j] for x in feature_pyramid], v) for j, v in enumerate([voxels])])

    @torch.no_grad()
    def momentum_update(self):
        momentum = 0.99
        with torch.no_grad():
            for param_q, param_k in zip(self.backbone.parameters(), self.backbone_key.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
            # for param_q, param_k in zip(self.proj_head.parameters(), self.proj_head_key.parameters()):
            #     param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        


    def training_step(self, batch, batch_idx):
        patches_1, patches_1_positive, anchor_voxel_1, positive_voxels, negative_voxels = batch['pretrain']
        # voxels (bs, num_pos, dim)
        assert self.backbone.training
        self.backbone_key.training = False
        assert not self.backbone_key.training
        
        self.momentum_update()
        bs = positive_voxels.size(0)

        # embeds_1_key = self.proj_head_key((self._vox_to_vec_key(patches_1, negative_voxels)) + (self.pe(anchor_voxel_1.float(), negative_voxels.float())).view(-1, self.pe_size))
        # self.queue.update(embeds_1_key)

        running_loss = 0
        
        # anchor_voxel_1, positive_voxels, negative_voxels = anchor_voxel_1.view(-1, 3), positive_voxels.view(-1, 3), negative_voxels.view(-1, 3)

        embeds_anchor = self.projector(self._vox_to_vec(patches_1, anchor_voxel_1), anchor_voxel_1, anchor_voxel_1)
        embeds_positive = self.projector(self._vox_to_vec(patches_1_positive, positive_voxels), anchor_voxel_1, positive_voxels)
        embeds_negative = self.projector(self._vox_to_vec(patches_1, negative_voxels), anchor_voxel_1, negative_voxels)
        
        embeds_anchor = F.normalize(embeds_anchor, p=2, dim=1)
        embeds_positive = F.normalize(embeds_positive, p=2, dim=1)
        embeds_negative = F.normalize(embeds_negative, p=2, dim=1)

        # print(embeds_anchor.shape, embeds_positive.shape, embeds_negative.shape) #(bs, 128), (bs*num_positive, 128), (bs*num_negative, 128)
        
        embeds_anchor = embeds_anchor.view(bs, -1, 128)
        embeds_positive = embeds_positive.view(bs, -1, 128)
        embeds_negative = embeds_negative.view(bs, -1, 128)

        '''        
        pos_sim = torch.matmul(embeds_anchor, embeds_positive.transpose(-1, -2)) / self.temperature  # (batch_size, 1, num_pos)
        neg_sim = torch.matmul(embeds_anchor, embeds_negative.transpose(-1, -2)) / self.temperature  # (batch_size, 1, num_neg)
        pos_exp = torch.exp(pos_sim)
        neg_exp = torch.exp(neg_sim).sum(dim=-1, keepdim=True)
        running_loss = -torch.log(pos_exp / (pos_exp + neg_exp)).mean()
        
        '''

        pos_sim = torch.bmm(embeds_positive, embeds_anchor.squeeze(1).unsqueeze(-1)).squeeze(-1)  # (bs, num_positive)
        neg_sim = torch.bmm(embeds_negative, embeds_anchor.squeeze(1).unsqueeze(-1)).squeeze(-1)  # (bs, num_negative)

       
        pos_sim = pos_sim / self.temperature  # (bs, num_positive)
        neg_sim = neg_sim / self.temperature  # (bs, num_negative)

        # Create labels, the positives should be classified higher than any negative (logits)
        labels = torch.zeros(embeds_anchor.size(0), dtype=torch.long, device=embeds_anchor.device)  # (bs)

        # Concatenate positive and negative similarities along the second dimension
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # (bs, num_positive + num_negative)

        # Cross entropy loss to classify the positives (first `num_positive` logits should be higher)
        running_loss = F.cross_entropy(logits, labels)










        # pos_distance = F.pairwise_distance(embeds_anchor, embeds_positive, keepdim=True) 
        # neg_distance = F.pairwise_distance(embeds_anchor, embeds_negative, keepdim=True)
        # positive_loss = pos_distance.mean()
        # negative_loss = F.relu(3 - neg_distance.mean())
        
        # running_loss = positive_loss + negative_loss
        
        # print(f'{self.temperature}, --expl---{positive_loss.item()}, -----------, {negative_loss.item()}, ----- {self.backbone.first_conv.weight[0, 0, 0]}')


        # # pos_similarities = torch.bmm(embeds_positive, embeds_anchor.unsqueeze(2)).squeeze(2) / self.temperature
        # pos_similarities = torch.mm(embeds_anchor, embeds_positive.T)/self.temperature

        # neg_similarities = torch.mm(embeds_anchor, embeds_negative.T) / self.temperature

        # logits = torch.cat([pos_similarities, neg_similarities], dim=1)  # Shape (bs, 20 + 65000)
        # # labels = torch.zeros(logits.size(0), dtype=torch.long).to(embeds_anchor.device)
        # labels_positive = torch.ones(pos_similarities.size()).to(pos_similarities.device)
        # labels_negative = torch.zeros(neg_similarities.size()).to(pos_similarities.device) 
        # # labels = torch.cat([labels_positive, labels_negative], dim=1)  # Shape (bs, 20 + 65000)
        # labels = torch.arange(logits.size(0), dtype=torch.long).to(embeds_anchor.device) 
        # # log_probs = F.log_softmax(logits, dim=1)
        # # log_probs = F.sigmoid(logits)

        # running_loss = F.cross_entropy(logits, labels)
        # # running_loss = F.nll_loss(log_probs, labels)
        # # loss = F.mse_loss(log_probs, labels)
        # # loss = F.binary_cross_entropy(log_probs, labels)    
        
        # global_step = str(self.epoch) + "_" + str(batch_idx)
        # metadata= ['anchor']*bs + ['positive']*bs + ['negative']*embeds_negative.size(0)
        # all_embeddings = torch.cat((embeds_anchor, embeds_positive.view(-1, embeds_positive.size(-1)), embeds_negative), dim=0)


        # loss_pos = F.mse_loss(embeds_anchor, embeds_positive)
        # loss_neg = F.mase_loss(embeds_anchor, embeds_negative)
        
        # running_loss = F.relu(loss_neg- loss_pos + 100)
        
        


        
    
            

        # if batch_idx+1 ==100:
        #     tb.add_embedding(all_embeddings, metadata=metadata, label_img=None, global_step=global_step, tag='all_embedding')
        #     self.epoch+=1
    

        # metadata = labels_positive + labels_positive + labels_negative

# Combine embeddings
        # all_embeddings = torch.cat([embeds_anchor, embeds_positive, embeds_negative], dim=0)

        return running_loss

    
    
    
            




    tb.close()
    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not self.trainer.training:
            # skip device transfer for the val dataloader
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
