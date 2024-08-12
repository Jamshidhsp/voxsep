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
max_sampling = 1

img_save_dir = '/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/vox2vec/img_save/'


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
            lr: float = 5e-5,
    ):

        super().__init__()

        self.save_hyperparameters(ignore='backbone')
        self.epoch = 0

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
            # nn.Sigmoid(),
            # Lambda(F.normalize)
        )

        self.backbone_key = deepcopy(self.backbone)
        for param_k in self.backbone_key.parameters():
            param_k.requires_grad = False  
        
        
        self.proj_head_key = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
            # nn.Sigmoid(),
            # Lambda(F.normalize)
        )
        
        for param_q, param_k in zip(self.proj_head.parameters(), self.proj_head_key.parameters()):
            param_k.requires_grad = False  
        





        self.temperature = torch.nn.Parameter(torch.tensor(0.9))
        # self.temperature = 1.0
        # self.temp = 1.0
        self.lr = lr
        self.queue = Queue(max_size=20, embedding_size=proj_dim)

    def _vox_to_vec(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
        feature_pyramid = self.backbone(patches)[:]
        return torch.cat([select_from_pyramid([x[j] for x in feature_pyramid], v) for j, v in enumerate(voxels)])
    
    def _vox_to_vec_key(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
        feature_pyramid = self.backbone_key(patches)[:]
        return torch.cat([select_from_pyramid([x[j] for x in feature_pyramid], v) for j, v in enumerate(voxels)])

    @torch.no_grad()
    def momentum_update(self):
        momentum = 0.99
        with torch.no_grad():
            for param_q, param_k in zip(self.backbone.parameters(), self.backbone_key.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
            for param_q, param_k in zip(self.proj_head.parameters(), self.proj_head_key.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        


    def training_step(self, batch, batch_idx):
        patches_1, patches_1_positive, anchor_voxel_1, positive_voxels, negative_voxels = batch['pretrain']
        # current_time = str(time.time())
        
        assert self.backbone.training
        assert self.proj_head.training
        
        self.backbone_key.training = False
        self.proj_head_key.training = False

        assert not self.backbone_key.training
        assert not self.proj_head_key.training
        
        self.momentum_update()

        # with torch.no_grad():
        #     embeds_1_key = self.proj_head_key(self._vox_to_vec_key(patches_1, negative_voxels))
        # self.queue.update(embeds_1_key)

        
        # anchor_voxel_1 = [voxels.view(1, 3) for voxels in anchor_voxel_1]
        # with torch.no_grad():
        #     embeds_anchor = self.proj_head(self._vox_to_vec(patches_1, anchor_voxel_1))
        # bs = embeds_anchor.size(0)
        # embeds_positive = [self.proj_head(self._vox_to_vec(patches_1_positive, [voxels])) for voxels in positive_voxels]
        # embeds_positive = [embed[None, :, : ] for embed in embeds_positive]
        # embeds_positive = torch.cat(embeds_positive, dim=0) #(bs, num_positive, proj_dim)
        
        # self.queue.shuffle() 
        # embeds_negative = self.queue.get().to(embeds_positive.device)
        
        with torch.no_grad():
            embeds_1_key = self.proj_head_key(self._vox_to_vec_key(patches_1, negative_voxels))
        self.queue.update(embeds_1_key)

        anchor_voxel_1 = torch.stack([voxels.view(1, 3) for voxels in anchor_voxel_1])

        with torch.no_grad():
            embeds_anchor = self.proj_head(self._vox_to_vec(patches_1, anchor_voxel_1))

        embeds_positive = [self.proj_head(self._vox_to_vec(patches_1_positive, [voxels]))[None, :, :] for voxels in positive_voxels]

        embeds_positive = torch.cat(embeds_positive, dim=0)  # (bs, num_positive, proj_dim)

        self.queue.shuffle()
        embeds_negative = self.queue.get().to(embeds_positive.device)




        
        
        embeds_anchor = F.normalize(embeds_anchor, p=2, dim=1)
        embeds_positive = F.normalize(embeds_positive, p=2, dim=2)
        embeds_negative = F.normalize(embeds_negative, p=2, dim=1)

        # Compute similarities with embeds_positive
        running_loss = 0
        # This results in a tensor of shape (bs, 20)
        pos_similarities = torch.bmm(embeds_positive, embeds_anchor.unsqueeze(2)).squeeze(2) / self.temperature


        neg_similarities = torch.mm(embeds_anchor, embeds_negative.T) / self.temperature

        logits = torch.cat([pos_similarities, neg_similarities], dim=1)  # Shape (bs, 20 + 65000)
        # labels = torch.zeros(logits.size(0), dtype=torch.long).to(embeds_anchor.device)
        labels_positive = torch.ones(pos_similarities.size()).to(pos_similarities.device)
        labels_negative = torch.zeros(neg_similarities.size()).to(pos_similarities.device) 
        labels = torch.cat([labels_positive, labels_negative], dim=1)  # Shape (bs, 20 + 65000)
        # labels = torch.zeros(logits.size(0), dtype=torch.long).to(embeds_anchor.device) 
        # log_probs = F.log_softmax(logits, dim=1)
        log_probs = F.sigmoid(logits)
        
        # print(log_probs.shape, labels.shape)

        
        # loss = F.nll_loss(log_probs, roi_voxels_1labels)
        loss = F.mse_loss(log_probs, labels)
        # loss = F.binary_cross_entropy(log_probs, labels)    
        running_loss=loss
    
        global_step = str(self.epoch) + "_" + str(batch_idx)
        metadata= ['anchor']*4 + ['positive']*40 + ['negative']*embeds_negative.size(0)
        all_embeddings = torch.cat((embeds_anchor, embeds_positive.view(-1, embeds_positive.size(-1)), embeds_negative), dim=0)

    # tb.add_embedding(embeds_anchor, metadata=None, label_img=None, global_step=batch_idx, tag='Anchor_Embeddings')


    # tb.add_embedding(embeds_positive.view(-1, embeds_positive.size(-1)), metadata=None, label_img=None, global_step=batch_idx, tag='Positive_Embeddings')


    # tb.add_embedding(embeds_negative, metadata=None, label_img=None, global_step=batch_idx, tag='Negative_Embeddings')
        if batch_idx+1 ==100:
            tb.add_embedding(all_embeddings, metadata=metadata, label_img=None, global_step=global_step, tag='all_embedding')
            self.epoch+=1
    

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
