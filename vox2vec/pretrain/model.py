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

# tb = SummaryWriter()

proj_dim = 128
# proj_dim = 32



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
            # x = self.avg_pool(x).view(x.shape[0], -1)
            x = self.linear1(x.view(x.shape[0], -1))
            x = self.relu(x)
            x = self.linear2(x)
            return x



# class Projector(nn.Module):
#     def __init__(self, proj_dim, embed_dim):
#         super(Projector, self).__init__()
#         self.pe = VoxelPositionalEncoding(proj_dim)
#         self.proj_head = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.BatchNorm1d(embed_dim),
#             nn.ReLU(),
#             nn.Linear(embed_dim, embed_dim),
#             nn.BatchNorm1d(embed_dim),
#             nn.ReLU(),
#             nn.Linear(embed_dim, proj_dim),
#         )
    
#     def forward(self, x, anchor, target):
#         x = self.proj_head(x) 
#         position = self.pe(anchor, target).view(-1, x.size(-1))
#         # x_encoded = x + position
#         # return x_encoded
#         return x

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
        # q = F.normalize(self.queue.clone(), p=2, dim=1)[:1024]
        q = F.normalize(self.queue.clone(), p=2, dim=1)[:]
        # q = self.queue.clone()[:]
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
            lr: float = 3e-4,
            # lr: float = 1e-3,
            # lr: float = 5e-5,
    ):

        super().__init__()


        


        self.save_hyperparameters(ignore='backbone')
        self.epoch = 0
        self.backbone = backbone
        embed_dim = sum_pyramid_channels(base_channels, num_scales)
        self.pe_size = embed_dim
        # self.attention = FeatureAttention(embed_dim)
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


        self.backbone_key = deepcopy(self.backbone)
        for param_k in self.backbone_key.parameters():
            param_k.requires_grad = False  

        # self.projector = Projector(proj_dim, embed_dim)

        self.pe = VoxelPositionalEncoding(dim=self.pe_size)
        self.temperature = 0.1
        # self.temperature = torch.nn.Parameter(torch.tensor(0.1))
        # self.reg_lambda = torch.nn.Parameter(torch.tensor(0.5))
        
        self.lr = lr
        self.queue = Queue(max_size=6500, embedding_size=proj_dim)
        self.global_projector = Global_projector()

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

        if self.epoch < 1:
            self.reg_lambda = 1.0
            self.warm_lambda = 0.0
        else:
            self.reg_lambda = 0.05
            self.warm_lambda = 1.0

        patches_1, patches_1_positive, anchor_voxel_1, _, _ = batch['pretrain']
        
        positive_voxels = anchor_voxel_1
        patches_1_negative = patches_1.clone()  
        
        '''
        max_shuffle_size=int(0.5*patches_1.size(-1))
        for i in range(patches_1.size(0)):
            # patches_1_negative_list = []
            for _ in range(np.random.randint(1, 10)):
                size = np.random.randint(4, max_shuffle_size)
                # size = 8
                src_h, src_w, src_d = np.random.randint(0, patches_1.size(-1)-size, 3)
                des_h, des_w, des_d =  np.random.randint(0, patches_1.size(-1)-size, 3)
                patches_1_negative[i, 0, src_h:src_h+size, src_w:src_w+size, src_d:src_d+size] = patches_1_negative[i, 0, des_h:des_h+size, des_w:des_w+size, des_d:des_d+size]
                # patches_1_negative_list.append(patches_1_negative)
        
        assert self.backbone.training
        self.backbone_key.training = False
        assert not self.backbone_key.training
        # plt.imshow(torch.cat(patches_1[0, 0][:, :, 16], patches_1_negative[0, 0][:, :, 16]).detach().cpu().numpy())
        bs = positive_voxels.size(0)

        self.momentum_update()
        running_loss = 0
        # embeds_anchor = self.proj_head(self._vox_to_vec(patches_1, anchor_voxel_1), anchor_voxel_1, anchor_voxel_1)
        embeds_anchor = self.proj_head(self._vox_to_vec(patches_1, anchor_voxel_1))
        with torch.no_grad():
        # embeds_positive = self.proj_head(self._vox_to_vec_key(patches_1_positive, positive_voxels), anchor_voxel_1, anchor_voxel_1)
            embeds_positive = self.proj_head_key(self._vox_to_vec_key(patches_1_positive, positive_voxels))
        # embeds_negative = self.projector(self._vox_to_vec(patches_1, negative_voxels), anchor_voxel_1, anchor_voxel_1)
        
        # self.queue.shuffle()
        embeds_key = self.queue.get().to(embeds_anchor.device)
        # print(self.backbone.right_blocks[0].layers[1].layers[5].grad[0])

        embeds_anchor = F.normalize(embeds_anchor, p=2, dim=1)
        embeds_positive = F.normalize(embeds_positive, p=2, dim=1)
        # embeds_negative = F.normalize(embeds_negative, p=2, dim=1)
        # embeds_key = F.normalize(embeds_key, p=2, dim=1)
        
        # embeds_anchor = embeds_anchor.view(bs, 128)
        # embeds_anchor = embeds_anchor.view(bs, -1, 128)
        # embeds_positive = embeds_positive.view(bs, -1, 128)
        # embeds_negative = embeds_negative.view(bs, -1, 128)

                
        # pos_sim = torch.matmul(embeds_anchor, embeds_positive.transpose(-1, -2)) / self.temperature  # (batch_size, 1, num_pos)
        # neg_sim = torch.matmul(embeds_anchor, embeds_negative.transpose(-1, -2)) / self.temperature  # (batch_size, 1, num_neg)
        logits_12 = torch.matmul(embeds_anchor, embeds_positive.T)/self.temperature
        logits_22 = torch.matmul(embeds_anchor, embeds_key.T)/self.temperature
        # pos_exp = torch.exp(pos_sim)
        self.queue.update(embeds_positive.view(-1, 128))
        # neg_exp = torch.exp(neg_sim).sum(dim=-1, keepdim=True)
        # running_loss = -torch.log(pos_exp / (pos_exp + neg_exp)).mean()

        loss_1 = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_12.fill_diagonal_(float('-inf')), logits_22], dim=1), dim=1))
        # loss_1 = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_12, logits_12], dim=1), dim=1))
        '''
        global_positive = self.global_projector(self.backbone(patches_1)[-1])
        global_negative = self.global_projector(self.backbone(patches_1_negative)[-1])
        global_logits = torch.cat((global_positive, global_negative), dim=1)
        labels = torch.arange(2.0).repeat(global_logits.shape[0], 1).to(global_logits.device)
        global_loss = F.binary_cross_entropy_with_logits(global_logits, labels)
        # global_logits = F.softmax(global_logits)
        # global_loss = F.binary_cross_entropy_with_logits(global_logits, labels)

        # global_loss = F.relu(10+global_positive[0]-global_negative[0])
        # print(f'{loss_1.item()}----------{global_loss.item()} ')


        
        
        
        
        
        running_loss = global_loss
        # running_loss = 0.*(loss_1) + 1.*global_loss

        # self.log(f'pretrain/info_nce_loss_local', loss_1, on_epoch=True)
        # self.log(f'pretrain/global_loss', global_loss, on_epoch=True)
        # self.log(f'pretrain/running_loss', running_loss, on_epoch=True)


        
        # pos_sim = torch.einsum('b d, b p d -> b p', embeds_anchor, embeds_positive)  # (bs, num_positive)
        # neg_sim = torch.einsum('b d, b n d -> b n', embeds_anchor, embeds_negative)  # (bs, num_negative)
       
        # pos_sim = pos_sim / self.temperature  # (bs, num_positive)
        # neg_sim = neg_sim / self.temperature  # (bs, num_negative)

        # labels = torch.zeros(embeds_anchor.size(0), dtype=torch.long, device=embeds_anchor.device)  # (bs)

        # logits = torch.cat([pos_sim, neg_sim], dim=1)
        # # logits = pos_sim
        # log_probs = F.log_softmax(logits, dim=1)
        # # log_probs = F.softmax(logits, dim=1)
        # # log_probs = F.sigmoid(logits)
        # # pos_loss = log_probs[:, :pos_sim.size(1)].mean(dim=1)
        # pos_loss = log_probs[:, :pos_sim.size(1)]
        # # labels = torch.zeros(logits.size()).to(log_probs.device)
        # # labels[:, :pos_loss.size(1)] = 1
        # # labels = torch.zeros(pos_loss.size()).to(pos_loss.device)
        # # loss = nn.BCELoss()(pos_loss, labels)
        # loss = -pos_loss.mean()
        # # loss_pos = F.mse_loss(embeds_anchor, embeds_positive)
        # # loss_neg = F.mse_loss(embeds_anchor, embeds_negative)
        
        # reg_1_loss = orthogonality(embeds_anchor)
        # reg_2_loss = orthogonality(embeds_positive.view(-1, 128))
        
        # print(f'------------{self.warm_lambda}, --------{self.reg_lambda}')
        
        # running_loss = self.warm_lambda*(loss) + ((self.reg_lambda))*(reg_1_loss +reg_2_loss) 
        # running_loss = 1+ warm_lambda*(loss_pos - loss_neg) + (torch.abs(self.reg_lambda))*(reg_1_loss +reg_2_loss) 
        # tb.add_text('experiment', 'only_positive_sanity_check', 0)
        # tb.add_scalar('Loss/loss_neg', loss_neg.item(), batch_idx)
        # tb.add_scalar('Loss/loss_pos', loss_pos.item(), batch_idx)
        # tb.add_scalar('Loss/regularizer', self.reg_lambda.item(), batch_idx)
        # tb.add_scalar('Loss/running_loss', running_loss.item(), batch_idx)
        
        # global_step = str(self.epoch) + "_" + str(batch_idx)
        # metadata= ['anchor']*bs*embeds_positive.size(1) + ['positive']*embeds_positive.size(1)*bs + ['negative']*embeds_negative.size(1)*bs
        # all_embeddings = torch.cat((embeds_anchor, embeds_positive.view(-1, embeds_positive.size(-1)), embeds_negative.view(-1, embeds_negative.size(-1))), dim=0)
        # all_embeddings = torch.cat((embeds_anchor.view(-1, embeds_positive.size(-1)), embeds_positive.view(-1, embeds_positive.size(-1)), embeds_negative.view(-1, embeds_negative.size(-1))), dim=0)
        # if batch_idx+1 ==100:
            # tb.add_embedding(all_embeddings, metadata=metadata, label_img=None, global_step=global_step, tag='all_embedding')
        # self.epoch+=1
    

        # metadata = labels_positive + labels_positive + labels_negative

# Combine embeddings
        # all_embeddings = torch.cat([embeds_anchor, embeds_positive, embeds_negative], dim=0)

        return running_loss

    
    
    
            




    # tb.close()
    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not self.trainer.training:
            # skip device transfer for the val dataloader
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
