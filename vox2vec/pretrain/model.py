from typing import *

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
        # self.queue = torch.randn(max_size, embedding_size)
        self.queue = torch.ones((max_size, embedding_size))
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
        q = F.normalize(self.queue.clone(), p=2, dim=1)
        # q = self.queue.clone()
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
        """vox2vec model.

        Args:
            backbone (nn.Module):
                Takes an image of size ``(n, c, h, w, d)`` and returns a feature pyramid of sizes
                ``[(n, c_b, h_b, w_b, d_b), (n, c_b * 2, h_b // 2, w_b // 2, d_b // 2), ...]``,
                where ``c_b = base_channels`` and ``(h_b, w_b, d_b) = (h, w, d)``.
            base_channels (int):
                A number of channels in the base of the output feature pyramid.
            num_scales (int):
                A number of feature pyramid levels.
            proj_dim (int, optional):
                The output dimensionality of the projection head. Defaults to 128.
            temp (float, optional):
                Info-NCE loss temperature. Defaults to 0.1.
            lr (float, optional):
                Learning rate. Defaults to 3e-4.
        """
        super().__init__()

        self.save_hyperparameters(ignore='backbone')

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
        





        # self.temp = torch.nn.Parameter(torch.tensor(0.9))
        self.temp = 1.0
        self.lr = lr
        self.queue = Queue(max_size=65000, embedding_size=proj_dim)

    def _vox_to_vec(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
        feature_pyramid = self.backbone(patches)[:]
        return torch.cat([select_from_pyramid([x[j] for x in feature_pyramid], v) for j, v in enumerate(voxels)])
    
    def _vox_to_vec_key(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
        feature_pyramid = self.backbone_key(patches)[:]
        return torch.cat([select_from_pyramid([x[j] for x in feature_pyramid], v) for j, v in enumerate(voxels)])

    @torch.no_grad()
    def momentum_update(self):
        momentum = 0.9
        with torch.no_grad():
            for param_q, param_k in zip(self.backbone.parameters(), self.backbone_key.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
            for param_q, param_k in zip(self.proj_head.parameters(), self.proj_head_key.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        


    def neighboring_sampling(self, voxels_list):
        step = 5
        positive_list_all = []
        negative_list_all = []
        
        for i in range(len(voxels_list)):
            voxels = voxels_list[i]
            positive_list = []
            negative_list = []
            voxels_numpy = voxels.cpu().numpy()
            counter = 0
            while counter < max_sampling:  
                random_sample_index = np.random.randint(voxels.shape[0], size=1)
                mask = np.all(voxels_numpy[:, :2]==voxels_numpy[:, :2], axis=1)
                # mask=True

                
                valid_1_1 = np.all((voxels_numpy >= voxels[random_sample_index].cpu().numpy() - step) & (voxels_numpy < voxels[random_sample_index].cpu().numpy() + step), axis=1)
                # valid_1_2 = np.all((voxels_numpy >= voxels[random_sample_index].cpu().numpy() + step), axis=1)
                valid_1 = mask & valid_1_1
                valid_1 = np.where(valid_1)[0]
                # valid_2 = np.where(valid_1_2)


                



                if len(valid_1) >= num_positives:
                    positive_indices = np.random.choice(valid_1, num_positives, replace=False)
                    positive_voxels = voxels[positive_indices]
                    positive_list.append(positive_voxels)
                    # negative_list.append(negative_voxels)
                    counter += 1
                    
            positive_list_all.append(torch.stack(positive_list))
            # negative_list_all.append(torch.stack(negative_list))
        
        
        return torch.stack(positive_list_all)

    
    def training_step(self, batch, batch_idx):
        # print('batch_idx-----------------', batch_idx)

        patches_1, _, voxels_1, _ = batch['pretrain']
        # random_sample_indice = random.randint(1, voxels_1[0].shape[0])
        # positive_voxels_list, negative_voxels_list = self.neighboring_sampling(voxels_1)
        voxel_list_positive = self.neighboring_sampling(voxels_1)      
        
        # print(f'voxel_list_positive: {voxel_list_positive.shape} ----- voxel_list_negative: {voxel_list_negative.shape}')
        
        '''
        checking the sampling strategy   
            
                all_voxels= [voxel_list_positive, voxel_list_negative]
                for i in range(2):
                    for j in range(max_sampling):
                        voxels_plt = all_voxels[i][0][j].detach().cpu().numpy()
                        empty_plane = np.zeros((128, 128))
                        empty_plane[voxels_plt[:, 0], voxels_plt[:, 1]] = 100
                        plt.imshow(empty_plane)
                        name = img_save_dir + str(time.time()) + 'image_'+ str(j+1) + 'out of'  + str(i+1)+'.png'
                        plt.savefig(name)
        '''
        
        assert self.backbone.training
        assert self.proj_head.training

        # print('self.backbone_query', self.backbone.first_conv.weight[0, 0, 0])
        # print('self.backbone_key', self.backbone_key.first_conv.weight[0, 0, 0])
        self.momentum_update()

        # with torch.no_grad():
        with torch.no_grad():
            embeds_1_key = self.proj_head_key(self._vox_to_vec_key(patches_1, [voxel_list_positive.view(-1, 3)]))
        embeds_1 = self.proj_head(self._vox_to_vec(patches_1, [voxel_list_positive.view(-1, 3)]))

        embeds_1 = embeds_1.reshape(-1, num_positives, proj_dim )    #(batch, ,#num_poisitve, embedding)

        self.queue.update(embeds_1_key.view(-1, proj_dim))
        running_loss = 0
        loss_list = []
        for i in range(embeds_1.size(0)):  #for the batch_size
            emb1 = embeds_1[i] 
            emb1_key = embeds_1_key[i] # shape is (num_posive, 128)
            rnd_indx = np.random.randint(len(emb1))
            # rnd_indx = 0
            emb1 = F.normalize(emb1, dim=1)
            shuffle_idx = np.arange((emb1.size(0)))[::-1]
            positive_pair = emb1_key.flip(dims=[0])
            self.queue.shuffle() 
            negative_pair = self.queue.get().to(positive_pair.device)
            

            anchor = emb1[0]
            positive_pairs = emb1[1:]
            negative_pair = self.queue.get().to(positive_pair.device)
            loss_positive = (positive_pairs-anchor).pow(2).sum(1)[0]
            loss_negative = (negative_pair-anchor).pow(2).sum(1)[0]
            # tb.add_scalar('Loss_positive', loss_positive, batch_idx)
            # tb.add_scalar('Loss_negative', loss_negative, batch_idx)
            if (batch_idx+1)%10==0:
                
                tb.add_scalar('temp', self.temp, batch_idx)
                # tb.add_histogram('loss_positive_logits', logits_positive[0], batch_idx)
                # tb.add_histogram('loss_negative_logits', logits_negative[0], batch_idx)
                # tb.add_histogram('conv1.weight', self.backbone.first_conv.weight[0, 0], batch_idx)


            loss = F.relu(loss_positive - loss_negative+1)
            # print('losssss', loss, loss_positive, loss_negative)
            loss_list.append(loss.detach().cpu().numpy())
            running_loss+=loss
        # print(f'loss: {np.mean(np.array(loss_list))}')

        
        return running_loss.mean()

    
    
    
            




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
