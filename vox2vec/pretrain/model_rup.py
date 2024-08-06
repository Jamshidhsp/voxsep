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

num_positives = 50
num_negative = 50
batch = 5
proj_dim = 128



class Queue:
    def __init__(self, max_size, embedding_size):
        self.max_size = max_size
        self.embedding_size = embedding_size
        self.queue = torch.randn(max_size, embedding_size)
        self.ptr = 0  # Pointer to keep track of the current position in the queue
    @torch.no_grad()
    def enqueue(self, item):
        self.queue[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.max_size  # Wrap around if queue is full
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
        return self.queue.clone()

    def size(self):
        return min(self.max_size, self.ptr)



class Vox2Vec(pl.LightningModule):
    def __init__(
            self,
            backbone: nn.Module,
            base_channels: int,
            num_scales: int,
            proj_dim: int = proj_dim,
            temp: float = 0.1,
            lr: float = 3e-4,
            # lr: float = 1e-6,
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
            Lambda(F.normalize)
        )

        self.temp = temp
        self.lr = lr
        self.queue = Queue(max_size=600, embedding_size=proj_dim)

    def _vox_to_vec(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
        feature_pyramid = self.backbone(patches)
        return torch.cat([select_from_pyramid([x[j] for x in feature_pyramid], v) for j, v in enumerate(voxels)])

    

    

    def neighboring_sampling(self, voxels_list, voxels_list_2):
        """
        Sample neighboring voxels for positive and negative samples.
        
        Args:
        - voxels_list (list of torch.Tensor): List of tensors containing voxels.
        - num_positives (int): Number of positive samples to generate.
        - num_negative (int): Number of negative samples to generate.
        
        Returns:
        - positive_list_all (torch.Tensor): Tensor containing positive voxel samples.
        - negative_list_all (torch.Tensor): Tensor containing negative voxel samples.
        """
        step = 10
        positive_list_all = []
        negative_list_all = []
        
        for i in range(len(voxels_list)):
            voxels = voxels_list[i]
            voxels_2 = voxels_list_2[i]

            positive_list = []
            negative_list = []
            voxels_numpy = voxels.cpu().numpy()

            voxels_numpy_2 = voxels_2.cpu().numpy()

            counter = 0
            while counter < 2:  
                random_sample_index = np.random.randint(voxels.shape[0], size=1)
                # valid_1 = np.all((voxels_numpy >= voxels[random_sample_index].cpu().numpy() - step) &
                #                 (voxels_numpy < voxels[random_sample_index].cpu().numpy() + step), axis=1)
                
                
                valid_1 = np.all((voxels_numpy >= voxels[random_sample_index].cpu().numpy() - step) & (voxels_numpy < voxels[random_sample_index].cpu().numpy() + step), axis=1)
                
                random_sample_index = np.random.randint(voxels_2.shape[0], size=1)
                valid_2 = np.all((voxels_numpy_2 >= voxels_2[random_sample_index].cpu().numpy()- step) & (voxels_numpy_2 < voxels_2[random_sample_index].cpu().numpy() + step), axis=1)

                # valid_2 = np.all((voxels_numpy >= voxels[random_sample_index].cpu().numpy() - 3*step) &
                #                 (voxels_numpy < voxels[random_sample_index].cpu().numpy() + 3*step), axis=1)
                
                # valid_2 = np.where(valid_2 == False)[0]
                valid_1 = np.where(valid_1)[0]
                # valid_2 = np.where(valid_2)[0]
                
                valid_2 = np.where(valid_2)[0]

                



                if len(valid_1) >= num_positives and len(valid_2) >= num_negative:
                    positive_indices = np.random.choice(valid_1, num_positives, replace=False)
                    # negative_indices = np.random.choice(valid_2, num_negative, replace=False)
                    
                    positive_voxels = voxels[positive_indices]
                    # i = random.randint(0, 7)
                    # voxel2 = voxels_2[i].cpu().numpy()
                    # valid = np.where((voxel2>0))[0]
                    negative_indices = np.random.choice(valid_2, num_negative, replace=False)
                    negative_voxels = voxels_2[negative_indices]
                    # negative_voxels = voxels[negative_indices]
                    
                    positive_list.append(positive_voxels)
                    negative_list.append(negative_voxels)
                    counter += 1
                    
            positive_list_all.append(torch.stack(positive_list))
            negative_list_all.append(torch.stack(negative_list))
        
        return torch.stack(positive_list_all), torch.stack(negative_list_all)

    
    def training_step(self, batch, batch_idx):
        """Computes Info-NCE loss.
        """
        patches_1, patches_2, voxels_1, voxels_2 = batch['pretrain']
        # random_sample_indice = random.randint(1, voxels_1[0].shape[0])
        # positive_voxels_list, negative_voxels_list = self.neighboring_sampling(voxels_1)
        voxel_list_positive, voxel_list_negative = self.neighboring_sampling(voxels_1, voxels_2)      
        # batch, num_sample, pos_neg_voxels, _ = voxel_list.shape
        assert self.backbone.training
        assert self.proj_head.training
        
        # alpha = self._vox_to_vec(patches_1, [voxel_list_positive.view(-1, 3)])
        # print(alpha.shape)
        
        
        embeds_1 = self.proj_head(self._vox_to_vec(patches_1, [voxel_list_positive.view(-1, 3)]))
        embeds_1 = embeds_1.reshape(-1, num_positives, proj_dim )
        # embeds_1_positive = self.proj_head(self._vox_to_vec(patches_2, [voxel_list_positive.view(-1, 3)]))
        # embeds_1_positive = embeds_1_positive.reshape(-1, num_positives, proj_dim )
        embeds_2 = self.proj_head(self._vox_to_vec(patches_2, [voxel_list_negative.view(-1, 3)]))
        embeds_2 = embeds_2.reshape(-1, num_negative, proj_dim)
        # print(embeds_2.shape)
        # print(embeds_1.shape)
        
        running_loss = 0
        for i in range(embeds_1.size(0)):
            emb1 = embeds_1[i] # shape is (num_posive, 128)
            emb2 = embeds_2[i]
            self.queue.update(emb2)
            l_pos =  torch.einsum('pe, qe->pq', [emb1, emb1])
            l_neg =  torch.einsum('pe,ke->pk', [emb1, emb2])
            logits_pos = l_pos/self.temp
            labels_pos = torch.zeros((logits_pos.shape ), dtype=torch.float).to(l_pos.device)
            loss_ce_p = F.mse_loss(logits_pos, labels_pos)/emb1.shape[0]
            # print(loss_ce_p)

            logits_neg = l_neg/self.temp
            labels_neg = 10*torch.ones((logits_neg.shape ), dtype=torch.float).to(l_neg.device)
            loss_ce_n = F.mse_loss(logits_neg, labels_neg)/emb2.shape[0]
            # print(loss_ce_n)
            loss = (loss_ce_n + loss_ce_p)/2

            # l_neg =  torch.einsum('pe,ke->pk', [emb1, self.queue.get().detach().to(emb1.device)])
            # N = l_pos.size(0)
            # logits = torch.cat((l_pos, l_neg), dim=1)
            # logits /= self.temp
            # # print(l_pos.shape, l_neg.shape)      
            # labels = torch.zeros((logits.shape[0], ), dtype=torch.long).to(l_pos.device)
            # labels[N:]=1  
            # loss = F.cross_entropy(logits, labels)
            
            # print('loss', loss)
            running_loss =+ loss/emb1.shape[0]
            self.log(f'pretrain/l_pos', loss_ce_p, on_epoch=True)
            self.log(f'pretrain/l_neg', loss_ce_n, on_epoch=True)
            self.log(f'pretrain/running_loss', running_loss, on_epoch=True)
        return running_loss
        
        # self.queue.update(embeds_2.view(-1, proj_dim))
        # # print('embeds_1.shape', embeds_1.shape)
        # l_pos =  torch.einsum('bpe, bqe->p', [embeds_1, embeds_1])#.unsqueeze(-1)
        # # print('l_pos.shape', l_pos.shape)
        #     # l_neg =  torch.einsum('pe,ke->pk', [emb1, emb2])
        # l_neg =  torch.einsum('bpe,ke->pk', [embeds_1, self.queue.get().detach().to(embeds_1.device)])
        # # print('l_neg.shape', l_neg.shape)
        # N = l_pos.size(0)
        # logits = torch.cat((l_pos, l_neg), dim=1)
        # logits /= self.temp
        # labels = torch.zeros((logits.shape[0], ), dtype=torch.long).to(l_pos.device)
        # labels[N:]=1        
        # loss = F.cross_entropy(logits, labels)
        # return loss
        
        # logits_11 = torch.matmul(embeds_1, embeds_1.T) / self.temp
        # logits_11.fill_diagonal_(float('-inf'))
        # logits_12 = torch.matmul(embeds_1, embeds_2.T) / self.temp
        # logits_22 = torch.matmul(embeds_2, embeds_2.T) / self.temp
        # logits_22.fill_diagonal_(float('-inf'))
        # loss_1 = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_11, logits_12], dim=1), dim=1))
        # loss_2 = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_12.T, logits_22], dim=1), dim=1))
        # loss = (loss_1 + loss_2) / 2
        
        # self.log(f'pretrain/l_pos', loss_1, on_epoch=True) 
        # self.log(f'pretrain/l_neg', loss_2, on_epoch=True)
        # self.log(f'pretrain/running_loss', loss, on_epoch=True)
        
        
        # return loss

    
    
    
            





    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not self.trainer.training:
            # skip device transfer for the val dataloader
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
