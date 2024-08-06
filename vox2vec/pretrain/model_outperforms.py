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


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        pos = (1-label) * torch.pow(euclidean_distance, 2)
        neg = (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss_contrastive = torch.mean( pos + neg )
        return loss_contrastive


class Vox2Vec(pl.LightningModule):
    def __init__(
            self,
            backbone: nn.Module,
            base_channels: int,
            num_scales: int,
            # proj_dim: int = 128,
            proj_dim: int = 512,
            temp: float = 0.8,
            # lr: float = 3e-2,
            lr: float = 1e-6,
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
            # nn.Linear(embed_dim, embed_dim),
            # nn.BatchNorm1d(embed_dim),
            # nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
            Lambda(F.normalize)
        )

        self.temp = temp
        self.lr = lr
        self.queue = Queue(max_size=100, embedding_size=128)

    def _vox_to_vec(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
        feature_pyramid = self.backbone(patches)
        return torch.cat([select_from_pyramid([x[j] for x in feature_pyramid], v) for j, v in enumerate(voxels)])





    

    def siamese(self, voxels_1):
        
        patch1 = []
        patch2 = []
        label = []
        for i in range(50):
            x = int(np.random.uniform(0, voxels_1[0].shape[0]))
            # y = int(np.random.uniform(10,512))
            # z = int(np.random.uniform(10,130))

            patch_a = voxels_1[0][x:x+1]
            dif = np.random.randint(32,64)
            x =  x + dif
            patch_p = voxels_1[0][x:x+1]
            
            if patch_a.shape ==(11,3):
                if patch_p.shape==(1,3):
                    patch1.append(patch_a)
                    patch2.append(patch_p)
                    for _ in range(1):
                        label.append(0)


            x = int(np.random.uniform(0, voxels_1[0].shape[0]))
            # y = int(np.random.uniform(10,512))
            # z = int(np.random.uniform(10,130))

            patch_n = voxels_1[0][x:x+1]

            if patch_a.shape ==(1,3):
                if patch_n.shape==(1,3):
                    patch1.append(patch_a)
                    patch2.append(patch_n)

                    for _ in range(1):
                        label.append(1)                    
                    # label.append(1)



        # patch1 = torch.tensor(np.asarray(patch1)).float()
        # patch2 = torch.tensor(np.asarray(patch2)).float()
        # label = torch.tensor(np.asarray(label)).float()

        return patch1, patch2, label

    

    
    def training_step(self, batch, batch_idx):
        """Computes Info-NCE loss.
        """
        patches_1, patches_2, voxels_1, _ = batch['pretrain']
        # random_sample_indice = random.randint(1, voxels_1[0].shape[0])
        # positive_voxels_list, negative_voxels_list = self.neighboring_sampling(voxels_1)
        # voxel_list_positive, voxel_list_negative = self.neighboring_sampling(voxels_1)      
        patch1, patch2, label = self.siamese(voxels_1)
        patch1, patch2 = torch.stack(patch1), torch.stack(patch2)
        label = torch.tensor(label)

        # batch, num_sample, pos_neg_voxels, _ = voxel_list.shape
        assert self.backbone.training
        assert self.proj_head.training
        criterion = ContrastiveLoss()




        
        
        
        embeds_1 = self.proj_head(self._vox_to_vec(patches_1, [patch1.view(-1, 3)]))
        # print(embeds_1.shape)
        # embeds_1 = embeds_1.reshape(-1, num_positives, 128 )
        embeds_1_positive = self.proj_head(self._vox_to_vec(patches_2, [patch1.view(-1, 3)]))
        # embeds_1_positive = embeds_1_positive.reshape(-1, num_positives, 128 )
        embeds_2 = self.proj_head(self._vox_to_vec(patches_1, [patch2.view(-1, 3)]))
        # embeds_2 = embeds_2.reshape(-1, num_negative, 128)
        
        
        loss = criterion(embeds_1, embeds_2, label.to(embeds_1.device))


            
            # self.log(f'pretrain/running_loss', running_loss, on_epoch=True)
        # self.log(f'pretrain/loss_1', loss_1, on_epoch=True)
        # self.log(f'pretrain/loss_2', loss_2, on_epoch=True)
            
            








        return loss
        # return (loss1+loss2)/2

    
    
    
            





    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not self.trainer.training:
            # skip device transfer for the val dataloader
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
