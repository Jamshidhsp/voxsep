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
from vox2vec.nn import FPN3d
from vox2vec.nn import FPN3d_iternorm
import sys

num_positives = 50
num_negative = 500
batch = 5
proj_dim = 128
max_sampling = 1
step =20
d_list = []
d_sorted_list=[]
index_list=[]
color_list = ['blue', 'Red',]
label_list = ['vox2vec', 'Ours']
import gc
import matplotlib.pyplot as plt



def plot(d):
    d_sorted_list=[]
    index_list=[]
    color_list = ['blue', 'Red',]
    label_list = ['vox2vec', 'Ours']
                  
                
    for i in range(len(d)):
    # d_sorted = np.log(np.sort(d))
        array = d[i]
        # array[array<1e-8]=1e-13
        array[array<1e-5]=np.NINF
        d[i]=array
        d_sorted = np.log(d[i])
        d_sorted_list.append(d_sorted)
    # d_sorted = (np.sort(d))
    # d_sorted = d_sorted[::-1]
        index = np.arange(d[i].shape[0])
        index_list.append(index)
    
        plt.plot(index_list[i], d_sorted_list[i], color=color_list[i], label=label_list[i])
        plt.legend(loc="upper right")
        plt.xlabel('Singular Value Rank Index')
        plt.ylabel('log of Singular Value')
    # plt.plot(index_list[1], d_sorted_list[1], color='red', label='test')
    # plt.plot(index_list[2], d_sorted_list[2], color='green', label='test')
    # plt.plot(index_list[3], d_sorted_list[3], color='black', label='test')
    plt.show()
    plt.savefig('all_models.png')


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
        q = F.normalize(self.queue.clone(), p=2, dim=1)
        # Shuffle the embeddings batch
        idx = torch.randperm(q.size(0))
        q = q[idx]
        return q

    def size(self):
        return min(self.max_size, self.ptr)



class Vox2Vec(pl.LightningModule):
    def __init__(
            self,
            backbone: nn.Module,
            base_channels: int,
            num_scales: int,
            proj_dim: int = proj_dim,
            temp: float = 0.5,
            lr: float = 3e-4,
            # lr: float = 1e-5,
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
        # self.backbone = backbone
        self.backbone =  FPN3d(1, 16, 6) 
        self.backbone.traning= False
        
        # ckpt_path = '/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/vox2vec/server_results/checkpoints/iter/epoch=289-step=29000.ckpt'
        ckpt_path = "/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/vox2vec/server_results/checkpoints/normal/epoch=299-step=30000.ckpt"
        sd = torch.load(ckpt_path)['state_dict']
        modified_state_dict = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        self.backbone.load_state_dict(modified_state_dict)
        self.backbone.training=False

        self.backbone_iter = FPN3d_iternorm(1, 16, 6) 
        # ckpt_path = '/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/vox2vec/server_results/checkpoints/iter/epoch=289-step=29000.ckpt'
        ckpt_path = '/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/vox2vec/vox2vec/pretrain/models/best.ckpt'
        sd = torch.load(ckpt_path)['state_dict']
        modified_state_dict = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        self.backbone_iter.load_state_dict(modified_state_dict)
        self.backbone_iter.training=False

        
        
        
        # self.encoder_key = FPN3d(1, 16, 6)
        # self.encoder_key.training = False
        # embed_dim = sum_pyramid_channels(base_channels, num_scales)
        embed_dim = 112
        self.proj_head = nn.Sequential(
            nn.Identity()
        )

        self.temp = temp
        self.lr = lr
        self.queue = Queue(max_size=6000000, embedding_size=proj_dim)

    def _vox_to_vec(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
        feature_pyramid = self.backbone(patches)[:]
        return torch.cat([select_from_pyramid([x[j] for x in feature_pyramid], v) for j, v in enumerate(voxels)])
    def _vox_to_vec_2(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
        feature_pyramid = self.backbone_iter(patches)[:]
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
        # step = 20
        positive_list_all = []
        negative_list_all = []
        
        
        for i in range(len(voxels_list)):
            voxels = voxels_list[i]
            voxels_2 = voxels_list[i]

            positive_list = []
            negative_list = []
            voxels_numpy = voxels.cpu().numpy()

            voxels_numpy_2 = voxels_2.cpu().numpy()

            counter = 0
            while counter < max_sampling:  
                random_sample_index = np.random.randint(voxels.shape[0], size=1)
                # valid_1 = np.all((voxels_numpy >= voxels[random_sample_index].cpu().numpy() - step) &
                #                 (voxels_numpy < voxels[random_sample_index].cpu().numpy() + step), axis=1)
                
                
                valid_1 = np.all((voxels_numpy >= voxels[random_sample_index].cpu().numpy() - step) & (voxels_numpy <= voxels[random_sample_index].cpu().numpy() + step), axis=1)
                valid_2 = np.all((voxels_numpy_2>= voxels[random_sample_index].cpu().numpy()- 1*step) & (voxels_numpy_2< voxels[random_sample_index].cpu().numpy() + 0*step), axis=1)      
                valid_2 = np.where(valid_2 == False)[0]
                valid_1 = np.where(valid_1)[0]


                



                if len(valid_1) >= num_positives and len(valid_2) >= num_negative:
                    positive_indices = np.random.choice(valid_1, num_positives, replace=False)
                    positive_voxels = voxels[positive_indices]
                    negative_indices = np.random.choice(valid_2, num_negative, replace=False)
                    negative_voxels = voxels_2[negative_indices]
                    
                    positive_list.append(positive_voxels)
                    negative_list.append(negative_voxels)
                    counter += 1
                    
            positive_list_all.append(torch.stack(positive_list))
            negative_list_all.append(torch.stack(negative_list))
        
        return torch.stack(positive_list_all), torch.stack(negative_list_all)

    
    
    def singular(self, z, patches, voxels):
        print(z.shape)

        
        
        
        # print(embeds_1.shape)
        with torch.no_grad():
            gc.collect()
            latents = []
            torch.cuda.empty_cache()
            
            self.proj_head.training= False
            
            # print(embeds_1.shape)
            z = torch.nn.functional.normalize(z, dim=1)
            # z = latents
            # calculate covariance
            gc.collect()
            # z = z.cpu().detach().numpy()
            z = torch.transpose(z, 0, 1)
            c = torch.cov(z[:])
            c = c.cpu().detach().numpy()
            rank = np.linalg.matrix_rank(c, 1e-10)
            # rank = torch.linalg.matrix_rank(c)
            _, d, _ = np.linalg.svd(c)
            # z = np.transpose(z)
            # c = np.cov(z)
            # rank = np.linalg.matrix_rank(c, 1e-10)
            print('convariance matrix rank is', rank )
            # _, d, _ = np.linalg.svd(c)

        return d
    
    
    
    def plot(self, d):
        d_sorted_list = []
        index_list = []
        color_list = ['blue', 'red', 'green', 'black']
        label_list = ['Ours', 'vox2vec', 'Local Features w\ Decorrelation', 'Regular']
                    
        fig, ax = plt.subplots()  # Create a single figure and axes outside the loop

        for i in range(len(d)):
            # array = d[i]
            d[i][d[i] < 1e-10] = np.NINF
            # d[i] = array
            d_sorted = np.log(d[i])
            d_sorted_list.append(d_sorted)
            
            index = np.arange(d[i].shape[0])
            index_list.append(index)
            
            ax.plot(index_list[i], d_sorted_list[i], color=color_list[i], label=label_list[i])  # Plot on the same axes

        ax.legend(loc="upper right")
        ax.set_xlabel('Singular Value Rank Index')
        ax.set_ylabel('log of Singular Value')
        plt.savefig('all_models.png')  # Save the figure after plotting all datasets
        # plt.show()  # Display the plot

    
    
    
    
    
    
    
    def training_step(self, batch, batch_idx):
        # print('batch_idx', batch_idx)
        """Computes Info-NCE loss.
        """
        # self.backbone.training=False
        with torch.no_grad():
            gc.collect()
            
    

        patches_1, _, voxels_1, _ = batch['pretrain']
        # voxel_list_positive, voxel_list_negative = self.neighboring_sampling(voxels_1, voxels_1)      
        # assert self.proj_head.training
        self.proj_head.training= False
        gc.collect()
        embeds_1 = 0.0010*self.proj_head(self._vox_to_vec(patches_1, [voxels_1[0][:20000]]))
        # print(embeds_1.shape)
        d = []
        gc.collect()
        d_1 = self.singular(embeds_1, patches_1, voxels_1[:])
        d.append(d_1)
        embeds_1 = self.proj_head(self._vox_to_vec_2(patches_1, [voxels_1[0][:20000]]))
        d_1 = self.singular(embeds_1, patches_1, voxels_1[:])
        gc.collect()
        d.append(d_1)
        # d.append(d_2)
        self.plot(d)
        gc.collect()
        
        # sys.exit()
        
        
    #     z = torch.nn.functional.normalize(embeds_1, dim=1)
    #     print("z.shape", z.shape)
    #     # z = z.cpu().detach().numpy()
    #     # print('z.shape', z.shape)
    #     torch.cuda.empty_cache()
    #     z = torch.transpose(z, 0, 1)

    #     c = torch.cov(z[:])
    #     c = c.cpu().detach().numpy()
    #     rank = np.linalg.matrix_rank(c, 1e-8)
    #     # rank = torch.linalg.matrix_rank(c)
    #     _, d, _ = np.linalg.svd(c)
    #     d_list.append(d)
    #     print('rank', rank)
        
        
        
    #     for i in range(len(d)):
    # # d_sorted = np.log(np.sort(d))
        
       
    #         array = d_list[i]
    #     # array[array<1e-8]=1e-13
    #         array[array<1e-8]=np.NINF
    #         d_list[i]=array
    #         d_sorted = np.log(d[i])
    #         d_sorted_list.append(d_sorted)
    # # d_sorted = (np.sort(d))
    # # d_sorted = d_sorted[::-1]
    #         index = np.arange(d_list[i].shape[0])
    #         index_list.append(index)
    
    #         plt.plot(index_list[i], d_sorted_list[i], color=color_list[i], label=label_list[i])
    #         plt.legend(loc="upper right")
    #         plt.xlabel('Singular Value Rank Index')
    #         plt.ylabel('log of Singular Value')
    #     plt.savefig('all_models.png')
            
        

    
    
    
            





    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not self.trainer.training:
            # skip device transfer for the val dataloader
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)





import matplotlib.pyplot as plt
import numpy as np
