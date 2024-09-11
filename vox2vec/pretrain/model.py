from typing import *
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from vox2vec.nn import Lambda
import time
import matplotlib.pyplot as plt
import math


proj_dim = 1
embed_dim = 1008

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

class Global_projector(nn.Module):
    def __init__(self):
        super(Global_projector, self).__init__()
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

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

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
        temp: float = 0.5,
        lr: float = 5e-5,
        output_size=(1, 1, 1),
        hidden_dim: int = 64  
    ):
        super().__init__()
        self.backbone = backbone
        self.proj_dim = proj_dim
        self.temperature = 0.1
        self.lr = 3e-4
        self.num_scales = 6
        self.output_size = output_size
        self.num_negatives = 10

        self.adaptive_pooling_layers = nn.ModuleList(
            [nn.AdaptiveAvgPool3d(output_size) for _ in range(num_scales)]
        )

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
        self.queue = Queue(max_size=400, embedding_size=proj_dim)
        self.global_projector = Global_projector()

    def extract_features_across_scales(self, feature_pyramid):
        pooled_features = []
        for scale_idx, feature_map in enumerate(feature_pyramid):
            pooled_feature = self.adaptive_pooling_layers[scale_idx](feature_map)
            pooled_features.append(pooled_feature)

        concatenated_features = torch.cat(pooled_features, dim=1)
        return concatenated_features

    def _vox_to_vec(self, patches: torch.Tensor) -> torch.Tensor:
        feature_pyramid = self.backbone(patches)
        concatenated_features = self.extract_features_across_scales(feature_pyramid)
        flattened_features = concatenated_features.view(concatenated_features.size(0), -1)
        return flattened_features

    def ranking_loss(self, similarities):

        rank_loss = 0.0
        for i in range(self.num_negatives-1 ):
            rank_loss += F.relu(-similarities[:, i] + similarities[:, i + 1] + 0.01)
        return rank_loss.mean()
    

    def contrastive_loss(self, logits, scores):
        return F.cross_entropy(logits, scores)


    def generate_single_negative(self, patches):
        negatives = []
        bs = patches.shape[0]
        for i in range(bs):
            negative_patch = patches[i].clone()
            max_num_swaps = 50
            num_swaps = np.random.randint(1, max_num_swaps)
            max_swap_size = int(0.9 * patches.size(-1))  
            for _ in range(num_swaps):
                size = np.random.randint(4, max_swap_size)
                src_h, src_w, src_d = np.random.randint(0, patches.size(-1) - size, 3)
                des_h, des_w, des_d = np.random.randint(0, patches.size(-1) - size, 3)
                negative_patch[:, src_h:src_h+size, src_w:src_w+size, src_d:src_d+size] = \
                    negative_patch[:, des_h:des_h+size, des_w:des_w+size, des_d:des_d+size]
            negatives.append(negative_patch)
        return torch.stack(negatives)







    def generate_negatives(self, patches, num_negatives=4):
        negatives = []
        max_num_swaps = 50

        for patch_idx in range(patches.size(0)):
              
            for i in range(num_negatives):
                negative_patch = patches[patch_idx].clone()
                # total_swap_size = 0
                # num_swaps = math.ceil((i) / (num_negatives * max_num_swaps))  
                num_swaps = np.random.randint(1, max_num_swaps)
                max_swap_size = int(0.9 * patches.size(-1))  
                min_swap_size = 2
                if num_swaps==0:
                    negatives.append(negative_patch)
                else:
                    for _ in range(num_swaps):
                        # size = int(min_swap_size + (i / num_negatives) * (max_swap_size - min_swap_size)) 
                        size = np.random.randint(4, max_swap_size)
                        # size = 16
                        # total_swap_size += size
                        src_h, src_w, src_d = np.random.randint(0, patches.size(-1) - size, 3)
                        des_h, des_w, des_d = np.random.randint(0, patches.size(-1) - size, 3)
                        negative_patch[:, src_h:src_h+size, src_w:src_w+size, src_d:src_d+size] = \
                            negative_patch[:, des_h:des_h+size, des_w:des_w+size, des_d:des_d+size]
                        # negative_patch[:, src_h:src_h+size, src_w:src_w+size, src_d:src_d+size] = 0.
                        

                    negatives.append(negative_patch)

        negatives = torch.stack(negatives)
        return negatives

    def training_step(self, batch, batch_idx):
        patches_1, patches_1_positive, _, _, _ = batch['pretrain']
        num_negatives = self.num_negatives
        # negative_patches = self.generate_negatives(patches_1, num_negatives)
        '''
        slices = torch.cat([negative_patches[i, 0, :, :, 0] for i in range(negative_patches.shape[0])], dim=1)
        plt.imshow(slices.detach().cpu().numpy())
        plt.savefig(str(self.trainer.current_epoch)+"_"+ str(batch_idx)+'.png')
        '''
        
        # with torch.no_grad():
        #     global_anchor = self.proj_head(self._vox_to_vec(patches_1))  # Shape: [B, proj_dim]
        single_negative = self.generate_single_negative(patches_1)
        global_anchor = self.proj_head(self._vox_to_vec(patches_1))  
        global_negative = self.proj_head(self._vox_to_vec(single_negative))

        logits = torch.cat((global_anchor, global_negative), axis=1)
        labels_pos = torch.ones_like(global_anchor).to(global_anchor.device)
        labels_neg = torch.zeros_like(global_anchor).to(global_anchor.device)
        
        labels = torch.cat((labels_pos, labels_neg), dim=1)
        
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        # reg_logits = (torch.matmul(global_negatives, global_negatives.T))
        # reg_loss = torch.mean(-reg_logits.diag() + torch.logsumexp(reg_logits,dim=-1))
        # global_negatives = global_negatives.reshape(-1, num_negatives, self.proj_dim)

        
        # logits = torch.einsum('nc,nkc->nk', global_anchor, global_negatives) / self.temperature
        # softmax_logits = F.softmax(logits, dim=1)

        
        # # target_similarities = torch.tensor([[1, 0.75, 0.5, 0.25]]).to(logits.device)  # You can adjust these values

        
        # # loss = F.mse_loss(softmax_logits, target_similarities.expand_as(softmax_logits))
        # scores = torch.arange(0, num_negatives, device=logits.device).repeat(logits.size(0), 1).float() / (num_negatives - 1)
        
        # scores = F.softmax(scores, dim=-1)
        
        # contrast_loss = self.contrastive_loss(logits, scores)
        # ranking_loss = self.ranking_loss(softmax_logits)
        # loss = 1*contrast_loss + 0.0*reg_loss +1 * ranking_loss
        # print(contrast_loss.item(), ranking_loss.item(), reg_loss.item())

        # if batch_idx % 50 == 0:
        #     print("Softmax Logits:", softmax_logits, self.temperature)
        #     print("Target Similarities:", scores)
        # # distance = -torch.mean(torch.log(torch.abs(scores-logits)), dim=1)
        # distance = torch.nn.L1Loss()(scores, softmax_logits)
        
        # loss = -torch.log(distance) #+ 0.1*self.ranking_loss(softmax_logits)
        
        
        return loss
        
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)



    def validation_step(self, batch, batch_idx):
        pass

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not self.trainer.training:
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
