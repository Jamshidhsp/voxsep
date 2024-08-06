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
num_negative = 500
batch = 5
class Vox2Vec(pl.LightningModule):
    def __init__(
            self,
            backbone: nn.Module,
            base_channels: int,
            num_scales: int,
            proj_dim: int = 128,
            temp: float = 0.1,
            # lr: float = 3e-4,
            lr: float = 1e-5,
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

    def _vox_to_vec(self, patches: torch.Tensor, voxels: Iterable[torch.Tensor]) -> torch.Tensor:
        feature_pyramid = self.backbone(patches)
        return torch.cat([select_from_pyramid([x[j] for x in feature_pyramid], v) for j, v in enumerate(voxels)])

    
    # def neighboring_sampling(self, voxels_list):
        
    #     # num_positives = num_positives
    #     # num_negative = num_negative
    #     step = 5
    #     positive_list_all = []
    #     negative_list_all = []
    #     positive_negative_all = []
        
    #     for i in range(len(voxels_list)):
    #         # for each batch:
    #         positive_list = []
    #         negative_list = []
    #         positive_negative_list = []
    #         voxels = voxels_list[i]
    #         voxels_numpy = voxels.cpu().numpy()
    #         counter = 0
    #         while counter <20:
    #             # labels = torch.zeros(num_positives+num_negative)
    #             random_sample_indice = np.random.randint(voxels.shape[0], size=1)
    #             valid_1 = np.all((voxels_numpy >= voxels[random_sample_indice].cpu().numpy()-step) & (voxels_numpy < voxels[random_sample_indice].cpu().numpy()+step), axis=1)
    #             valid_2 = np.where(valid_1==False)
    #             valid_1 = np.where(valid_1)
    #             valid_1 = valid_1[0]
    #             valid_1 = np.random.permutation(valid_1)
    #             positive_voxels = voxels[valid_1[:num_positives]]
                
                

            
    #             valid_2 = valid_2[0]
    #             valid_2 = np.random.permutation(valid_2)
    #             negative_voxels = voxels[valid_2[:num_negative]]
    #             # labels[:num_positives] = 1
    #             # positive_list.append(torch.cat((positive_voxels, negative_voxels), axis=0))
    #             if len(negative_voxels) == num_negative:
    #                  if len(positive_voxels) == num_negative:

    #                     positive_list.append(positive_voxels)
    #                     negative_list.append(negative_voxels)
    #                     counter+=1

    #             # positive_negative_list.append(positive_voxels)
    #             # print('1', len(positive_negative_list))
    #             # positive_negative_list.append(negative_voxels)
    #             # print('2', len(positive_negative_list))

                
    #         # return positive_voxels, negative_voxels
    #         # positive_negative_all.append(torch.stack(positive_list))
    #         positive_list_all.append(torch.stack(positive_list))
    #         negative_list_all.append(torch.stack(negative_list))
    #     return torch.stack(positive_list_all), torch.stack(negative_list_all)
        # return torch.stack(positive_negative_all)
    
    

    def neighboring_sampling(self, voxels_list):
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
        step = 5
        positive_list_all = []
        negative_list_all = []
        
        for voxels in voxels_list:
            positive_list = []
            negative_list = []
            voxels_numpy = voxels.cpu().numpy()
            counter = 0
            while counter < 20:  
                random_sample_index = np.random.randint(voxels.shape[0], size=1)
                valid_1 = np.all((voxels_numpy >= voxels[random_sample_index].cpu().numpy() - step) &
                                (voxels_numpy < voxels[random_sample_index].cpu().numpy() + step), axis=1)
                valid_2 = np.all((voxels_numpy >= voxels[random_sample_index].cpu().numpy() - 3*step) &
                                (voxels_numpy < voxels[random_sample_index].cpu().numpy() + 3*step), axis=1)
                
                valid_2 = np.where(valid_2 == False)[0]
                valid_1 = np.where(valid_1)[0]
                # valid_2 = np.where(valid_2)[0]
                
                if len(valid_1) >= num_positives and len(valid_2) >= num_negative:
                    positive_indices = np.random.choice(valid_1, num_positives, replace=False)
                    negative_indices = np.random.choice(valid_2, num_negative, replace=False)
                    
                    positive_voxels = voxels[positive_indices]
                    negative_voxels = voxels[negative_indices]
                    
                    positive_list.append(positive_voxels)
                    negative_list.append(negative_voxels)
                    counter += 1
                    
            positive_list_all.append(torch.stack(positive_list))
            negative_list_all.append(torch.stack(negative_list))
        
        return torch.stack(positive_list_all), torch.stack(negative_list_all)

    
    def training_step(self, batch, batch_idx):
        """Computes Info-NCE loss.
        """
        patches_1, patches_2, voxels_1, _ = batch['pretrain']
        # random_sample_indice = random.randint(1, voxels_1[0].shape[0])
        # positive_voxels_list, negative_voxels_list = self.neighboring_sampling(voxels_1)
        voxel_list_positive, voxel_list_negative = self.neighboring_sampling(voxels_1)       
        # batch, num_sample, pos_neg_voxels, _ = voxel_list.shape
        assert self.backbone.training
        assert self.proj_head.training
        
        
        
        
        embeds_1 = self.proj_head(self._vox_to_vec(patches_1, [voxel_list_positive.view(-1, 3)]))
        embeds_1 = embeds_1.reshape(-1, num_positives, 128 )
        embeds_1_positive = self.proj_head(self._vox_to_vec(patches_2, [voxel_list_positive.view(-1, 3)]))
        embeds_1_positive = embeds_1_positive.reshape(-1, num_positives, 128 )
        embeds_2 = self.proj_head(self._vox_to_vec(patches_1, [voxel_list_negative.view(-1, 3)]))
        embeds_2 = embeds_2.reshape(-1, num_negative, 128)
        
        ruuning_loss = 0
        for i in range(embeds_1.size(0)):
            emb1 = embeds_1[i] # shape is (num_posive, 128)
            emb1_shuffle = embeds_1_positive[i]
            emb2 = embeds_2[i].view(-1, 128)
            shuffle_idx = np.arange(emb1.shape[0])
            shuffle_idx = np.random.permutation(shuffle_idx)
            emb1_shuffle = emb1[shuffle_idx]
            
            l_pos =  torch.einsum('pe, pe->p', [emb1, emb1_shuffle]).unsqueeze(-1)
            # l_neg =  torch.einsum('pe,ke->pk', [emb1, emb2])
            l_neg =  torch.einsum('pe,ke->p', [emb1, emb2])
            N = l_pos.size(0)
            # logits = torch.cat((l_pos, l_neg), dim=1)
            logits = torch.cat((l_pos, l_neg.unsqueeze(1)), dim=0)
            logits /= self.temp
            # labels = torch.zeros((logits.shape[0],), dtype=torch.long).to(l_pos.device)
            labels = torch.zeros((logits.shape), dtype=torch.float).to(l_pos.device)
            labels[N:]=1        
            # loss = F.cross_entropy(logits, labels)
            # apr
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            running_loss =+ loss
            
            
            
            # logits_11 = torch.matmul(emb1, emb1.T) / self.temp # (20,20)
            # logits_12 = torch.matmul(emb1, emb2.T) / self.temp # (20, 400)
            # logits_11.fill_diagonal_(float('-inf'))
            # logits_22 = torch.matmul(emb2, emb2.T) / self.temp
            # logits_22.fill_diagonal_(float('-inf'))
            # loss_1 = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_11, logits_12], dim=1), dim=1))
            # loss_2 = torch.mean(-logits_12.diag() + torch.logsumexp(torch.cat([logits_12.T, logits_22], dim=1), dim=1))
            # loss = 0.5*loss_1 + 0.5*loss_2
            
            
            
            
            
            
            # self.log(f'pretrain/loss1', loss, on_epoch=True)
            # self.log(f'pretrain/loss_1', loss_1, on_epoch=True)
            # self.log(f'pretrain/loss_2', loss_2, on_epoch=True)
            
            
        # self.log(f'pretrain/loss21', loss21, on_epoch=True)
        # self.log(f'pretrain/net_loss', net_loss, on_epoch=True)
            
        return running_loss
        # return loss
    
    
    
            




   
        
        # l_pos = torch.einsum('bpd,bpd->bpp', [embeds_1, embeds_1])
        # l_neg = torch.einsum('bpd,bpd->bpp', [embeds_1, embeds_2])
        # l_pos = torch.einsum('bp, bp->b', [embeds_1, embeds_1]).unsqueeze(0)
        # # l_neg = torch.einsum('bpd,bpd->bp', [embeds_1, embeds_2])
        # l_neg = torch.einsum('bc,bc->b', [embeds_1, embeds_2]).unsqueeze(0)
        # # print(l_pos.shape)

        # # l_pos = torch.einsum('bpd,bjk->b', [embeds_1, embeds_1]).unsqueeze(0)
        # # l_neg = torch.einsum('bpd,bjk->b', [embeds_1, embeds_2]).unsqueeze(0)
        # # print('l_pos.shape', l_pos.shape)
        # N = l_pos.size(0)
        # # print('N', N)
        # logits = torch.cat((l_pos, l_neg), dim=1)
        # logits /= self.temp
        # labels = torch.zeros((logits.shape[0], ), dtype=torch.long).to(l_pos.device)
        # labels[N:]=1        
        # loss = F.cross_entropy(logits, labels)

        # pos_label = torch.ones(l_pos.shape).to(l_pos.device)
        # neg_label = -1*torch.ones(l_neg.shape).to(l_pos.device)

        # loss1 = F.mse_loss(pos_label, l_pos)
        # loss2 = F.mse_loss(neg_label, l_neg, reduction='mean')




        # batch = 4
        # samples = 20
        # embeds_1_positive = torch.normal(0,1,(batch,samples,128))
        # embeds_1_negative = torch.normal(0,1,(batch,samples,128))



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
