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

num_positives =10
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


    def neighboring_sampling(self, voxels_list):

        step = 5
        positive_list_all = []
        negative_list_all = []
        
        for voxels in voxels_list:
            positive_list = []
            negative_list = []
            voxels_numpy = voxels.cpu().numpy()
            counter = 0
            while counter < 1:  
                random_sample_index = np.random.randint(voxels.shape[0]-10*step, voxels.shape[0]-1*step, size=1)
                valid_1 = np.all(voxels_numpy >= voxels[random_sample_index].cpu().numpy() - step, axis=1) & np.all(voxels_numpy < voxels[random_sample_index].cpu().numpy() + step, axis=1)
                valid_2 = np.all(voxels_numpy >= voxels[random_sample_index].cpu().numpy() - 10*step, axis=1) & np.all((voxels_numpy < voxels[random_sample_index].cpu().numpy() + 10*step), axis=1)
                
                # valid_2 = np.all((voxels_numpy >= voxels[random_sample_index].cpu().numpy() - 10*step) &
                #                 (voxels_numpy < voxels[random_sample_index].cpu().numpy() + 10*step), axis=1)
                
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
        
        running_loss = 0
        for i in range(embeds_1.size(0)):
            j = np.random.randint(0, embeds_1.size(0))
            emb1 = embeds_1[i] # shape is (num_posive, 128)
            emb1_shuffle = embeds_1_positive[i]
            emb2 = embeds_2[j].view(-1, 128)
            shuffle_idx = np.arange(emb1.shape[0])
            shuffle_idx = np.random.permutation(shuffle_idx)
            emb1_shuffle = emb1[shuffle_idx]
            
            # l_pos =  torch.einsum('pe, pe->pe', [emb1, emb1_shuffle])#.unsqueeze(-1)
            
            # l_pos = torch.matmul(emb1, emb1.T)
            # l_pos.fill_diagonal_(float(1))
            
            # l_pos = l_pos/self.temp
            # # l_neg =  torch.einsum('pe,ke->pk', [emb1, emb2])
            # # l_neg =  torch.einsum('pe,ke->p', [emb1, emb2])
            # l_neg =  torch.einsum('pe,ke->pk', [emb1, emb2])
            # l_neg = l_neg/self.temp
            # N = l_pos.size(0)
            
            # labels_pos = torch.zeros(l_pos.shape).to(l_pos.device)
            # labels_neg = torch.ones(l_neg.shape).to(l_pos.device)
            
            
            # labels_pos = torch.ones(l_pos.shape).to(l_pos.device)
            # labels_neg = torch.zeros(l_neg.shape).to(l_pos.device)
            
            # loss_pos = F.binary_cross_entropy(labels_pos, l_pos)
            # loss_neg = F.binary_cross_entropy(labels_neg, l_neg)
            
            # loss_pos = F.mse_loss(labels_pos, l_pos, reduction='mean')
            # loss_neg = F.mse_loss(labels_neg, l_neg, reduction='mean')

            
            
            l_pos = F.cosine_similarity(emb1, emb1_shuffle)
            # print('l_pos.shape', l_pos.shape)
            stacked_emb1 = [emb1]*int(num_negative/num_positives)
            stacked_emb1 = torch.stack(stacked_emb1)
            stacked_emb1 = stacked_emb1.view(-1, 128)
            l_neg = F.cosine_similarity(stacked_emb1, emb2)
            
            



            labels_pos = 1*torch.ones(l_pos.shape).to(l_pos.device)
            labels_neg = torch.zeros(l_neg.shape).to(l_pos.device)
            labels = torch.cat((labels_pos, labels_neg), dim=0)
            logits = torch.cat((l_pos, l_neg), dim=0)
            # print(F.sigmoid(logits), labels)
            
            # loss_pos = F.binary_cross_entropy(labels_pos, l_pos, reduction='mean')
            # loss_neg = F.binary_cross_entropy(labels_neg, l_neg, reduction='mean')
            
            # loss_pos = F.mse_loss(labels_pos, l_pos, reduction='mean')
            # loss_neg = F.mse_loss(labels_neg, l_neg, reduction='mean')

            
            l_pos =  torch.einsum('pe,qe->pq', [emb1, emb1_shuffle])/self.temp
            l_neg = torch.einsum('pe,qe->pq', [emb1, emb2])/self.temp
            
            loss = torch.mean(-torch.logsumexp(l_pos, dim=1)+torch.logsumexp(l_neg, dim=1))
            # loss = torch.mean(torch.logsumexp(torch.cat(-l_pos, l_neg, dim=1), dim=1))
            # dot_product = torch.sum(torch.matmul(emb1, emb1_shuffle.T))
            # magnitude1 = torch.sqrt(torch.sum(emb1 ** 2))
            # magnitude2 = torch.sqrt(torch.sum(emb1_shuffle ** 2))
            # l_pos =  dot_product / (magnitude1 * magnitude2)
            
            

            # dot_product = torch.sum(torch.matmul(emb1, emb2.T))
            # magnitude1 = torch.sqrt(torch.sum(emb1 ** 2))
            # magnitude2 = torch.sqrt(torch.sum(emb2 ** 2))
            # l_neg =  dot_product / (magnitude1 * magnitude2)
            
            
            # loss = -loss_pos + loss_neg            
            # loss = F.binary_cross_entropy(F.sigmoid(logits), labels)
            # loss = F.cross_entropy((logits), labels)
            # logits = torch.cat((l_pos, l_neg), dim=1)
            # logits = torch.cat((l_pos, l_neg.unsqueeze(1)), dim=0)
            # logits /= self.temp
            # labels = torch.zeros((logits.shape[0],), dtype=torch.long).to(l_pos.device)
            # labels = torch.zeros((logits.shape), dtype=torch.float).to(l_pos.device)
            # labels[l_pos.size(0):]=1        
            # loss = F.cross_entropy(logits, labels)
            # apr
            # loss = F.binary_cross_entropy_with_logits(logits, labels)
            # self.log(f'pretrain/loss_poss', loss_pos, on_epoch=True)
            # self.log(f'pretrain/loss_neg', loss_neg, on_epoch=True)
            # self.log(f'pretrain/loss', loss, on_epoch=True)
            
            # running_loss = 0*loss_pos+ 1*loss_neg
            running_loss += loss
            
            
        # self.log(f'pretrain/loss', running_loss, on_epoch=True)    
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
