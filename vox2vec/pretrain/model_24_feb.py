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

num_positives = 20
num_negative = 20
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

    
    def neighboring_sampling(self, voxels_list):
        
        # num_positives = num_positives
        # num_negative = num_negative
        step = 5
        positive_list = []
        negative_list = []
        
        for i in range(len(voxels_list)):
            
        # random_sample_indice = random.randint(1, voxels.shape[0])
            voxels = voxels_list[i]
            random_sample_indice = random.randint(step, voxels.shape[0]-step)
            voxels_numpy = voxels.cpu().numpy()
            valid_1 = np.all((voxels_numpy >= voxels[random_sample_indice].cpu().numpy()-step) & (voxels_numpy < voxels[random_sample_indice].cpu().numpy()+step), axis=1)
            valid_2 = np.where(valid_1==False)
            valid_1 = np.where(valid_1)
            
            
            valid_1 = valid_1[0]
            valid_1 = np.random.permutation(valid_1)
            positive_voxels = voxels[valid_1[:num_positives]]
            
            
            valid_2 = valid_2[0]
            valid_2 = np.random.permutation(valid_2)
            negative_voxels = voxels[valid_2[:num_negative]]
            
            positive_list.append(positive_voxels)
            negative_list.append(negative_voxels)
            # return positive_voxels, negative_voxels
        return positive_list, negative_list
    
    
    def training_step(self, batch, batch_idx):
        """Computes Info-NCE loss.
        """
        patches_1, patches_2, voxels_1, _ = batch['pretrain']
        # random_sample_indice = random.randint(1, voxels_1[0].shape[0])
        positive_voxels_list, negative_voxels_list = self.neighboring_sampling(voxels_1)
        # print(positive_voxels.shape, negative_voxels.shape)
        assert self.backbone.training
        assert self.proj_head.training
        
        
        

        embeds_1 = self.proj_head(self._vox_to_vec(patches_1, positive_voxels_list))
        embeds_1 = embeds_1.reshape(5, num_positives, 128 )
        embeds_1_positive = self.proj_head(self._vox_to_vec(patches_2, positive_voxels_list))
        embeds_1_positive = embeds_1_positive.reshape(5, num_positives, 128 )
        embeds_2 = self.proj_head(self._vox_to_vec(patches_1, negative_voxels_list))
        embeds_2 = embeds_2.reshape(5, num_negative, 128)
        # l_pos = torch.einsum('nc,nc->n', [embeds_1, embeds_1_positive]).unsqueeze(-1)
        # l_neg = torch.einsum('nc,ck->nk', [embeds_1, embeds_2.T]) 
        # N = l_pos.size(0)
        # logits = torch.cat((l_pos, l_neg), dim=1)
        # logits /= self.temp
        # labels = torch.zeros((N, ), dtype=torch.long).to(l_pos.device)
        # loss = F.cross_entropy(logits, labels)


        # batch = 4
        # samples = 20
        # embeds_1_positive = torch.normal(0,1,(batch,samples,128))
        # embeds_1_negative = torch.normal(0,1,(batch,samples,128))

        pos_mat = torch.einsum('bpd,bjk->bpj', [embeds_1, embeds_1_positive])
        neg_mat = torch.einsum('bpd,bjk->bpj', [embeds_1, embeds_2])
        pos_label = torch.ones(pos_mat.shape).to('cuda')
        neg_label = -1*torch.ones(pos_mat.shape).to('cuda')

        loss1 = F.mse_loss(pos_label, pos_mat)
        loss2 = F.mse_loss(neg_label, neg_mat)

        loss11 = F.cross_entropy(pos_label, pos_mat)
        loss21 = F.cross_entropy(neg_label, neg_mat)
        # net_loss = torch.mean(torch.tensor([loss1, loss2, loss11, loss1]))
        # net_loss2 = torch.sum(torch.tensor([loss1, loss2, loss11, loss1]))

        net_loss = (loss1+loss2)/2



        self.log(f'pretrain/loss1', loss1, on_epoch=True)
        self.log(f'pretrain/loss2', loss2, on_epoch=True)
        self.log(f'pretrain/loss11', loss2, on_epoch=True)
        self.log(f'pretrain/loss21', loss21, on_epoch=True)
        self.log(f'pretrain/net_loss', net_loss, on_epoch=True)

        return (loss1+loss2)/2

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not self.trainer.training:
            # skip device transfer for the val dataloader
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
