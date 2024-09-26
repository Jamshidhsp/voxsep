from typing import *
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from vox2vec.nn import Lambda
from skimage.metrics import structural_similarity as ssim
import math
import matplotlib.pyplot as plt
import time
from copy import deepcopy
proj_dim = 128
embed_dim = 512


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class Vox2Vec(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        base_channels: int,
        num_scales: int,
        proj_dim: int = proj_dim,
        temp: float = 0.5,
        lr: float = 3e-4,
        output_size=(1, 1, 1),
        hidden_dim: int = 64  
    ):
        super().__init__()
        self.backbone = backbone
        # self.backbone_key = deepcopy(backbone) 
        # self.backbone_key.training=False
        self.proj_dim = proj_dim
        self.temperature = 0.2
        self.lr = 3e-4
        self.num_scales = num_scales
        # self.num_scales = 1
        self.output_size = output_size
        self.num_negatives = 4
        
        self.alpha = torch.nn.Parameter(torch.tensor(0.6))
        
        self.adaptive_pooling_layers = nn.ModuleList(
            [nn.AdaptiveAvgPool3d(output_size) for _ in range(num_scales)]
        )

        self.proj_head = nn.Sequential(
            nn.Linear(1008, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
            # Lambda(F.normalize)
        )
        # self.proj_head
        # self.proj_head_key = deepcopy(self.proj_head)
        # self.proj_head_key.training=False

        self.similarity_predictor = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
            nn.Linear(64, 1),
            # nn.ReLU()
            nn.Sigmoid() 
        )
        self.decoder = nn.Sequential(
            nn.Linear(proj_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 262144),
            nn.Sigmoid()
        )
    
        self.discriminator = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, 1), 
            nn.Sigmoid()
        )


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
        normalized_features = F.normalize(flattened_features, p=2, dim=-1)
        return normalized_features
    

    # def _vox_to_vec_key(self, patches: torch.Tensor) -> torch.Tensor:
    #     feature_pyramid = self.backbone_key(patches)
    #     concatenated_features = self.extract_features_across_scales(feature_pyramid)
    #     flattened_features = concatenated_features.view(concatenated_features.size(0), -1)
    #     return flattened_features

        

    def generate_negatives(self, patches, num_negatives=4):
        negatives = []
        scores = []
        max_num_swaps =100
        for patch_idx in range(patches.size(0)):  
            for i in range(num_negatives):
                negative_patch = patches[patch_idx].clone().detach()
                # print(negative_patch.min(), negative_patch.max())
                num_swaps = np.random.randint(1, max_num_swaps)
                max_swap_size = int(0.9 * patches.size(-1))  
                min_swap_size = int(0.1 * patches.size(-1))

                if num_swaps==0:
                    negatives.append(negative_patch)
                    scores.append(torch.tensor(0.99))
                else:
                    for _ in range(num_swaps):
                    
                        size = np.random.randint(min_swap_size, max_swap_size)
                        # size = 2
                        src_h, src_w, src_d = np.random.randint(0, patches.size(-1) - size, 3)
                        des_h, des_w, des_d = np.random.randint(0, patches.size(-1) - size, 3)
                        negative_patch[:, src_h:src_h+size, src_w:src_w+size, src_d:src_d+size] = \
                            negative_patch[:, des_h:des_h+size, des_w:des_w+size, des_d:des_d+size]
                        
                        # negative_patch[:, src_h:src_h+size, src_w:src_w+size, src_d:src_d+size] = 0.
                        

                score, _ = ssim(
                    patches[patch_idx][0].detach().cpu().numpy(),
                    negative_patch[0].detach().cpu().numpy(),
                    data_range=1.0,  # Normalized data
                    full=True,
                    multichannel=False
                )
                negatives.append(negative_patch)
                scores.append(torch.tensor(score).float())
        negatives = torch.stack(negatives)
        # plt.imshow(torch.cat([(negatives[i,0, :, :, 16]) for i in range(negatives.shape[0])] , dim=1).detach().cpu().numpy())
        # plt.savefig(str(time.time())+'.png')
        scores = torch.stack(scores)
        return negatives, scores

    def training_step(self, batch, batch_idx):
        patches_1,_, _, _, _ = batch['pretrain']
        # plt.imshow(patches_1[0, 0, :, :, 16].detach().cpu().numpy())
        # plt.savefig('patches_1'+str(time.time())+'.png')
        # plt.imshow(torch.cat((patches_1[0, 0, :, :, 16],patches_1_positive[0, 0, :, :, 16]), dim=1).detach().cpu().numpy())
        # plt.savefig('patches_1_positive'+str(time.time())+'.png')
        num_negatives = self.num_negatives
        negative_patches, scores = self.generate_negatives(patches_1, num_negatives)

        # with torch.no_grad():
        anchor_backbone_features = self._vox_to_vec(patches_1)
        anchor_embeddings = self.proj_head(anchor_backbone_features)  
        
        negative_embeddings = self.proj_head(self._vox_to_vec(negative_patches))  
        
        # reconstructed_features = self.decoder(anchor_embeddings)
        # original_patches = patches_1.view(patches_1.size(0), -1)
        # original_features = anchor_backbone_features.view(anchor_backbone_features.size(0), -1)
        
        # reconstruction_loss = F.mse_loss(reconstructed_features, original_features)
        
        temp_neg = F.normalize(negative_embeddings, p=2, dim=1)
        negative_embeddings_similarity = torch.matmul(temp_neg, temp_neg.T)
        negative_embeddings_similarity.fill_diagonal_(float('-inf'))
        
        self_anchor_similarity = torch.matmul(anchor_embeddings, anchor_embeddings.T)
        self_anchor_similarity.fill_diagonal_(float('-inf'))

        anchor_embeddings_expanded = anchor_embeddings.unsqueeze(1).repeat(1, num_negatives, 1)
        negative_embeddings_reshaped = negative_embeddings.view(-1, num_negatives, self.proj_dim)

        # delta_embeddings = torch.cat((anchor_embeddings_expanded, negative_embeddings_reshaped), dim=-1)
        delta_embeddings = anchor_embeddings_expanded - negative_embeddings_reshaped
        # delta_embeddings = anchor_embeddings_expanded + negative_embeddings_reshaped
        predictor_input = delta_embeddings.view(-1, delta_embeddings.size(-1))
        # predictor_input = delta_embeddings.view(-1, proj_dim)
        
        # predicted_scores = self.similarity_predictor(predictor_input).squeeze()
              
        predicted_scores = F.cosine_similarity(anchor_embeddings_expanded, delta_embeddings, dim=-1)
        
        # actual_scores = scores.view(-1).to(predicted_scores.device)
        actual_scores = scores.view(-1, num_negatives).to(predicted_scores.device)

        print(predicted_scores)
        print(actual_scores)
        loss = (F.l1_loss(predicted_scores, actual_scores))
        # loss = F.binary_cross_entropy(predicted_scores,actual_scores)
        
        
        # discriminator_input = torch.cat((anchor_embeddings, negative_embeddings), dim=0)
        # discriminator_output = self.discriminator(discriminator_input).squeeze()
        # discriminator_labels = torch.cat((torch.ones(anchor_embeddings.size(0)), torch.zeros(negative_embeddings.size(0))), dim=0).to(anchor_embeddings.device)
        # loss_discriminator = F.binary_cross_entropy(discriminator_output, discriminator_labels)




        # anchor_negative = torch.einsum('nc,nkc->nk', anchor_embeddings, negative_embeddings)
        # anchor_negative = torch.logsumexp(anchor_negative/0.2, dim=1)

    
        # anchor_negative_loss = torch.mean(anchor_negative)
        # self.log('negative_anchor', anchor_negative_loss)

        # negative_negative_loss = torch.mean(torch.logsumexp(negative_embeddings_similarity, dim=1))
        # self.log('negative_negative_loss', negative_negative_loss)
        # self_anchor_contrastive_loss = torch.mean(self_anchor_similarity)
        # self.log('self_contrastive_loss', self_anchor_contrastive_loss)
        # self.log('delta_mean', delta_embeddings.mean())
        # self.log('anchor_embedding_mean', anchor_embeddings.mean())
        # self.log('anchor_embedding_std', anchor_embeddings.var())
        # self.log('negative_embedding_std', negative_embeddings.var())

    
    # Total loss
        # total_loss = predictor_loss + self.contrastive_weight * contrastive_loss
        # 
        
        # predicted_scores = self.similarity_predictor(predictor_input).squeeze()
        # predicted_scores = self.similarity_predictor(delta_embeddings) #shpae: (bs, num_negatives)
        

        # # actual_scores = scores.view(-1).to(predicted_scores.device)
        # actual_scores = scores.view(-1, num_negatives).to(predicted_scores.device)

        if batch_idx%50==0:
            print((predicted_scores))
            print((actual_scores))
            # print('loss_discriminator', loss_discriminator)
        # loss = torch.mean(torch.abs(10.*(actual_scores-predicted_scores))) #+ 0.1*negative_negative_loss
        # loss = F.l1_loss(predicted_scores, actual_scores)+F.mse_loss(predicted_scores, actual_scores)  +1 - self_contrastive_loss
        # loss = (F.l1_loss(predicted_scores, actual_scores)) #+ 0.5*reconstruction_loss#+ 0.1*contrastive_loss     
        # loss = (F.binary_cross_entropy(predicted_scores, actual_scores))  
        # kl_loss = (F.kl_div(F.log_softmax(predicted_scores.T, dim=-1), F.log_softmax(actual_scores, dim=-1)))
        # kl_loss = F.cross_entropy(predicted_scores.view(actual_scores.shape), actual_scores)
        
        # loss = kl_loss + negative_negative_loss 
        # print(kl_loss.item(), negative_negative_loss.item())
        self.log('train_loss', loss)
        # embedding_norms = anchor_embeddings.norm(p=2, dim=1).mean()
        # anchor_norm_variance = anchor_embeddings.var()
        # negative_mean_norm = negative_embeddings.mean()
        # negative_norm_variance = negative_embeddings.var()
    
        # self.log('embedding_norms', embedding_norms)
        # self.log('anchor_norm_variance', anchor_norm_variance)
        # self.log('negative_mean_norm', negative_mean_norm)
        # self.log('negative_norm_variance', negative_norm_variance)
            
        return loss #+ loss_discriminator


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def validation_step(self, batch, batch_idx):
        pass

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not self.trainer.training:
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)