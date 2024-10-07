import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from skimage.metrics import structural_similarity as ssim

proj_dim = 128
embed_dim = 512

# Placeholder for SwinViT and other required classes
# Make sure to replace these placeholders with your actual implementations

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=2048, out_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, input):
        if torch.is_tensor(input):
            x = input
        else:
            x = input[-1]
            b = x.size()[0]
            x = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(b, -1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

class Vox2Vec(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        base_channels: int,
        num_scales: int,
        proj_dim: int = proj_dim,
        temp: float = 0.5,
        lr: float = 3e-3,
        output_size=(1, 1, 1),
        hidden_dim: int = 64
    ):
        super().__init__()
        self.backbone = backbone
        self.proj_dim = proj_dim
        self.temperature = temp
        self.lr = lr
        self.num_scales = num_scales
        self.output_size = output_size
        self.num_negatives = 4

        self.adaptive_pooling_layers = nn.ModuleList(
            [nn.AdaptiveAvgPool3d(output_size) for _ in range(num_scales)]
        )

        self.proj_head = nn.Sequential(
            nn.Linear(1008, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            nn.Linear(256, proj_dim),
        )

        self.similarity_predictor = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.BatchNorm1d(64),
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
        flattened_features = concatenated_features.contiguous().view(concatenated_features.size(0), -1)
        normalized_features = F.normalize(flattened_features, p=2, dim=-1)
        return normalized_features

    def generate_negatives(self, patches, num_negatives=4):
        negatives = []
        scores = []
        max_num_swaps = 100
        batch_size = patches.size(0)
        for patch_idx in range(batch_size):
            for _ in range(num_negatives):
                negative_patch = patches[patch_idx].clone().detach()
                num_swaps = np.random.randint(1, max_num_swaps)
                max_swap_size = int(0.9 * patches.size(-1))
                min_swap_size = int(0.1 * patches.size(-1))

                for _ in range(num_swaps):
                    size = np.random.randint(min_swap_size, max_swap_size)
                    src_h, src_w, src_d = np.random.randint(0, patches.size(-1) - size, 3)
                    des_h, des_w, des_d = np.random.randint(0, patches.size(-1) - size, 3)
                    # negative_patch[:, src_h:src_h+size, src_w:src_w+size, src_d:src_d+size] = \
                        # negative_patch[:, des_h:des_h+size, des_w:des_w+size, des_d:des_d+size]
                    
                    negative_patch[:, src_h:src_h+size, src_w:src_w+size, src_d:src_d+size] = 0.

                score, _ = ssim(
                    patches[patch_idx][0].detach().cpu().numpy(),
                    negative_patch[0].detach().cpu().numpy(),
                    data_range=1.0,
                    full=True,
                    multichannel=False
                )
                negatives.append(negative_patch)
                scores.append(torch.tensor(score).float())

        negatives = torch.stack(negatives)
        scores = torch.stack(scores)
        return negatives, scores

    def training_step(self, batch, batch_idx):
        patches_1, _, _, _, _ = batch['pretrain']
        batch_size = patches_1.size(0)
        num_negatives = self.num_negatives

        # Generate negatives and scores
        negative_patches, scores = self.generate_negatives(patches_1, num_negatives)

        # Compute embeddings
        anchor_embeddings = self.proj_head(self._vox_to_vec(patches_1))  # Shape: [B, D]
        negative_embeddings = self.proj_head(self._vox_to_vec(negative_patches))  # Shape: [B*N, D]


        # Normalize embeddings
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)

        # Prepare target labels
        total_negatives = batch_size * num_negatives
        target_labels = torch.zeros(batch_size, total_negatives, device=anchor_embeddings.device)
        for i in range(batch_size):
            start_idx = i * num_negatives
            end_idx = start_idx + num_negatives
            # Set labels to 1 for own negatives
            target_labels[i, start_idx:end_idx] = scores.view(-1, self.num_negatives)[i]
            # For other negatives, labels remain 0

        # Compute differences between each anchor and all negatives
        anchor_embeddings_expanded = anchor_embeddings.unsqueeze(1)  # Shape: [B, 1, D]
        negative_embeddings_expanded = negative_embeddings.unsqueeze(0)  # Shape: [1, B*N, D]
        differences = anchor_embeddings_expanded - negative_embeddings_expanded  # Shape: [B, B*N, D]
        differences_flat = differences.view(-1, self.proj_dim)  # Shape: [B * B*N, D]

        # Predict scores
        predicted_scores_flat = self.similarity_predictor(differences_flat).squeeze()  # Shape: [B * B*N]
        predicted_scores = predicted_scores_flat.view(batch_size, total_negatives)  # Shape: [B, B*N]
        # print(predicted_scores)
        # print(target_labels)
        # Compute loss using Binary Cross-Entropy
        # loss = F.binary_cross_entropy(predicted_scores, target_labels)
        # loss = F.mse_loss(predicted_scores, target_labels)
        loss = F.l1_loss(predicted_scores, target_labels)
        # Optional: Add self-anchor similarity loss
        # Compute cosine similarities between anchor embeddings
        # print('anchor_embeddings.shape', anchor_embeddings.shape)
        anchor_similarity = torch.mm(anchor_embeddings, anchor_embeddings.T)
        # Exclude self-similarity
        mask = torch.eye(batch_size, device=anchor_embeddings.device).bool()
        anchor_similarity = anchor_similarity.masked_fill(mask, float('-inf'))
        # Compute self-anchor similarity loss
        self_anchor_similarity_loss = torch.mean(torch.logsumexp(anchor_similarity / self.temperature, dim=1))
        self.log('self_anchor_similarity_loss', self_anchor_similarity_loss)

        # Total loss
        # total_loss = loss + 0.1*self_anchor_similarity_loss
        total_loss = self_anchor_similarity_loss
        total_loss =F.l1_loss(anchor_embeddings, torch.zeros_like(anchor_embeddings))

        self.log('train_loss', total_loss)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.Adam([
        #     {'params': self.backbone.parameters(), 'lr':1e-3},
        #     {'params': list(self.proj_head.parameters())+ list(self.similarity_predictor.parameters()), 'lr':5e-5}
        # ])
        # return optimizer

    def validation_step(self, batch, batch_idx):
        pass

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not self.trainer.training:
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
