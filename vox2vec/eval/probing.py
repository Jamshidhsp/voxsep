from typing import *

import torch
from torch import nn

import pytorch_lightning as pl
import scipy.ndimage as ndimage
import nibabel as nib
import numpy as np
from vox2vec.nn.functional import (
    compute_binary_segmentation_loss, compute_dice_score, eval_mode
)
from .predict import predict


class Probing(pl.LightningModule):
    def __init__(
            self,
            backbone: nn.Module,
            *heads: nn.Module,
            patch_size: Tuple[int, int, int],
            threshold: float = 0.5,
            lr: float = 3e-4,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=['backbone', 'heads'])

        self.backbone = backbone
        self.heads = nn.ModuleList(heads)
        

        self.patch_size = patch_size
        self.threshold = threshold
        self.lr = lr

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        images, rois, gt_masks = batch

        with torch.no_grad(), eval_mode(self.backbone):
            backbone_outputs = self.backbone(images)

        for i, head in enumerate(self.heads):
            pred_logits = head(backbone_outputs)
            loss, logs = compute_binary_segmentation_loss(pred_logits, gt_masks, rois, logs_prefix=f'train/head_{i}_')
            self.log_dict(logs, on_epoch=True, on_step=False)
            self.manual_backward(loss)

        optimizer.step()

    def validation_step(self, batch, batch_idx):
        image, roi, gt_mask = batch
        for i, head in enumerate(self.heads):
            pred_probas = predict(image, self.patch_size, self.backbone, head, self.device, roi)
            dice_scores = compute_dice_score(pred_probas, gt_mask, reduce=lambda x: x)
            for j, dice_score in enumerate(dice_scores):
                self.log(f'val/head_{i}_dice_score_for_cls_{j}', dice_score, on_epoch=True)
            self.log(f'val/head_{i}_avg_dice_score', dice_scores.mean(), on_epoch=True)

    def test_step(self, batch, batch_idx):
        image, roi, gt_mask = batch
        for i, head in enumerate(self.heads):
            pred_probas = predict(image, self.patch_size, self.backbone, head, self.device, roi)
            pred_mask = pred_probas >= self.threshold
            pred_mask = (pred_probas >= self.threshold).float()
            pred_mask = torch.argmax(pred_mask, axis=0)
            test = nib.load('/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/vox2vec/btcv_single_volume/RawData_backup/Training/label/label0001.nii.gz')
            target_size = test.shape
            # segmentation_map = pred_mask.cpu().numpy()
            segmentation_map = self.resample_3d(pred_mask, test.shape)
            nifti_image = nib.Nifti1Image(segmentation_map.astype(np.uint8), test.affine)
            nib.save(nifti_image, '/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/vox2vec/supplementary/'+str(i)+'segmentation_full_size.nii.gz')
            
            
            seg_mask = torch.argmax(gt_mask, axis=0)

            # segmentation_map = pred_mask.cpu().numpy()
            original_seg = self.resample_3d(seg_mask, test.shape)
            nifti_image = nib.Nifti1Image(original_seg.astype(np.uint8), test.affine)
            nib.save(nifti_image, '/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/vox2vec/supplementary/'+str(i)+'original_segmentation.nii.gz')
            
            
            
            # img = image[0].cpu().numpy
            image_ = self.resample_3d(100*image[0], test.shape)
            nifti_image = nib.Nifti1Image(image_.astype(np.uint8), test.affine)
            nib.save(nifti_image, '/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/vox2vec/supplementary/'+str(i)+'heimage.nii.gz')
            
            
            
        
            
            
            
            dice_scores = compute_dice_score(pred_mask, gt_mask, reduce=lambda x: x)
            for j, dice_score in enumerate(dice_scores):
                self.log(f'test/head_{i}_dice_score_for_cls_{j}', dice_score, on_epoch=True)
            self.log(f'test/head_{i}_avg_dice_score', dice_scores.mean(), on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not self.trainer.training:
            # skip device transfer for the val and test dataloaders
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
    
    def resample_3d(self, img, target_size):
        imx, imy, imz = img.shape
        tx, ty, tz = target_size
        zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
        img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
        return img_resampled

