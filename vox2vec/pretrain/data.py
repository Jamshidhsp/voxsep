from typing import *
from tqdm import tqdm
import numpy as np
import random
from imops import crop_to_box

import torch
from torch.utils.data import Dataset

from amid.amos import AMOS
from amid.flare2022 import FLARE2022
from amid.nlst import NLST
from amid.lidc import LIDC
from amid.nsclc import NSCLC
from amid.midrc import MIDRC

from connectome import Chain, Transform, Filter, Apply, GroupBy, Merge, CacheToDisk
from vox2vec.pretrain.my_transormations import get_non_overlapping_crops
from vox2vec.pretrain.my_transormations import rot_rand

from vox2vec.processing import (
    LocationsToSpacing, FlipAxesToCanonical, CropToBox, RescaleToSpacing,
    get_body_mask, BODY_THRESHOLD_HU, sample_box, gaussian_filter, gaussian_sharpen,
    scale_hu
)
from vox2vec.utils.box import mask_to_bbox
from vox2vec.utils.misc import is_diagonal
# jamshid was here
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import copy
img_save_dir = "/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/vox2vec/img_save/"


def prepare_nlst_ids(nlst_dir, patch_size):
    nlst = NLST(root=nlst_dir)
    for id_ in tqdm(nlst.ids, desc='Warming up NLST().patient_id method'):
        nlst.patient_id(id_)

    nlst_patients = nlst >> GroupBy('patient_id')
    ids = []
    for patient_id in tqdm(nlst_patients.ids, desc='Preparing NLST ids'):
        id, slice_locations = max(nlst_patients.slice_locations(patient_id).items(), key=lambda i: len(i[1]))
        if len(slice_locations) >= patch_size[2]:
            ids.append(id)

    return ids


class PretrainDataset(Dataset):
    def __init__(
            self,
            cache_dir: str,
            spacing: Tuple[float, float, float],
            patch_size: Tuple[int, int, int],
            window_hu: Tuple[float, float],
            min_window_hu: Tuple[float, float],
            max_window_hu: Tuple[float, float],
            max_num_voxels_per_patch: int,
            batch_size: int,
            amos_dir: Optional[str] = None,
            flare_dir: Optional[str] = None,
            nlst_dir: Optional[str] = None,
            midrc_dir: Optional[str] = None,
            nsclc_dir: Optional[str] = None,
    ) -> None:
        parse_affine = Transform(
            __inherit__=True,
            flipped_axes=lambda affine: tuple(np.where(np.diag(affine[:3, :3]) < 0)[0] - 3),  # enumerate from the end
            spacing=lambda affine: tuple(np.abs(np.diag(affine[:3, :3]))),
        )

        # amos_ct_ids = AMOS(root=amos_dir).ids[:500]
        amos_ct_ids = AMOS(root=amos_dir).ids[:]
        amos = Chain(
            AMOS(root=amos_dir),
            Filter.keep(amos_ct_ids),
            parse_affine,
            FlipAxesToCanonical(),
        )

        pipeline = Chain(
            Merge(
                amos,  # 500 abdominal CTs
            ),  # ~6550 openly available CTs in total, covering abdomen and thorax domains
            # cache spacing
            CacheToDisk.simple('spacing', root=cache_dir),
            Filter(lambda spacing: spacing[-1] is not None, verbose=True),
            CacheToDisk.simple('ids', root=cache_dir),
            # cropping, rescaling
            Transform(__inherit__=True, cropping_box=lambda image: mask_to_bbox(image >= BODY_THRESHOLD_HU)),
            CropToBox(axis=(-3, -2, -1)),
            RescaleToSpacing(to_spacing=spacing, axis=(-3, -2, -1), image_fill_value=lambda x: np.min(x)),
            Apply(image=lambda x: np.int16(x)),
            CacheToDisk.simple('image', root=cache_dir),
            Apply(image=lambda x: np.float32(x)),
            # filtering by shape
            Filter(lambda image: np.all(np.array(image.shape) >= patch_size), verbose=True),
            CacheToDisk.simple('ids', root=cache_dir),
            # adding body_voxels
            Transform(__inherit__=True, body_voxels=lambda image: np.argwhere(get_body_mask(image))),
            CacheToDisk.simple('body_voxels', root=cache_dir),
            Filter(lambda body_voxels: len(body_voxels) > 0, verbose=True),
            CacheToDisk.simple('ids', root=cache_dir),
        )

        self.pipeline = pipeline
        self.ids = pipeline.ids
        self.load_example = pipeline._compile(['image', 'body_voxels'])
        self.patch_size = patch_size
        self.window_hu = window_hu
        self.min_window_hu = min_window_hu
        self.max_window_hu = max_window_hu
        self.max_num_voxels_per_patch = max_num_voxels_per_patch
        self.batch_size = batch_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        args = [*self.load_example(self.ids[i]), self.patch_size,
                self.window_hu, self.min_window_hu, self.max_window_hu,
                self.max_num_voxels_per_patch]
        views = [sample_views(*args) for _ in range(self.batch_size)]
        # patches_1, patches_1_positive, patches_2, voxels_1, voxels_2 = zip(*views)
        # patches_1, patches_1_positive, voxels_1, voxels_2 = zip(*views)
        patches_1, patches_1_positive, anchor_voxel_1, positive_voxels, negative_voxels = zip(*views)
        
        patches_1 = torch.tensor(np.stack([p[None] for p in patches_1]))
        patches_1_positive = torch.tensor(np.stack([p[None] for p in patches_1_positive]))
        
        # patches_1_positive = MyAugmentation(patches_1_positive)
        
        # patches_2 = torch.tensor(np.stack([p[None] for p in patches_2]))
        # voxels_1 = [torch.tensor(voxels) for voxels in voxels_1]
        # voxels_2 = [torch.tensor(voxels) for voxels in voxels_2]



        positive_voxels = [torch.tensor(voxels) for voxels in positive_voxels]
        negative_voxels = [torch.tensor(voxels) for voxels in negative_voxels]
        anchor_voxel_1 =  [torch.tensor(voxels) for voxels in anchor_voxel_1]
        
        disp_0 = torch.cat((patches_1[0, 0, :, :, 16], patches_1_positive[0, 0, :, :, 16]), axis=1)
        disp_0 = disp_0.detach().cpu().numpy()
        img_name = str(time.time())
        plt.imshow(disp_0)
        
        # plt.savefig(img_save_dir+img_name+'.png')
        
        
        
        # return patches_1, patches_1_positive, voxels_1, voxels_2
        return patches_1, patches_1_positive, anchor_voxel_1, positive_voxels, negative_voxels


def sample_views(
        image: np.ndarray,
        roi_voxels: np.ndarray,
        patch_size: Tuple[int, int, int],
        window_hu: Tuple[float, float],
        min_window_hu: Tuple[float, float],
        max_window_hu: Tuple[float, float],
        max_num_voxels: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    anchor_voxel = random.choice(roi_voxels)
    patch_1, roi_voxels_1, patch_1_positive = sample_view(image, roi_voxels, anchor_voxel, patch_size,
                                         window_hu, min_window_hu, max_window_hu)
    
    patch_1_positive = MyAugmentation(patch_1_positive)
    # patch_1_positive = rot_rand(patch_1)
    # patch_2, roi_voxels_2, _ = sample_view(image, roi_voxels, anchor_voxel, patch_size,
    #                                      window_hu, min_window_hu, max_window_hu)
    # patch_2= patch_2.flip(dims=[-1])
    # random_axis = np.random.randint(0, 3)
    # patch_2= np.flip(patch_2, axis=[random_axis])

  

    valid_1 = np.all((roi_voxels_1 >= 0) & (roi_voxels_1 < patch_size), axis=1)
    # valid_2 = np.all((roi_voxels_2 >= 0) & (roi_voxels_2 < patch_size), axis=1)
    valid_2 = True
    valid = valid_1 & valid_2
    assert valid.any()
    indices = np.where(valid)[0]

    num_negative = max_num_voxels
    num_neighbors = 20
    anchor_id = random.choice(np.arange(len(indices)))
    # anchor_id = random.choice(indices[:-num_neighbors])  # (3,)
    positive_voxels = roi_voxels_1[indices[anchor_id:anchor_id+num_neighbors]] 
    negative_voxels = roi_voxels_1[np.random.choice(indices, num_negative, replace=False)]    

    # if len(indices) > max_num_voxels:
    #     indices = np.random.choice(indices, max_num_voxels, replace=False)

    
    # return patch_1, patch_1_positive, roi_voxels_1_1[indices], roi_voxels_1_2[indices]
    return patch_1, patch_1_positive, roi_voxels_1[indices[anchor_id]], positive_voxels, negative_voxels


def sample_view(image, voxels, anchor_voxel, patch_size, window_hu, min_window_hu, max_window_hu):
# def sample_view(image, voxels, anchor_voxel, patch_size, window_hu, min_window_hu, max_window_hu, rotate_angle):
    assert image.ndim == 3

    # spatial augmentations: random rescale, rotation and crop
    box = sample_box(image.shape, patch_size, anchor_voxel)
    image = crop_to_box(image, box, axis=(-3, -2, -1))
    image_positive = copy.copy(image)
    shift = box[0]
    voxels = voxels - shift
    anchor_voxel = anchor_voxel - shift
    # return image, voxels
    
    # intensity augmentations
    
    if random.uniform(0, 1) < 0.5:
        if random.uniform(0, 1) < 0.5:
            # random gaussian blur in axial plane
            sigma = random.uniform(0.5, 1.5)
            image = gaussian_filter(image, sigma, axis=(0, 1))
        else:
            # random gaussian sharpening in axial plane
            sigma_1 = random.uniform(0.5, 1.0)
            sigma_2 = 0.5
            alpha = random.uniform(10.0, 30.0)
            image = gaussian_sharpen(image, sigma_1, sigma_2, alpha, axis=(0, 1))

    if random.uniform(0, 1) < 0.5:
        sigma_hu = random.uniform(0, 30)
        image = image + np.random.normal(0, sigma_hu, size=image.shape).astype('float32')

    if random.uniform(0, 1) < 0.8:
        window_hu = (random.uniform(max_window_hu[0], min_window_hu[0]),
                     random.uniform(min_window_hu[1], max_window_hu[1]))
    # window_hu = [-1350, 1000]
    image = scale_hu(image, window_hu)
    # image_positive = scale_hu(image_positive, window_hu)


    
    
    return image, voxels, image_positive





def MyAugmentation(image):
    window_hu = [-1350, 1000]
    min_window_hu = [-1000, 300]
    max_window_hu = [-1350, 1000]
# def sample_view(image, voxels, anchor_voxel, patch_size, window_hu, min_window_hu, max_window_hu, rotate_angle):
    assert image.ndim == 3
    # intensity augmentations
    if random.uniform(0, 1) < 0.5:
        if random.uniform(0, 1) < 0.5:
            # random gaussian blur in axial plane
            sigma = random.uniform(0.5, 1.5)
            image = gaussian_filter(image, sigma, axis=(0, 1))
        else:
            # random gaussian sharpening in axial plane
            sigma_1 = random.uniform(0.5, 1.0)
            sigma_2 = 0.5
            alpha = random.uniform(10.0, 30.0)
            image = gaussian_sharpen(image, sigma_1, sigma_2, alpha, axis=(0, 1))

    if random.uniform(0, 1) < 0.5:
        sigma_hu = random.uniform(0, 30)
        image = image + np.random.normal(0, sigma_hu, size=image.shape).astype('float32')

    if random.uniform(0, 1) < 0.8:
        window_hu = (random.uniform(max_window_hu[0], min_window_hu[0]),
                     random.uniform(min_window_hu[1], max_window_hu[1]))
    image = scale_hu(image, window_hu)

    
    return image

