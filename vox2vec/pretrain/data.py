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

import scipy.ndimage

from connectome import Chain, Transform, Filter, Apply, GroupBy, Merge, CacheToDisk

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

import numpy as np
import scipy.ndimage

import nibabel as nib
import time

def save_nii(patch, name):
    print('------------------', patch.shape)
    save_dir='/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/vox2vec/dataset_check_save_dir/'+'_'
    nii_img = nib.Nifti1Image(patch, affine=np.eye(4))
    nib.save(nii_img, save_dir+name+'.nii.gz')


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
        try:
            args = [*self.load_example(self.ids[i]), self.patch_size,
                self.window_hu, self.min_window_hu, self.max_window_hu,
                self.max_num_voxels_per_patch]
        
        except ValueError:
                args = [*self.load_example(self.ids[i+1]), self.patch_size,
                self.window_hu, self.min_window_hu, self.max_window_hu,
                self.max_num_voxels_per_patch]
        

        views = [sample_views(*args) for _ in range(self.batch_size)]

        # patches_1, patches_1_positive, anchor_voxel_1, positive_voxels, negative_voxels, shifts = zip(*views)
        patches_1, patches_1_positive, anchor_voxel_1, _, _, _ = zip(*views)
        
        
        patches_1 = torch.tensor(np.stack([p[None] for p in patches_1]))
        patches_1_positive = torch.tensor(np.stack([p[None] for p in patches_1_positive]))

        # positive_voxels = torch.stack([torch.tensor(voxels) for voxels in positive_voxels])
        # negative_voxels = torch.stack([torch.tensor(voxels) for voxels in negative_voxels])
        anchor_voxel_1 = torch.stack([torch.tensor(voxels) for voxels in anchor_voxel_1])
        
        # return patches_1, patches_1_positive, anchor_voxel_1, positive_voxels, negative_voxels #, shifts
        return patches_1, patches_1_positive, anchor_voxel_1, _, _ #, shifts


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
    patch_1, roi_voxels_1_dict, patch_1_positive = sample_view(image, roi_voxels, anchor_voxel, patch_size,
                                         window_hu, min_window_hu, max_window_hu)
    
    roi_voxels_1 = roi_voxels_1_dict.get('voxels_shifted')
    
    
    # patch_1_positive = MyAugmentation(patch_1_positive)
    
    # patch_1_positive, adjusted_voxels = random_rotation(patch_1_positive, roi_voxels_1)
    

    valid = np.all((roi_voxels_1 >= 0) & (roi_voxels_1 < patch_size), axis=1)
    # valid_2 = np.all((roi_voxels_2 >= 0) & (roi_voxels_2 < patch_size), axis=1)

    
    # valid_2 = np.all((adjusted_voxels >= 0) & (adjusted_voxels < patch_size), axis=1)
    # valid_2 = True
    # valid = valid_1 & valid_2

    assert valid.any()
    indices = np.where(valid)[0]
    # assert indices.shape[0] ==524288

    all_coords = roi_voxels_1[indices]




    # selected_voxels = roi_voxels_1[indices]
    # patch_1_positive = MyAugmentation(patch_1_positive)
    # adjusted_voxels = roi_voxels_1
    # patch_1_positive, adjusted_voxels = random_rotation(patch_1_positive, roi_voxels_1)
    # num_negative = max_num_voxels
    num_neighbors = max_num_voxels
    # anchor_id = random.choice(np.arange(len(indices)-num_negative-1))
    # positive_voxels = roi_voxels_1[indices[anchor_id:anchor_id+num_neighbors]]
    # anchor_voxels =  roi_voxels_1[indices[anchor_id:anchor_id+1]]
    # distances = np.linalg.norm(all_coords[:, :2] - anchor_voxels[:, :2], axis=1)
    # distances = np.linalg.norm(all_coords - anchor_voxels, axis=1)
    # pos_radius = 10
    # neg_radius = 20
    # positive_voxels = all_coords[distances <= pos_radius]
    # negative_voxels = all_coords[distances>=neg_radius]
    

    if indices.shape[0] > num_neighbors:
        anchor_voxels = all_coords[np.random.choice(indices.shape[0], num_neighbors, replace=False)]
    else:
        anchor_voxels = all_coords[np.random.choice(indices.shape[0], num_neighbors, replace=True)]

        # negative_voxels = all_coords[np.random.choice(indices.shape[0], num_negative, replace=False)]
        # positive_voxels = anchor_voxels
        # patch_1_positive, positive_voxels = random_rotation(patch_1_positive, anchor_voxels)

        # positive_voxels = positive_voxels[np.random.choice(positive_voxels.shape[0], num_neighbors, replace=False)]
        # negative_voxels = negative_voxels[np.random.choice(negative_voxels.shape[0], num_negative, replace=False)]    
        
        # negative_voxels = roi_voxels_1[np.random.choice(indices[(len(indices))//2:], num_negative, replace=False)]    

        # positive_voxels, adjusted_voxels = random_rotation(patch_1_positive, positive_voxels)

        # shift = roi_voxels_1_dict.get('shift')
        
    

        
        # return patch_1, patch_1_positive, roi_voxels_1_1[indices], roi_voxels_1_2[indices]
        
        # return patch_1, patch_1_positive, anchor_voxels, positive_voxels, negative_voxels, shift
        # return patch_1, patch_1_positive, anchor_voxels, positive_voxels, 0, 0
    return patch_1, patch_1_positive, anchor_voxels, 0, 0, 0


def sample_view(image, voxels, anchor_voxel, patch_size, window_hu, min_window_hu, max_window_hu):
# def sample_view(image, voxels, anchor_voxel, patch_size, window_hu, min_window_hu, max_window_hu, rotate_angle):
    assert image.ndim == 3

    # spatial augmentations: random rescale, rotation and crop
    box = sample_box(image.shape, patch_size, anchor_voxel)
    image = crop_to_box(image, box, axis=(-3, -2, -1))
    image_positive = copy.copy(image)

    # image_positive = np.apply_along_axis(np.random.permutation, axis=2, arr=image_positive)
    
    shift = box[0]
    voxels_dict = {}
    voxels = voxels - shift
    voxels_dict['voxels_shifted'] = voxels
    voxels_dict['shift'] = shift
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


    
    # voxels={}
    return image, voxels_dict, image_positive





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





def rotate_patch(patch, rotation_type):
    
    rotation_matrices = {
        0: np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),   
        1: np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  
        2: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),   
        3: np.array([[0, 1, 0], [0, 0, 1], [-1, 0, 0]]),   
        4: np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),  
        5: np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]]),  
    }

    rotation_matrix = rotation_matrices[rotation_type]
    
    
    rotated_patch = scipy.ndimage.affine_transform(
        patch, 
        rotation_matrix, 
        offset=0, 
        order=1, 
        mode='nearest'
    )
    
    return rotated_patch

def random_rotation(patch, voxels):
    
    center = np.array(patch.shape) / 2
    centered_voxels = voxels - center
    rotation_type = np.random.randint(0, 6)
    rotated_patch = rotate_patch(patch, rotation_type)
    

    rotation_matrices = {
        0: np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),   
        1: np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  
        2: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),   
        3: np.array([[0, 1, 0], [0, 0, 1], [-1, 0, 0]]),   
        4: np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),  
        5: np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]]),  
    }
    
    
    rotation_matrix = rotation_matrices[rotation_type]
    adjusted_voxels = np.dot(centered_voxels, rotation_matrix.T)
    
    
    adjusted_voxels += center
    adjusted_voxels = np.round(adjusted_voxels).astype(int)
   
    adjusted_voxels = np.clip(adjusted_voxels, 0, np.array(rotated_patch.shape) - 1)
    
    return rotated_patch, adjusted_voxels